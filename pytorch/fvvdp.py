from numpy.lib.shape_base import expand_dims
import torch
import torch.nn.functional as Func
import numpy as np 
import os
import sys
import argparse
import time
import math
import torch.utils.benchmark as torchbench
import logging

# For debugging only
# from pfs import pfs

from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from third_party.cpuinfo import cpuinfo
from hdrvdp_lpyr_dec import hdrvdp_lpyr_dec
from interp import interp1, interp3
from utils import *
from fvvdp_test import FovVideoVDP_Testbench

class DisplayModel():
    def __init__(self, W, H, diagonal_fov_degrees, distance_m, diag_size_m, min_luminance, max_luminance, gamma_func, rgb2X, rgb2Y, rgb2Z):
        super(DisplayModel, self).__init__()
        self.W = W
        self.H = H
        self.deg2rad = np.pi / 180.0
        self.rad2deg = 180.0 / np.pi

        assert (diagonal_fov_degrees is not None) or (distance_m is not None and diag_size_m is not None)

        self.diagonal_fov_degrees = diagonal_fov_degrees
        self.distance_m = distance_m
        self.diag_size_m = diag_size_m

        if self.diagonal_fov_degrees is None:
            self.diagonal_fov_degrees = self.rad2deg * 2.0 * math.atan(self.diag_size_m * 0.5 / self.distance_m)

        # compute hfov, vfov
        self.hfov_degrees = self.rad2deg * 2.0 * math.atan(
            math.tan(self.diagonal_fov_degrees * self.deg2rad * 0.5) * math.cos(math.atan(float(self.H)/float(self.W))))
        self.vfov_degrees = self.rad2deg * 2.0 * math.atan(
            math.tan(self.diagonal_fov_degrees * self.deg2rad * 0.5) * math.sin(math.atan(float(self.H)/float(self.W))))

        self.min_luminance = min_luminance
        self.max_luminance = max_luminance

        if gamma_func == 'sRGB':
            self.apply_gamma_correction = lambda x: srgb2linear_torch(x)
            self.apply_gamma            = lambda x: linear2srgb_torch(x)
        elif float(gamma_func) > 0.0:
            self.apply_gamma_correction = lambda x: torch.pow(x, float(gamma_func))
            self.apply_gamma            = lambda x: torch.pow(x, 1.0/float(gamma_func))
        else:
            logging.error("Error: Gamma function '%s' not recognized!" % gamma_func)

        self.rgb2X = rgb2X
        self.rgb2Y = rgb2Y
        self.rgb2Z = rgb2Z

        # # calculate display size
        # distance_px = math.sqrt(W*W + H*H) / (2.0 * math.tan(diagonal_fov_degrees * 0.5 * self.deg2rad))
        # height_deg = self.rad2deg * math.atan( H / (2.0 * distance_px) ) * 2.0 
        # height_m = 2 * math.tan( self.deg2rad * height_deg / 2.0 ) * self.distance_m
        # self.display_size_m = [height_m * W / H, height_m]
        # # print(vars(self))

    @classmethod
    def load(cls, model_name, color_space_name):
        models_file      = os.path.join(os.path.dirname(__file__), "../fvvdp_data/display_models.json")
        colorspaces_file = os.path.join(os.path.dirname(__file__), "../fvvdp_data/color_spaces.json")

        obj = cls.__new__(cls)
        obj.display_name = model_name

        models = json2dict(models_file)
        colorspaces = json2dict(colorspaces_file)

        if color_space_name is None or color_space_name not in colorspaces:
            color_space_name = "sRGB"

        logging.info("Using color space %s" % color_space_name)

        color_space = colorspaces[color_space_name]

        assert "gamma" in color_space and "RGB2Y" in color_space

        gamma_func = color_space["gamma"]
        rgb2Y      = color_space["RGB2Y"]

        if "RGB2X" in color_space: rgb2X = color_space["RGB2X"]
        else:                      rgb2X = None

        if "RGB2Z" in color_space: rgb2Z = color_space["RGB2Z"]
        else:                      rgb2Z = None

        for mk in models:
            #if models[mk]["name"] == model_name:
            if mk == model_name:
                model = models[mk]
                assert "resolution" in model

                W, H = model["resolution"]

                inches_to_meters = 0.0254

                if "fov_diagonal" in model: fov_diagonal = model["fov_diagonal"]
                else:                       fov_diagonal = None

                if   "viewing_distance_meters" in model: distance_m = model["viewing_distance_meters"]
                elif "viewing_distance_inches" in model: distance_m = model["viewing_distance_inches"] * inches_to_meters
                else:                                    distance_m = None

                if   "diagonal_size_meters" in model: diag_size_m = model["diagonal_size_meters"]
                elif "diagonal_size_inches" in model: diag_size_m = model["diagonal_size_inches"] * inches_to_meters
                else:                                 diag_size_m = None

                if "max_luminance" in model and "min_luminance" in model:
                    max_luminance = model["max_luminance"]
                    min_luminance = model["min_luminance"]
                elif "max_luminance" in model and "contrast" in model:
                    max_luminance = model["max_luminance"]
                    min_luminance = max_luminance / model["contrast"]
                else:
                    max_luminance = 200.0
                    min_luminance = 0.1

                # Ambient light
                if "E_ambient" in model:
                    E_ambient = model["E_ambient"]
                else:
                    E_ambient = 0
                
                # Reflectivity of the display panel
                if "k_refl" in model: 
                    k_refl = model["k_refl"]
                else:
                    k_refl = 0.005

                # Add screen reflections
                min_luminance += E_ambient*k_refl/math.pi

                obj.__init__(W, H, fov_diagonal, distance_m, diag_size_m, min_luminance, max_luminance, gamma_func, rgb2X, rgb2Y, rgb2Z)

                return obj

        logging.error("Error: Display model '%s' not found in display_models.json" % model_name)
        return None


    def get_ppd(self, ecc_deg = 0.0):

        diagonal_fov_rad = self.diagonal_fov_degrees * self.deg2rad
        ecc_rad = ecc_deg * self.deg2rad

        # we only consider horizontal axis
        eyedist_in_pixels = math.sqrt(self.W*self.W + self.H*self.H) / (2.0 * math.tan(diagonal_fov_rad * 0.5))

        ecc_in_pixels = eyedist_in_pixels * math.tan(ecc_rad)

        pix_begin_deg = math.atan( (ecc_in_pixels - 0.5)/(eyedist_in_pixels) ) * self.rad2deg
        pix_end_deg   = math.atan( (ecc_in_pixels + 0.5)/(eyedist_in_pixels) ) * self.rad2deg

        dpp = pix_end_deg - pix_begin_deg

        return (1.0 / dpp)

    def get_resolution_magnification(self, eccentricity):
        
        M = torch.ones_like(eccentricity, device=eccentricity.device)
        
        # pixel size in the centre of the display
        # pix_deg = 2.0 * self.rad2deg * torch.atan(torch.tensor(0.5*self.display_size_m[0]/(self.W * self.distance_m)))
        pix_deg = 2.0 * self.rad2deg * torch.atan(torch.tensor(math.tan(self.deg2rad * 0.5 * self.hfov_degrees)/self.W))
        
        delta = pix_deg / 2.0
        tan_delta = torch.tan(delta * self.deg2rad)
        tan_a = torch.tan(eccentricity * self.deg2rad)
        
        M = (torch.tan((eccentricity + delta) * self.deg2rad) - tan_a) / tan_delta

        return M

    def rgb_to_X(self, x_linear):
        assert self.rgb2X is not None
        return (
            x_linear[:,0:1,...] * self.rgb2X[0] +
            x_linear[:,1:2,...] * self.rgb2X[1] +
            x_linear[:,2:3,...] * self.rgb2X[2])

    def rgb_to_Y(self, x_linear):
        return (
            x_linear[:,0:1,...] * self.rgb2Y[0] +
            x_linear[:,1:2,...] * self.rgb2Y[1] +
            x_linear[:,2:3,...] * self.rgb2Y[2])

    def rgb_to_Z(self, x_linear):
        assert self.rgb2Z is not None
        return (
            x_linear[:,0:1,...] * self.rgb2Z[0] +
            x_linear[:,1:2,...] * self.rgb2Z[1] +
            x_linear[:,2:3,...] * self.rgb2Z[2])

    def rgb_to_XYZ(self, x_linear):
        XYZ = torch.zeros_like(x_linear)

        r = x_linear[:,0:1,...]
        g = x_linear[:,1:2,...] 
        b = x_linear[:,2:3,...]

        XYZ[:,0:1] = r*self.rgb2X[0] + g*self.rgb2X[1] + b*self.rgb2X[2]
        XYZ[:,1:2] = r*self.rgb2Y[0] + g*self.rgb2Y[1] + b*self.rgb2Y[2]
        XYZ[:,2:3] = r*self.rgb2Z[0] + g*self.rgb2Z[1] + b*self.rgb2Z[2]

        return XYZ

    def f(self, t):
        delta = 6./29.
        return torch.where(t > (delta*delta*delta), torch.pow(t, 0.333), t / (3.0 * delta * delta) + 4./29.)

    def rgb_to_Lab(self, x_linear):

        XYZ = self.rgb_to_XYZ(x_linear)

        # assumption: D65

        XbyXn = torch.div(XYZ[:,0:1], 0.950489)
        YbyYn = torch.div(XYZ[:,1:2], 1.000000)
        ZbyZn = torch.div(XYZ[:,2:3], 1.088840)

        Lab = torch.zeros_like(x_linear)

        Lab[:,0:1] = 1.16 * (self.f(YbyYn)) - 0.16
        Lab[:,1:2] = 5.00 * (self.f(XbyXn) - self.f(YbyYn))
        Lab[:,2:3] = 2.00 * (self.f(YbyYn) - self.f(ZbyZn))

        return Lab

    def Y_to_luminance(self, Y):
        return self.min_luminance + Y * (self.max_luminance - self.min_luminance)


    # de-gamma, convert to luminance 0..1, convert to real units 
    def get_luminance_pytorch(self, x):

        return self.Y_to_luminance(self.rgb_to_Y(self.apply_gamma_correction(x)))


class TestLoss(torch.nn.Module):
    def __init__(self, display_model, device, epsilon):
        
        super(TestLoss, self).__init__()

        self.display_model = display_model
        self.device        = device
        self.pix_per_deg   = self.display_model.get_ppd(0.0)
        self.epsilon       = epsilon

        mode = "blur2_adapt"

        self.mode          = mode

        if mode == "blur1_adapt":
            self.imgaussfilt   = ImGaussFilt(1.0 * self.pix_per_deg, self.device)
        elif mode == "blur2_adapt":
            self.imgaussfilt   = ImGaussFilt(0.5 * self.pix_per_deg, self.device)



        if mode is None:
            logging.error("Error: Need mode")
            sys.exit(1)

        logging.info("Test loss mode: %s" % mode)

    def forward(self, output, target):
        N,C,D,H,W = target.shape

        lossvals = torch.zeros((D,))

        for ff in range(D):
            T = target[:,:,ff,...]
            R = output[:,:,ff,...]

            if   self.mode == "no_adapt":    L_bkg = torch.ones_like(R)
            elif self.mode == "mean_adapt":  L_bkg = torch.clamp(torch.ones_like(R) * torch.mean(R), min=self.epsilon)
            elif self.mode == "blur1_adapt": L_bkg = torch.clamp(self.imgaussfilt.run(R),            min=self.epsilon)
            elif self.mode == "blur2_adapt": L_bkg = torch.clamp(self.imgaussfilt.run(R),            min=self.epsilon)
            elif self.mode == "mape_adapt":  L_bkg = torch.clamp(R,                                  min=self.epsilon)
            elif self.mode == "smape_adapt": L_bkg = torch.clamp((R+T) * 0.5,                        min=self.epsilon)

            lossvals[ff] = torch.mean(torch.abs(R/L_bkg - T/L_bkg))

        return torch.mean(lossvals), None, None


class FovVideoVDP(torch.nn.Module):
    def __init__(self,
                 H, W, display_model, frames_per_s, do_diff_map, device,
                 mask_s, mask_p, mask_c, pu_dilate, debug, fixation_point, w_transient, beta, beta_t, beta_tch, beta_sch,
                 filter_len, sustained_sigma, sustained_beta, csf_sigma, do_foveated, sensitivity_correction, masking_model, band_callback,
                 local_adapt, contrast, jod_a, log_jod_exp, use_gpu, do_temporal_channels, mask_q_sust, mask_q_trans, k_cm, frame_padding
    ):

        super(FovVideoVDP, self).__init__()

        self.display_model          = display_model
        self.device                 = device

        self.mask_p                 = mask_p
        self.mask_c                 = mask_c
        self.pu_dilate              = pu_dilate
        self.debug                  = debug
        self.fixation_point         = fixation_point
        self.w_transient            = w_transient
        self.beta                   = beta
        self.beta_t                 = beta_t
        self.beta_tch               = beta_tch
        self.beta_sch               = beta_sch
        self.filter_len             = filter_len

        self.sustained_sigma        = sustained_sigma
        self.sustained_beta         = sustained_beta
        self.csf_sigma              = csf_sigma
        self.do_foveated            = do_foveated
        self.sensitivity_correction = sensitivity_correction
        self.masking_model          = masking_model
        self.band_callback          = band_callback
        self.local_adapt            = local_adapt
        self.contrast               = contrast
        self.jod_a                  = jod_a
        self.log_jod_exp            = log_jod_exp
        self.do_diff_map            = do_diff_map
        self.use_gpu                = use_gpu
        self.do_temporal_channels   = do_temporal_channels

        self.mask_q_sust             = mask_q_sust
        self.mask_q_trans            = mask_q_trans

        self.k_cm = k_cm

        self.frame_padding          = frame_padding

        self.frames_per_s           = frames_per_s
        self.csf_cache              = {}
        self.csf_cache_dirs         = [
                                        "csf_cache",
                                        os.path.join(os.path.dirname(__file__), "csf_cache"),
                                      ]

        self.W = W
        self.H = H

        self.mask_s                 = mask_s

        #self.video_name             = video_name
        #self.te_slope                = te_slope
        #self.mask_s_sust             = mask_s_sust
        #self.mask_s_trans            = mask_s_trans
        #self.k_mask_self            = k_mask_self
        #self.mask_q                 = mask_q

        if self.debug:
            self.tb = FovVideoVDP_Testbench()

        # print("VDP Params:")
        # kidx = 0
        # for k, v in sorted(vars(self).items()):
        #     if not k.startswith("_"):
        #         print("%-30s" % (k + ": " + str(v)), end = '')
        #         if kidx%2==1: print()
        #         kidx+=1
        # print()

        self.pix_per_deg = self.display_model.get_ppd(0.0)

        if self.mask_s > 0:
            self.mask_p = self.mask_q + self.mask_s

        if len(self.fixation_point) == 0:
            self.fixation_point = [self.W//2, self.H//2]

        if self.mask_s > 0.0:
            self.mask_p = self.mask_q + self.mask_s

        is_image = (frames_per_s==0)

        if is_image:
            self.omega = torch.tensor([0,5], device=self.device, requires_grad=False)
        else:
            if self.filter_len == -1:
                self.filter_len = int(np.ceil( 250.0 / (1000.0/self.frames_per_s) ))
            self.F, self.omega = self.get_temporal_filters(self.frames_per_s)
            if self.debug: self.tb.verify_against_matlab(self.F,     'F_pod', self.device)
            if self.debug: self.tb.verify_against_matlab(self.omega, 'omega', self.device)

        for oo in self.omega:
            self.preload_cache(oo, self.csf_sigma)

        self.lpyr = hdrvdp_lpyr_dec(self.W, self.H, self.pix_per_deg, self.device)
        self.imgaussfilt = ImGaussFilt(0.5 * self.pix_per_deg, self.device)

    def print_version_info( self, parameters, display_model ):
        # Print the specification line, identical to Matlab's version

        logging.info("When reporting metric results, please include the following information:")
        if self.do_foveated:
            foveated_str = 'foveated'
        else:
            foveated_str = 'non-foveated'  

        if hasattr(display_model, 'display_name') and display_model.display_name.startswith('standard'):
            display_str=", ({disp_name})".format(disp_name=display_model.display_name)
        else:
            display_str=""

        logging.info("FovVideoVDP v{vnum:.1f}, {ppd:.1f} [pix/deg], Lpeak={lpeak}, Lblack={lblack:.4f} [cd/m^2], {fovstr}{disp_str}".format(vnum=parameters['version'], ppd=self.pix_per_deg, lpeak=display_model.max_luminance, lblack=display_model.min_luminance, fovstr=foveated_str, disp_str=display_str) )


    @classmethod
    def load(cls, H, W, display_model, frames_per_s, do_diff_map,
             do_foveated=False, # changed default foveation to False to match Matlab implementation
             device=torch.device('cpu') ):

        parameters_file = os.path.join(os.path.dirname(__file__), "../fvvdp_data/fvvdp_parameters.json")

        obj = cls.__new__(cls)

        parameters = json2dict(parameters_file)

        #all common parameters between Matlab and Pytorch, loaded from the .json file
        mask_p = parameters['mask_p']
        mask_c = parameters['mask_c'] # content masking adjustment
        pu_dilate = parameters['pu_dilate']
        w_transient = parameters['w_transient'] # The weight of the transient temporal channel
        beta = parameters['beta'] # The exponent of the spatial summation (p-norm)
        beta_t = parameters['beta_t'] # The exponent of the summation over time (p-norm)
        beta_tch = parameters['beta_tch'] # The exponent of the summation over temporal channels (p-norm)
        beta_sch = parameters['beta_sch'] # The exponent of the summation over spatial channels (p-norm)
        sustained_sigma = parameters['sustained_sigma']
        sustained_beta = parameters['sustained_beta']
        csf_sigma = parameters['csf_sigma']
        sensitivity_correction = parameters['sensitivity_correction'] # Correct CSF values in dB. Negative values make the metric less sensitive.
        masking_model = parameters['masking_model']
        local_adapt = parameters['local_adapt'] # Local adaptation: 'simple' or or 'gpyr'
        contrast = parameters['contrast']  # Either 'weber' or 'log'
        jod_a = parameters['jod_a']
        log_jod_exp = parameters['log_jod_exp']
        mask_q_sust = parameters['mask_q_sust']
        mask_q_trans = parameters['mask_q_trans']
        k_cm = parameters['k_cm']  # new parameter controlling cortical magnification
        filter_len = parameters['filter_len']

        # other parameters
        debug = False
        fixation_point = []  # in pixel coordinates (x,y)
        band_callback = []
        use_gpu = False # Set to False to disable processing on a GPU (eg. when CUDA is not supported)
        do_temporal_channels = True  # Set to False to disable temporal channels and treat each frame as a image (for an ablation study)
        frame_padding = "replicate"
        mask_s = -1  # 2.0, # don't know what this parameter does!! (Alex)
        #do_foveated = False,
        #device = torch.device('cpu')

        # usage of these parameters not found (probably historic debris)

        # video_name = 'channels',
        # Parameters that are specific to a given masking model
        # te_slope = 1,  # Slope of the threshold elevation function of Daly's model !!
        # mask_s_sust = 0.4,
        # mask_s_trans = 0.2,

        obj.__init__(H, W, display_model, frames_per_s, do_diff_map, device,
            mask_s, mask_p, mask_c, pu_dilate, debug, fixation_point, w_transient, beta, beta_t, beta_tch, beta_sch,
            filter_len, sustained_sigma, sustained_beta, csf_sigma, do_foveated, sensitivity_correction, masking_model, band_callback,
            local_adapt, contrast, jod_a, log_jod_exp, use_gpu, do_temporal_channels, mask_q_sust, mask_q_trans, k_cm, frame_padding)

        obj.print_version_info(parameters, display_model)

        return obj

    def forward(self, output, target):
        return self.compute_metric(output, target)

    def torch_scalar(self, val, dtype=torch.float32):
        return torch.tensor(val, dtype=dtype, device=self.device)

    def raw_to_internal_frame(self, raw_fr):
        if self.contrast == 'log':
            return torch.log10(torch.clamp(raw_fr, min=1e-6))
        else:
            return torch.clamp(raw_fr, min=1e-6)

    def compute_metric(self, T_vid, R_vid):
        # T_vid and R_vid are the tensors of the size (1,1,N,H,W)
        # where:
        # N - the number of frames
        # H - height in pixels
        # W - width in pixels
        # Both images must contain linear absolute luminance values in cd/m^2
        # 
        # We assume the pytorch default NCDHW layout
        assert self.W == R_vid.shape[-1] and self.H == R_vid.shape[-2]
        assert len(R_vid.shape)==5

        N = R_vid.shape[2] # number of frames

        is_image = (N==1)  # Can run faster on images

        if is_image:
            temp_ch = 1  # How many temporal channels
        else:
            temp_ch = 2

        if self.do_diff_map:
            diff_map = torch.zeros_like(R_vid)
        else:
            diff_map = None

        N_nCSF = []
        sw_buf = [None, None]
        Q_per_ch = None
        w_temp_ch = [1.0, self.w_transient]

        fl = self.filter_len

        ## Not sliding window
        # T_vid_int = self.raw_to_internal_frame(T_vid)
        # R_vid_int = self.raw_to_internal_frame(R_vid)
        # if self.frame_padding == "replicate":
        #     prepad_t = T_vid_int[:, :, 0:1, :, :].expand([T_vid_int.shape[0], T_vid_int.shape[1], fl-1, T_vid_int.shape[3], T_vid_int.shape[4]])
        #     prepad_r = R_vid_int[:, :, 0:1, :, :].expand([R_vid_int.shape[0], R_vid_int.shape[1], fl-1, R_vid_int.shape[3], R_vid_int.shape[4]])
        # elif self.frame_padding == "circular":
        #     prepad_t = T_vid_int[:, :, -fl:, :, :]
        #     prepad_r = R_vid_int[:, :, -fl:, :, :]
        # sw_buf[0] = torch.cat((prepad_t, T_vid_int), 2)
        # sw_buf[1] = torch.cat((prepad_r, R_vid_int), 2)
        for ff in range(N):

            if is_image:
                R = torch.zeros((1, 2, 1, R_vid.shape[3], R_vid.shape[4]), device=self.device)
                R[:,0, :, :, :] = self.raw_to_internal_frame(T_vid[:, :, 0, :, :])
                R[:,1, :, :, :] = self.raw_to_internal_frame(R_vid[:, :, 0, :, :])

            else: # This is video
                if self.debug: print("Frame %d:\n----" % ff)

                if ff == 0: # First frame
                    if self.frame_padding == "replicate":
                        sw_buf[0] = self.raw_to_internal_frame(T_vid[:, :, 0:1, :, :]).expand([T_vid.shape[0], T_vid.shape[1], fl, T_vid.shape[3], T_vid.shape[4]])
                        sw_buf[1] = self.raw_to_internal_frame(R_vid[:, :, 0:1, :, :]).expand([R_vid.shape[0], R_vid.shape[1], fl, R_vid.shape[3], R_vid.shape[4]])
                    elif self.frame_padding == "circular":
                        sw_buf[0] = torch.zeros([T_vid.shape[0], T_vid.shape[1], fl, T_vid.shape[3], T_vid.shape[4]], device=self.device)
                        sw_buf[1] = torch.zeros([R_vid.shape[0], R_vid.shape[1], fl, R_vid.shape[3], R_vid.shape[4]], device=self.device)
                        for kk in range(fl):
                            fidx = (N - 1 - fl + kk) % N
                            sw_buf[0][:,:,kk,...] = self.raw_to_internal_frame(T_vid[:,:,fidx,...])
                            sw_buf[1][:,:,kk,...] = self.raw_to_internal_frame(R_vid[:,:,fidx,...])
                    elif self.frame_padding == "pingpong":
                        sw_buf[0] = torch.zeros([T_vid.shape[0], T_vid.shape[1], fl, T_vid.shape[3], T_vid.shape[4]], device=self.device)
                        sw_buf[1] = torch.zeros([R_vid.shape[0], R_vid.shape[1], fl, R_vid.shape[3], R_vid.shape[4]], device=self.device)

                        pingpong = list(range(0,N)) + list(range(N-2,0,-1))
                        indices = []
                        while(len(indices) < (fl-1)):
                            indices = indices + pingpong
                        indices = indices[-(fl-1):] + [0]

                        for kk in range(fl):
                            fidx = indices[kk]
                            sw_buf[0][:,:,kk,...] = self.raw_to_internal_frame(T_vid[:,:,fidx,...])
                            sw_buf[1][:,:,kk,...] = self.raw_to_internal_frame(R_vid[:,:,fidx,...])
                else:
                    cur_tframe = self.raw_to_internal_frame(T_vid[:, :, ff:(ff+1), :, :])
                    cur_rframe = self.raw_to_internal_frame(R_vid[:, :, ff:(ff+1), :, :])

                    sw_buf[0] = torch.cat((sw_buf[0][:, :, 1:, :, :], cur_tframe), 2)
                    sw_buf[1] = torch.cat((sw_buf[1][:, :, 1:, :, :], cur_rframe), 2)

                # Order: test-sustained, ref-sustained, test-transient, ref-transient
                R = torch.zeros((1, 4, 1, R_vid.shape[3], R_vid.shape[4]), device=self.device)

                for cc in range(temp_ch):
                    corr_filter = self.F[cc].flip(0).view([1,1,1,1,self.F[cc].shape[0]]) # pytorch convolutions default to "correlation" instead
                    R[:,cc*2+0, :, :, :] = Func.conv1d(sw_buf[0].permute(0,1,3,4,2), corr_filter, padding=0).permute(0,1,4,2,3)
                    R[:,cc*2+1, :, :, :] = Func.conv1d(sw_buf[1].permute(0,1,3,4,2), corr_filter, padding=0).permute(0,1,4,2,3)

            if self.debug: self.tb.verify_against_matlab(R.permute(0,2,3,4,1), 'Rdata', self.device, file='R_%d' % (ff+1), tolerance = 0.01)

            # Perform Laplacian decomposition
            # B = [None] * R.shape[1]
            # for rr in range(R.shape[1]):
            #     B[rr] = hdrvdp_lpyr_dec( R[0,rr:(rr+1),0:1,:,:], self.pix_per_deg, self.device)

            B_bands, B_gbands = self.lpyr.decompose(R[0,...])

            if self.debug: assert len(B_bands) == self.lpyr.get_band_count()

            # CSF
            N_nCSF = [[None, None] for i in range(self.lpyr.get_band_count()-1)]

            if self.do_diff_map:
                Dmap_pyr_bands, Dmap_pyr_gbands = self.lpyr.decompose( torch.zeros_like(R_vid[0,0:1,0:1,...]))

            # L_bkg_bb = [None for i in range(self.lpyr.get_band_count()-1)]

            rho_band = self.lpyr.get_freqs()

            # Adaptation
            L_adapt = None

            if self.local_adapt == "simple":
                L_adapt = R[0,1,0,...] # reference, sustained
                if self.contrast == "log":
                    L_adapt = torch.pow(10.0, L_adapt)
                L_adapt = self.imgaussfilt.run(L_adapt)
            elif self.local_adapt == "global":
                print("ERROR: global adapt not supported")
                return

            for cc in range(temp_ch):
                for bb in range(self.lpyr.get_band_count()-1):

                    T_f = self.lpyr.get_band(B_bands, bb)[cc*2+0,0,...]
                    R_f = self.lpyr.get_band(B_bands, bb)[cc*2+1,0,...]

                    L_bkg, R_f, T_f = self.compute_local_contrast(R_f, T_f, 
                        self.lpyr.get_gband(B_gbands, bb+1)[1:2,...], L_adapt)

                    # temp_errs[ff] += torch.mean(torch.abs(R_f - T_f))
                    # continue

                    # if self.debug: self.tb.verify_against_matlab(L_bkg, 'L_bkg_data', self.device, file='L_bkg_%d_%d_%d' % (ff+1,bb+1,cc+1), tolerance = 0.01)

                    # #  TODO: check cc = 1
                    # if self.debug:
                    #     print("%d, %d" % (cc, bb))
                    #     h, w = T_f.shape[-2], T_f.shape[-1]
                    #     np2img(stack_horizontal([
                    #         (T_f * 100.0).squeeze().unsqueeze(-1).expand(h,w,3).cpu().numpy(), 
                    #         (R_f * 100.0).squeeze().unsqueeze(-1).expand(h,w,3).cpu().numpy()])).show()
                    if self.debug: self.tb.verify_against_matlab(T_f, 'T_f_data', self.device, file='T_f_%d_%d_%d' % (ff+1,bb+1,cc+1), tolerance = 0.001) 
                    if self.debug: self.tb.verify_against_matlab(R_f, 'R_f_data', self.device, file='R_f_%d_%d_%d' % (ff+1,bb+1,cc+1), tolerance = 0.001)

                    if N_nCSF[bb][cc] is None:
                        ecc, res_mag = self.apply_display_model(R_vid.shape, R_f.shape)
                        rho = rho_band[bb] * res_mag
                        # if self.debug: self.tb.verify_against_matlab(ecc, 'ecc_data', self.device, file='ecc_%d_%d_%d' % (ff+1,bb+1,cc+1), tolerance = 0.01)  
                        # if self.debug: self.tb.verify_against_matlab(rho, 'rho_data', self.device, file='rho_%d_%d_%d' % (ff+1,bb+1,cc+1), tolerance = 0.05)  

                        S = self.cached_sensitivity(rho, self.omega[cc], L_bkg, ecc, self.csf_sigma) * np.power(10.0, self.sensitivity_correction/20.0)
                        # if self.debug: self.tb.verify_against_matlab(S, 'S_data', self.device, file='S_%d_%d_%d' % (ff+1,bb+1,cc+1), tolerance = 0.01, verbose=False)

                        if self.contrast == "log": N_nCSF[bb][cc] = self.weber2log(torch.min(1./S, self.torch_scalar(0.9999999)))
                        else:                      N_nCSF[bb][cc] = torch.reciprocal(S)
                        # if self.debug: self.tb.verify_against_matlab(N_nCSF[bb][cc], 'N_nCSF_data', self.device, file='N_nCSF_%d_%d_%d' % (ff+1,bb+1,cc+1), tolerance = 0.01, relative = True)

                    D = self.apply_masking_model(T_f, R_f, N_nCSF[bb][cc], cc)
                    if self.debug: self.tb.verify_against_matlab(D, 'D_data', self.device, file='D_%d_%d_%d' % (ff+1,bb+1,cc+1), tolerance = 0.1, relative = True, verbose=False)

                    if self.do_diff_map:
                        if cc == 0: self.lpyr.set_band(Dmap_pyr_bands, bb, D)
                        else:       self.lpyr.set_band(Dmap_pyr_bands, bb, self.lpyr.get_band(Dmap_pyr_bands, bb) + w_temp_ch[cc] * D)

                    if Q_per_ch is None:
                        Q_per_ch = torch.zeros((len(B_bands)-1, 2, N), device=self.device)

                    Q_per_ch[bb,cc,ff] = w_temp_ch[cc] * self.lp_norm(D.flatten(), self.beta, 0, True)

                    if self.debug: self.tb.verify_against_matlab(Q_per_ch[bb,cc,ff], 'Q_per_ch_data', self.device, file='Q_per_ch_%d_%d_%d' % (bb+1,cc+1,ff+1), tolerance = 0.1, relative=True, verbose=False)
            # break
            if self.do_diff_map:
                diff_map[:,:,ff,...] = self.lpyr.reconstruct(Dmap_pyr_bands)

        Q_sc = self.lp_norm(Q_per_ch, self.beta_sch, 0, False)
        Q_tc = self.lp_norm(Q_sc,     self.beta_tch, 1, False)
        Q    = self.lp_norm(Q_tc,     self.beta_t,   2, True)

        Q = Q.squeeze()

        beta_jod = np.power(10.0, self.log_jod_exp)
        Q_jod = np.sign(self.jod_a) * torch.pow(torch.tensor(np.power(np.abs(self.jod_a),(1.0/beta_jod)), device=self.device)* Q, beta_jod) + 10.0 # This one can help with very large numbers

        if self.do_diff_map:
            diff_map = torch.pow(diff_map, beta_jod) * abs(self.jod_a)         

        if self.debug: self.tb.verify_against_matlab(Q_per_ch, 'Q_per_ch_data', self.device, file='Q_per_ch', tolerance = 0.1, relative=True, verbose=True)
        if self.debug: self.tb.verify_against_matlab(Q_sc,     'Q_sc_data',     self.device, file='Q_sc',     tolerance = 0.1, relative=True, verbose=True)
        if self.debug: self.tb.verify_against_matlab(Q_tc,     'Q_tc_data',     self.device, file='Q_tc',     tolerance = 0.1, relative=True, verbose=True)
        if self.debug: self.tb.verify_against_matlab(Q,        'Q_data',        self.device, file='Q',        tolerance = 0.1, relative=True, verbose=True)
        if self.debug: self.tb.verify_against_matlab(Q_jod,    'Q_jod_data',    self.device, file='Q_jod',    tolerance = 0.1, relative=True, verbose=True)
        if self.debug: self.tb.print_summary()

        return Q, Q_jod.squeeze(), diff_map

    def compute_local_contrast(self, R, T, next_gauss_band, L_adapt):
        if self.local_adapt=="simple":
            L_bkg = Func.interpolate(L_adapt.unsqueeze(0).unsqueeze(0), R.shape, mode='bicubic', align_corners=True)
            # L_bkg = torch.ones_like(R) * torch.mean(R)
            # np2img(l2rgb(L_adapt.unsqueeze(-1).cpu().data.numpy())/200.0).show()
            # np2img(l2rgb(L_bkg[0,0].unsqueeze(-1).cpu().data.numpy())/200.0).show()
        elif self.local_adapt=="gpyr":
            if self.contrast == "log":
                next_gauss_band = torch.pow(10.0, next_gauss_band)
            L_bkg = self.lpyr.gausspyr_expand(next_gauss_band, [R.shape[-2], R.shape[-1]])
        else:
            print("Error: local adaptation %s not supported" % self.local_adapt)
            return

        if self.contrast != "log":
            L_bkg_clamped = torch.clamp(L_bkg, min=0.1)
            T = torch.clamp(torch.div(T, L_bkg_clamped), max=1000.0)
            R = torch.clamp(torch.div(R, L_bkg_clamped), max=1000.0)

        return L_bkg, R, T

    def apply_display_model(self, base_shape, band_shape):
        if self.do_foveated:
            [yy, xx] = torch.meshgrid( torch.arange(band_shape[-2], device=self.device)+1, torch.arange(band_shape[-1], device=self.device)+1 )

            df = base_shape[-1]/band_shape[-1] # ratio of width

            ecc = torch.sqrt((xx-self.fixation_point[0]/df) ** 2 + (yy-self.fixation_point[1]/df) ** 2 ) / self.pix_per_deg
            # np2img((ecc * 0.05).squeeze().unsqueeze(-1).expand(ecc.shape[0], ecc.shape[1], 3).cpu().numpy()).show()
            res_mag = self.display_model.get_resolution_magnification(ecc)
        else:
            res_mag = torch.full((band_shape[-2], band_shape[-1]), 1.0, device=self.device)
            ecc     = torch.full((band_shape[-2], band_shape[-1]), 0.0, device=self.device)

        return ecc, res_mag

    def get_cache_key(self, omega, sigma, k_cm):
        return ("o%g_s%g_cm%f" % (omega, sigma, k_cm)).replace('-', 'n').replace('.', '_')

    def preload_cache(self, omega, sigma):
        key = self.get_cache_key(omega, sigma, self.k_cm)
        for csf_cache_dir in self.csf_cache_dirs:
            #fname = os.path.join(csf_cache_dir, key + '_cpu.mat')
            fname = os.path.join(csf_cache_dir, key + '_gpu0.mat')
            if os.path.isfile(fname):
                #lut = load_mat_dict(fname, "lut_cpu", self.device)
                lut = load_mat_dict(fname, "lut", self.device)
                for k in lut:
                    lut[k] = torch.tensor(lut[k], device=self.device, requires_grad=False)
                self.csf_cache[key] = {"lut" : lut}
                break
        if key not in self.csf_cache:
            print("Error: cache file for %s not found" % key)

    def cached_sensitivity(self, rho, omega, L_bkg, ecc, sigma):
        key = self.get_cache_key(omega, sigma, self.k_cm)

        if key in self.csf_cache:
            lut = self.csf_cache[key]["lut"]
        else:
            print("Error: Key %s not found in cache" % key)

        # ASSUMPTION: rho_q and ecc_q are not scalars
        rho_q = torch.log2(torch.clamp(rho,   lut["rho"][0], lut["rho"][-1]))
        Y_q   = torch.log2(torch.clamp(L_bkg, lut["Y"][0],   lut["Y"][-1]))
        ecc_q = torch.sqrt(torch.clamp(ecc,   lut["ecc"][0], lut["ecc"][-1]))

        interpolated = interp3( lut["rho_log"], lut["Y_log"], lut["ecc_sqrt"], lut["S_log"], rho_q, Y_q, ecc_q)

        S = torch.pow(2.0, interpolated)

        return S

    def weber2log(self, W):
        # Convert Weber contrast 
        #
        # W = (B-A)/A
        #
        # to log contrast
        #
        # G = log10( B/A );
        #
        return torch.log10(1.0 + W)

    def phase_uncertainty(self, M):
        if self.pu_dilate != 0:
            M_pu = imgaussfilt( M, self.pu_dilate ) * torch.pow(10.0, self.mask_c)
        else:
            M_pu = M * torch.pow(self.torch_scalar(10.0), self.torch_scalar(self.mask_c))
        return M_pu

    def mask_func_perc_norm(self, G, G_mask):
        # Masking on perceptually normalized quantities (as in Daly's VDP)
        
        p = self.mask_p
        #q = self.mask_q
        #k = self.k_mask_self
        
        R = torch.div(torch.pow(G,p), torch.sqrt( 1. + torch.pow(k * G_mask, 2*q)))

        return R

    def mask_func_perc_norm2(self, G, G_mask, p, q ):
        # Masking on perceptually normalized quantities (as in Daly's VDP)        
        R = torch.div(torch.pow(G,p), 1. + torch.pow(G_mask, q))
        return R

    def apply_masking_model(self, T, R, N, cc):
        # cc - temporal channel: 0 - sustained, 1 - transient
        if self.masking_model == "joint_mutual_masking_perc_norm":
            T = torch.div(T, N)
            R = torch.div(R, N)
            M = self.phase_uncertainty( torch.min( torch.abs(T), torch.abs(R) ) )
            d = torch.abs(T-R)
            M = M + d
            D = self.mask_func_perc_norm( d, M )
        elif self.masking_model == 'min_mutual_masking_perc_norm2':
            p = self.mask_p
            if cc==0:
                q = self.mask_q_sust
            else:
                q = self.mask_q_trans     
            T = torch.div(T, N)
            R = torch.div(R, N)
            M = self.phase_uncertainty( torch.min( torch.abs(T), torch.abs(R) ) )
            D = self.mask_func_perc_norm2( torch.abs(T-R), M, p, q )
        else:
            print("Error: Masking model" + self.masking_model + "not implemented")

        D = torch.clamp( D, max=1e4)
        return D

    def lp_norm(self, x, p, dim=0, normalize=True):
        if dim is None:
            dim = 0

        if normalize:
            N = x.shape[dim]
        else:
            N = 1.0

        return torch.norm(x, p, dim=dim, keepdim=True) / (float(N) ** (1./p))

    def get_temporal_filters(self, frames_per_s):
        t = torch.linspace(0.0, self.filter_len / frames_per_s, self.filter_len, device=self.device)
        F = torch.zeros((2, t.shape[0]), device=self.device)

        sigma = torch.tensor([self.sustained_sigma], device=self.device)
        beta = torch.tensor([self.sustained_beta], device=self.device)

        F[0] = torch.exp(-torch.pow(torch.log(t + 1e-4) - torch.log(beta), 2.0) / (2.0 * (sigma ** 2.0)))
        F[0] = F[0] / torch.sum(F[0])

        Fdiff = F[0, 1:] - F[0, :-1]

        k2 = 0.062170507756932
        # This one seems to be slightly more accurate at low sampling rates
        F[1] = k2*torch.cat([Fdiff/(t[1]-t[0]), torch.tensor([0.0], device=self.device)], 0) # transient

        omega = torch.tensor([0,5], device=self.device, requires_grad=False)

        F[0].requires_grad = False
        F[1].requires_grad = False

        return F, omega


def parse_args():
    parser = argparse.ArgumentParser(description="Video-VDP metric test app")
    parser.add_argument("--gpu",      type=int,  default=-1, help="use GPU")
    group = parser.add_mutually_exclusive_group(required = True)
    group.add_argument("--benchmark", type=str, default=None, required=False, help="benchmark performance for chosen resolution: e.g. 1920x1080x60")
    group.add_argument("--test", action='store_true', default=False, required=False, help="test against FovDots MATLAB dataset (must be present on disk)")

    return parser.parse_args()

if __name__ == '__main__':

    torch.set_printoptions(precision=4, sci_mode=False, linewidth=120)

    args = parse_args()

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device('cuda:' + str(args.gpu))
    else:
        device = torch.device('cpu')

    print("Running on " + str(device))

    if args.benchmark is None:

        n_contents = 18
        n_conditions = 9

        test_content = 14 # 15 in MATLAB
        test_condition = 7 # 8 in MATLAB

        H = 1600
        W = 1400

        vdp = FovVideoVDP( H=H, W=W, display_model=DisplayModel.load('HTC Vive Pro', 'sRGB'), frames_per_s=90.0, do_diff_map=False, device=device, debug=True)

        for i_content in range(n_contents):
            for i_condition in range(n_conditions):
                if i_content == test_content and i_condition == test_condition:

                    print("%d-%d" % (i_content+1, i_condition+1))
                    print("    loading")
                    ref = fovdots_load_ref(i_content, device, data_res="full").to(device)
                    test = fovdots_load_condition(i_content, i_condition, device, data_res="full").to(device)

                    start_time = time.time()
                    vdp.compute_metric(test, ref)
                    elapsed_time = time.time() - start_time
                    print("elapsed: %0.4f sec" % (elapsed_time))
    else:
        W, H, D = [int(x) for x in args.benchmark.split("x")]

        print("CPU: ")
        print(cpuinfo.cpu.info[0])

        print("GPU: ")
        print(torch.cuda.get_device_name())

        with torch.no_grad():
            vdp = FovVideoVDP(H=H, W=W, display_model=DisplayModel.load('HTC Vive Pro', 'sRGB'), frames_per_s=90.0, do_diff_map=False, device=device, debug=False)

            ref  = torch.rand(1,1,D,H,W, device=device)
            test = torch.rand(1,1,D,H,W, device=device)

            bench = torchbench.Timer(stmt='vdp.compute_metric(ref, test)', globals=globals())

            print("Torch Benchmark:")
            print(bench.timeit(100))
