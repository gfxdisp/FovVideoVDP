from abc import abstractmethod
from urllib.parse import ParseResultBytes
from numpy.lib.shape_base import expand_dims
import torch
from torch.functional import Tensor
import torch.nn.functional as Func
import numpy as np 
import os
import sys
import argparse
import time
import math
import torch.utils.benchmark as torchbench
import logging

from pyfvvdp.visualize_diff_map import visualize_diff_map

# For debugging only
# from gfxdisp.pfs.pfs_torch import pfs_torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from third_party.cpuinfo import cpuinfo
from hdrvdp_lpyr_dec import hdrvdp_lpyr_dec
from interp import interp1, interp3
from utils import *
from fvvdp_test import FovVideoVDP_Testbench

from pyfvvdp.fvvdp_display_model import fvvdp_display_photometry, fvvdp_display_geometry


"""
fvvdp_video_source_* objects are used to supply test/reference frames to FovVideoVDP. 
Those could be comming from memory or files. The subclasses of this abstract class implement
reading the frames and converting them to the approprtate format. 
"""
class fvvdp_video_source:
   
    # Return (height, width, frames) touple with the resolution and
    # the length of the video clip.
    @abstractmethod
    def get_video_size(self):
        pass

    # Return the frame rate of the video
    @abstractmethod
    def get_frames_per_second(self) -> int:
        pass
    
    # Get a pair of test and reference video frames as a single-precision luminance map
    # scaled in absolute inits of cd/m^2. 'frame' is the frame index,
    # starting from 0. 
    @abstractmethod
    def get_test_frame( self, frame, device ) -> Tensor:
        pass

    @abstractmethod
    def get_reference_frame( self, frame, device ) -> Tensor:
        pass


"""
Function for changing the order of dimensions, for example, from "WHC" (width, height, colour) to "BCHW" (batch, colour, height, width)
If a dimension is missing in in_dims, it will be added as a singleton dimension
"""
def reshuffle_dims( T: Tensor, in_dims: str, out_dims: str ) -> Tensor:
    in_dims = in_dims.upper()
    out_dims = out_dims.upper()    

    # Find intersection of two strings    
    inter_dims = ""
    for kk in range(len(out_dims)):
        if in_dims.find(out_dims[kk]) != -1:
            inter_dims += out_dims[kk]

    # First, permute into the right order
    perm = [0] * len(inter_dims)
    for kk in range(len(inter_dims)):
        ind = in_dims.find(inter_dims[kk])
        if ind == -1:
            raise RuntimeError( 'Dimension "{}" missing in the target dimensions: "{}"'.format(in_dims[kk],out_dims) )
        perm[kk] = ind                    
    T_p = torch.permute(T, perm)

    # Add missing dimensions
    out_sh = [1] * len(out_dims)
    for kk in range(len(out_dims)):        
        ind = inter_dims.find(out_dims[kk])
        if ind != -1:
            out_sh[kk] = T_p.shape[ind]

    return T_p.reshape( out_sh )


"""
This video_source uses a photometric display model to convert input content (e.g. sRGB) to luminance maps. 
"""
class fvvdp_video_source_dm( fvvdp_video_source ):

    def __init__( self,  display_photometry='sdr_4k_30', color_space_name='sRGB' ):
        colorspaces_file = os.path.join(os.path.dirname(__file__), "fvvdp_data/color_spaces.json")
        colorspaces = json2dict(colorspaces_file)

        if not color_space_name in colorspaces:
            raise RuntimeError( "Unknown color space: \"" + color_space_name + "\"" )

        self.color_to_luminance = colorspaces[color_space_name]['RGB2Y']

        if isinstance( display_photometry, str ):
            display_models_file = os.path.join(os.path.dirname(__file__), "fvvdp_data/display_models.json")
            display_models = json2dict(display_models_file)

            self.dm_photometry = fvvdp_display_photometry.load(display_photometry)
        elif isinstance( display_photometry, fvvdp_display_photometry ):
            self.dm_photometry = display_photometry
        else:
            raise RuntimeError( "display_model must be a string or fvvdp_display_photometry subclass" )


"""
This video source supplies frames from either Pytorch tensors and Numpy arrays. It also applies a photometric display model.

A batch of videos should be stored as a tensor or numpy array. Ideally, the tensor should have the dimensions BCFHW (batch, colour, frame, height, width).If tensor is stored in another formay, you can pass the order of dimsions as "dim_order" parameter. If any dimension is missing, it will
be added as a singleton dimension. 

This class is for display-encoded (gamma-encoded) content that will be processed by a display model to produce linear  absolute luminance emitted from a display.
"""
class fvvdp_video_source_array( fvvdp_video_source_dm ):
               
    # test_video, reference video - tensor with test and reference video frames. See the class description above for the explanation of dimensions of those tensors.
    # fps - frames per second. Must be 0 for images
    # dim_order - a string with the order of the dimensions. 'BCFHW' is the default.
    # display_model - object that implements fvvdp_display_photometry
    #   class
    # color_space_name - name of the colour space (see
    #   fvvdp_data/color_spaces.json)
    def __init__( self, test_video, reference_video, fps, dim_order='BCFHW', display_photometry='sdr_4k_30', color_space_name='sRGB' ):

        super().__init__(display_photometry=display_photometry, color_space_name=color_space_name)        

        if test_video.shape != reference_video.shape:
            raise RuntimeError( 'Test and reference image/video tensors must be exactly the same shape' )
        
        if len(dim_order) != len(test_video.shape):
            raise RuntimeError( 'Input tensor much have exactly as many dimensions as there are characters in the "dims" parameter' )

        # Convert numpy arrays to tensors. Note that we do not upload to device or change dtype at this point (to save GPU memory)
        if isinstance( test_video, np.ndarray ):
            if test_video.dtype == np.uint16:
                # Torch does not natively support uint16. A workaround is to pack uint16 values into int16.
                # This will be efficiently transferred and unpacked on the GPU.
                # logging.info('Test has datatype uint16, packing into int16')
                test_video = test_video.astype(np.int16)
            test_video = torch.tensor(test_video)
        if isinstance( reference_video, np.ndarray ):
            if reference_video.dtype == np.uint16:
                # Torch does not natively support uint16. A workaround is to pack uint16 values into int16.
                # This will be efficiently transferred and unpacked on the GPU.
                # logging.info('Reference has datatype uint16, packing into int16')
                reference_video = reference_video.astype(np.int16)
            reference_video = torch.tensor(reference_video)

        # Change the order of dimension to match BFCHW - batch, frame, colour, height, width
        test_video = reshuffle_dims( test_video, in_dims=dim_order, out_dims="BCFHW" )
        reference_video = reshuffle_dims( reference_video, in_dims=dim_order, out_dims="BCFHW" )

        B, C, F, H, W = test_video.shape
        
        if fps==0 and F>1:
            raise RuntimeError( 'When passing video sequences, you must set ''frames_per_second'' parameter' )

        if C!=3 and C!=1:
            raise RuntimeError( 'The content must have either 1 or 3 colour channels.' )

        self.fps = fps
        self.is_video = (fps>0)
        self.is_color = (C==3)
        self.test_video = test_video
        self.reference_video = reference_video


    def get_frames_per_second(self):
        return self.fps
            
    # Return a [height width frames] vector with the resolution and
    # the number of frames in the video clip. [height width 1] is
    # returned for an image. 
    def get_video_size(self):

        sh = self.test_video.shape
        return (sh[3], sh[4], sh[2])
    
    # % Get a test video frame as a single-precision luminance map
    # % scaled in absolute inits of cd/m^2. 'frame' is the frame index,
    # % starting from 0. If use_gpu==true, the function should return a
    # % gpuArray.

    def get_test_frame( self, frame, device=torch.device('cpu') ):
        return self._get_frame(self.test_video, frame, device )

    def get_reference_frame( self, frame, device=torch.device('cpu') ):
        return self._get_frame(self.reference_video, frame, device )

    def _get_frame( self, from_array, frame, device ):        
        # Determine the maximum value of the data type storing the
        # image/video

        if from_array.dtype is torch.float32:
            frame = from_array[:,:,frame:(frame+1),:,:].to(device)
        elif from_array.dtype is torch.int16:
            # Use int16 to losslessly pack uint16 values
            # Unpack from int16 by bit masking as described in this thread:
            # https://stackoverflow.com/a/20766900
            # logging.info('Found int16 datatype, unpack into uint16')
            max_value = 2**16 - 1
            # Cast to int32 to store values >= 2**15
            frame_int32 = from_array[:,:,frame:(frame+1),:,:].to(device).to(torch.int32)
            frame_uint16 = frame_int32 & max_value
            # Finally convert to float in the range [0,1]
            frame = frame_uint16.to(torch.float32) / max_value
        elif from_array.dtype is torch.uint8:
            frame = from_array[:,:,frame:(frame+1),:,:].to(device).to(torch.float32)/255
        else:
            raise RuntimeError( "Only uint8, uint16 and float32 is currently supported" )

        L = self.dm_photometry.forward( frame )
        
        if self.is_color:
            # Convert to grayscale
            L = L[:,0:1,:,:,:]*self.color_to_luminance[0] + L[:,1:2,:,:,:]*self.color_to_luminance[1] + L[:,2:3,:,:,:]*self.color_to_luminance[2]

        return L

"""
FovVideoVDP metric. Refer to pytorch_examples for examples on how to use this class. 
"""
class fvvdp:
    def __init__(self, display_name="standard_4k", display_photometry=None, display_geometry=None, color_space="sRGB", foveated=False, heatmap=None, quiet=False, device=None):
        self.quiet = quiet
        self.foveated = foveated
        self.heatmap = heatmap
        self.color_space = color_space

        if display_photometry is None:
            self.display_photometry = fvvdp_display_photometry.load(display_name)
        else:
            self.display_photometry = display_photometry
        
        self.do_heatmap = (not self.heatmap is None) and (self.heatmap != "none")

        if display_geometry is None:
            self.display_geometry = fvvdp_display_geometry.load(display_name)
        else:
            self.display_geometry = display_geometry

        self.pix_per_deg = self.display_geometry.get_ppd()

        # Use GPU if available
        if device is None:
            if torch.cuda.is_available() and torch.cuda.device_count()>0:
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device

        self.load_config()

        # if self.mask_s > 0.0:
        #     self.mask_p = self.mask_q + self.mask_s

        self.csf_cache              = {}
        self.csf_cache_dirs         = [
                                        "csf_cache",
                                        os.path.join(os.path.dirname(__file__), "csf_cache"),
                                      ]

        self.omega = torch.tensor([0,5], device=self.device, requires_grad=False)
        for oo in self.omega:
            self.preload_cache(oo, self.csf_sigma)

        self.lpyr = None
        self.imgaussfilt = ImGaussFilt(0.5 * self.pix_per_deg, self.device)


    def load_config( self ):

        parameters_file = os.path.join(os.path.dirname(__file__), "fvvdp_data/fvvdp_parameters.json")
        parameters = json2dict(parameters_file)

        #all common parameters between Matlab and Pytorch, loaded from the .json file
        self.mask_p = parameters['mask_p']
        self.mask_c = parameters['mask_c'] # content masking adjustment
        self.pu_dilate = parameters['pu_dilate']
        self.w_transient = parameters['w_transient'] # The weight of the transient temporal channel
        self.beta = parameters['beta'] # The exponent of the spatial summation (p-norm)
        self.beta_t = parameters['beta_t'] # The exponent of the summation over time (p-norm)
        self.beta_tch = parameters['beta_tch'] # The exponent of the summation over temporal channels (p-norm)
        self.beta_sch = parameters['beta_sch'] # The exponent of the summation over spatial channels (p-norm)
        self.sustained_sigma = parameters['sustained_sigma']
        self.sustained_beta = parameters['sustained_beta']
        self.csf_sigma = parameters['csf_sigma']
        self.sensitivity_correction = parameters['sensitivity_correction'] # Correct CSF values in dB. Negative values make the metric less sensitive.
        self.masking_model = parameters['masking_model']
        self.local_adapt = parameters['local_adapt'] # Local adaptation: 'simple' or or 'gpyr'
        self.contrast = parameters['contrast']  # Either 'weber' or 'log'
        self.jod_a = parameters['jod_a']
        self.log_jod_exp = parameters['log_jod_exp']
        self.mask_q_sust = parameters['mask_q_sust']
        self.mask_q_trans = parameters['mask_q_trans']
        self.k_cm = parameters['k_cm']  # new parameter controlling cortical magnification
        self.filter_len = parameters['filter_len']
        self.version = parameters['version']

        # other parameters
        self.debug = False

    '''
    Predict image/video quality using FovVideoVDP.

    test_cont and reference_cont can be either numpy arrays or PyTorch tensors with images or video frames. 
        Depending on the display model (display_photometry), the pixel values should be either display encoded, or absolute linear.
        The two supported datatypes are float16 and uint8.
    dim_order - a string with the order of dimensions of test_cont and reference_cont. The individual characters denote
        B - batch
        C - colour channel
        F - frame
        H - height
        W - width
        Examples: "HW" - gray-scale image (column-major pixel order); "HWC" - colour image; "FCHW" - colour video
        The default order is "BCFHW". The processing can be a bit faster if data is provided in that order. 
    frame_padding - the metric requires at least 250ms of video for temporal processing. Because no previous frames exist in the
        first 250ms of video, the metric must pad those first frames. This options specifies the type of padding to use:
          'replicate' - replicate the first frame
          'circular'  - tile the video in the front, so that the last frame is used for frame 0.
          'pingpong'  - the video frames are mirrored so that frames -1, -2, ... correspond to frames 0, 1, ...
    '''
    def predict(self, test_cont, reference_cont, dim_order="BCFHW", frames_per_second=0, fixation_point=None, frame_padding="replicate"):

        test_vs = fvvdp_video_source_array( test_cont, reference_cont, frames_per_second, dim_order=dim_order, display_photometry=self.display_photometry, color_space_name=self.color_space )

        return self.predict_video_source(test_vs, fixation_point=fixation_point, frame_padding=frame_padding)

    '''
    The same as `predict` but takes as input fvvdp_video_source_* object instead of Numpy/Pytorch arrays.
    '''
    def predict_video_source(self, vid_source, fixation_point=None, frame_padding="replicate"):

        # T_vid and R_vid are the tensors of the size (1,1,N,H,W)
        # where:
        # N - the number of frames
        # H - height in pixels
        # W - width in pixels
        # Both images must contain linear absolute luminance values in cd/m^2
        # 
        # We assume the pytorch default NCDHW layout

        vid_sz = vid_source.get_video_size() # H, W, F
        height, width, N_frames = vid_sz

        if fixation_point is None:
            fixation_point = torch.tensor([width//2, height//2])
        elif isinstance(fixation_point, np.ndarray):
            fixation_point = torch.tensor(fixation_point)

        if self.lpyr is None or self.lpyr.W!=width or self.lpyr.H!=height:
            self.lpyr = hdrvdp_lpyr_dec(width, height, self.pix_per_deg, self.device)

        #assert self.W == R_vid.shape[-1] and self.H == R_vid.shape[-2]
        #assert len(R_vid.shape)==5

        is_image = (N_frames==1)  # Can run faster on images

        if is_image:
            temp_ch = 1  # How many temporal channels
        else:
            temp_ch = 2
            self.filter_len = int(np.ceil( 250.0 / (1000.0/vid_source.get_frames_per_second()) ))

            self.F, self.omega = self.get_temporal_filters(vid_source.get_frames_per_second())
            # if self.debug: self.tb.verify_against_matlab(self.F,     'F_pod', self.device)
            # if self.debug: self.tb.verify_against_matlab(self.omega, 'omega', self.device)


        if self.do_heatmap:
            dmap_channels = 1 if self.heatmap == "raw" else 3
            heatmap = torch.zeros([1,dmap_channels,N_frames,height,width], dtype=torch.float, device=self.device) 
        else:
            heatmap = None

        N_nCSF = []
        sw_buf = [None, None]
        Q_per_ch = None
        w_temp_ch = [1.0, self.w_transient]

        fl = self.filter_len

        for ff in range(N_frames):

            if is_image:                
                R = torch.zeros((1, 2, 1, height, width), device=self.device)
                R[:,0, :, :, :] = vid_source.get_test_frame(0, device=self.device)
                R[:,1, :, :, :] = vid_source.get_reference_frame(0, device=self.device)

            else: # This is video
                if self.debug: print("Frame %d:\n----" % ff)

                if ff == 0: # First frame
                    if frame_padding == "replicate":
                        # TODO: proper handling of batches
                        sw_buf[0] = vid_source.get_test_frame(0, device=self.device).expand([1, 1, fl, height, width])
                        sw_buf[1] = vid_source.get_reference_frame(0, device=self.device).expand([1, 1, fl, height, width])
                    elif frame_padding == "circular":
                        sw_buf[0] = torch.zeros([1, 1, fl, height, width], device=self.device)
                        sw_buf[1] = torch.zeros([1, 1, fl, height, width], device=self.device)
                        for kk in range(fl):
                            fidx = (N_frames - 1 - fl + kk) % N_frames
                            sw_buf[0][:,:,kk,...] = vid_source.get_test_frame(fidx, device=self.device)
                            sw_buf[1][:,:,kk,...] = vid_source.get_reference_frame(fidx, device=self.device)
                    elif frame_padding == "pingpong":
                        sw_buf[0] = torch.zeros([1, 1, fl, height, width], device=self.device)
                        sw_buf[1] = torch.zeros([1, 1, fl, height, width], device=self.device)

                        pingpong = list(range(0,N_frames)) + list(range(N_frames-2,0,-1))
                        indices = []
                        while(len(indices) < (fl-1)):
                            indices = indices + pingpong
                        indices = indices[-(fl-1):] + [0]

                        for kk in range(fl):
                            fidx = indices[kk]
                            sw_buf[0][:,:,kk,...] = vid_source.get_test_frame(fidx,device=self.device)
                            sw_buf[1][:,:,kk,...] = vid_source.get_reference_frame(fidx,device=self.device)
                    else:
                        raise RuntimeError( 'Unknown padding method "{}"'.format(frame_padding) )
                else:
                    cur_tframe = vid_source.get_test_frame(ff, device=self.device)
                    cur_rframe = vid_source.get_reference_frame(ff, device=self.device)

                    sw_buf[0] = torch.cat((sw_buf[0][:, :, 1:, :, :], cur_tframe), 2)
                    sw_buf[1] = torch.cat((sw_buf[1][:, :, 1:, :, :], cur_rframe), 2)

                # Order: test-sustained, ref-sustained, test-transient, ref-transient
                R = torch.zeros((1, 4, 1, height, width), device=self.device)

                for cc in range(temp_ch):
                    # 1D filter over time (over frames)
                    corr_filter = self.F[cc].flip(0).view([1,1,self.F[cc].shape[0],1,1]) 
                    R[:,cc*2+0, :, :, :] = (sw_buf[0] * corr_filter).sum(dim=-3,keepdim=True)
                    R[:,cc*2+1, :, :, :] = (sw_buf[1] * corr_filter).sum(dim=-3,keepdim=True)

            if self.debug: self.tb.verify_against_matlab(R.permute(0,2,3,4,1), 'Rdata', self.device, file='R_%d' % (ff+1), tolerance = 0.01)

            # Perform Laplacian decomposition
            # B = [None] * R.shape[1]
            # for rr in range(R.shape[1]):
            #     B[rr] = hdrvdp_lpyr_dec( R[0,rr:(rr+1),0:1,:,:], self.pix_per_deg, self.device)

            B_bands, B_gbands = self.lpyr.decompose(R[0,...])

            if self.debug: assert len(B_bands) == self.lpyr.get_band_count()

            # CSF
            N_nCSF = [[None, None] for i in range(self.lpyr.get_band_count()-1)]

            if self.do_heatmap:
                Dmap_pyr_bands, Dmap_pyr_gbands = self.lpyr.decompose( torch.zeros([1,1,height,width], dtype=torch.float, device=self.device))

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

                        if self.foveated:   # Fixation, parafoveal sensitivity
                            if fixation_point.dim() == 2:
                                current_fixation_point = fixation_point[ff,:].squeeze()
                            else:
                                current_fixation_point = fixation_point

                            xv = torch.linspace( 0.5, vid_sz[1]-0.5, T_f.shape[-1], device=self.device )
                            yv = torch.linspace( 0.5, vid_sz[0]-0.5, T_f.shape[-2], device=self.device )
                            [xx, yy] = torch.meshgrid( xv, yv, indexing='xy' )

                            ecc = self.display_geometry.pix2eccentricity( torch.tensor((vid_sz[1], vid_sz[0])), xx, yy, current_fixation_point+0.5 )

                            # The same shape as bands
                            ecc = ecc.reshape( [1, 1, ecc.shape[-2], ecc.shape[-1]] )

                            res_mag = self.display_geometry.get_resolution_magnification(ecc)
                        else:   # No fixation, foveal sensitivity everywhere
                            res_mag = torch.ones(R_f.shape[-2:], device=self.device)
                            ecc = torch.zeros(R_f.shape[-2:], device=self.device)

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

                    if self.do_heatmap:
                        if cc == 0: self.lpyr.set_band(Dmap_pyr_bands, bb, D)
                        else:       self.lpyr.set_band(Dmap_pyr_bands, bb, self.lpyr.get_band(Dmap_pyr_bands, bb) + w_temp_ch[cc] * D)

                    if Q_per_ch is None:
                        Q_per_ch = torch.zeros((len(B_bands)-1, 2, N_frames), device=self.device)

                    Q_per_ch[bb,cc,ff] = w_temp_ch[cc] * self.lp_norm(D.flatten(), self.beta, 0, True)

                    if self.debug: self.tb.verify_against_matlab(Q_per_ch[bb,cc,ff], 'Q_per_ch_data', self.device, file='Q_per_ch_%d_%d_%d' % (bb+1,cc+1,ff+1), tolerance = 0.1, relative=True, verbose=False)
            # break
            if self.do_heatmap:
                beta_jod = np.power(10.0, self.log_jod_exp)
                dmap = torch.pow(self.lpyr.reconstruct(Dmap_pyr_bands), beta_jod) * abs(self.jod_a)         
                if self.heatmap == "raw":
                    heatmap[:,:,ff,...] = dmap 
                else:
                    ref_frame = R[:,0, :, :, :]
                    heatmap[:,:,ff,...] = visualize_diff_map(dmap, context_image=ref_frame, colormap_type=self.heatmap)


        Q_sc = self.lp_norm(Q_per_ch, self.beta_sch, 0, False)
        Q_tc = self.lp_norm(Q_sc,     self.beta_tch, 1, False)
        Q    = self.lp_norm(Q_tc,     self.beta_t,   2, True)

        Q = Q.squeeze()

        beta_jod = np.power(10.0, self.log_jod_exp)
        Q_jod = np.sign(self.jod_a) * torch.pow(torch.tensor(np.power(np.abs(self.jod_a),(1.0/beta_jod)), device=self.device)* Q, beta_jod) + 10.0 # This one can help with very large numbers

        stats = {}

        if self.do_heatmap:            
            stats['heatmap'] = heatmap

        if self.debug: self.tb.verify_against_matlab(Q_per_ch, 'Q_per_ch_data', self.device, file='Q_per_ch', tolerance = 0.1, relative=True, verbose=True)
        if self.debug: self.tb.verify_against_matlab(Q_sc,     'Q_sc_data',     self.device, file='Q_sc',     tolerance = 0.1, relative=True, verbose=True)
        if self.debug: self.tb.verify_against_matlab(Q_tc,     'Q_tc_data',     self.device, file='Q_tc',     tolerance = 0.1, relative=True, verbose=True)
        if self.debug: self.tb.verify_against_matlab(Q,        'Q_data',        self.device, file='Q',        tolerance = 0.1, relative=True, verbose=True)
        if self.debug: self.tb.verify_against_matlab(Q_jod,    'Q_jod_data',    self.device, file='Q_jod',    tolerance = 0.1, relative=True, verbose=True)
        if self.debug: self.tb.print_summary()    

        return (Q_jod.squeeze(), stats)

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
            raise RuntimeError("Error: cache file for %s not found" % key)

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

    def torch_scalar(self, val, dtype=torch.float32):
        return torch.tensor(val, dtype=dtype, device=self.device)
        

# def parse_args():
#     parser = argparse.ArgumentParser(description="Video-VDP metric test app")
#     parser.add_argument("--gpu",      type=int,  default=-1, help="use GPU")
#     group = parser.add_mutually_exclusive_group(required = True)
#     group.add_argument("--benchmark", type=str, default=None, required=False, help="benchmark performance for chosen resolution: e.g. 1920x1080x60")
#     group.add_argument("--test", action='store_true', default=False, required=False, help="test against FovDots MATLAB dataset (must be present on disk)")

#     return parser.parse_args()

# if __name__ == '__main__':

#     torch.set_printoptions(precision=4, sci_mode=False, linewidth=120)

#     args = parse_args()

#     if args.gpu >= 0 and torch.cuda.is_available():
#         device = torch.device('cuda:' + str(args.gpu))
#     else:
#         device = torch.device('cpu')

#     print("Running on " + str(device))

#     if args.benchmark is None:

#         n_contents = 18
#         n_conditions = 9

#         test_content = 14 # 15 in MATLAB
#         test_condition = 7 # 8 in MATLAB

#         H = 1600
#         W = 1400

#         vdp = FovVideoVDP( H=H, W=W, display_model=DisplayModel.load('HTC Vive Pro', 'sRGB'), frames_per_s=90.0, do_diff_map=False, device=device, debug=True)

#         for i_content in range(n_contents):
#             for i_condition in range(n_conditions):
#                 if i_content == test_content and i_condition == test_condition:

#                     print("%d-%d" % (i_content+1, i_condition+1))
#                     print("    loading")
#                     ref = fovdots_load_ref(i_content, device, data_res="full").to(device)
#                     test = fovdots_load_condition(i_content, i_condition, device, data_res="full").to(device)

#                     start_time = time.time()
#                     vdp.compute_metric(test, ref)
#                     elapsed_time = time.time() - start_time
#                     print("elapsed: %0.4f sec" % (elapsed_time))
#     else:
#         W, H, D = [int(x) for x in args.benchmark.split("x")]

#         print("CPU: ")
#         print(cpuinfo.cpu.info[0])

#         print("GPU: ")
#         print(torch.cuda.get_device_name())

#         with torch.no_grad():
#             vdp = FovVideoVDP(H=H, W=W, display_model=DisplayModel.load('HTC Vive Pro', 'sRGB'), frames_per_s=90.0, do_diff_map=False, device=device, debug=False)

#             ref  = torch.rand(1,1,D,H,W, device=device)
#             test = torch.rand(1,1,D,H,W, device=device)

#             bench = torchbench.Timer(stmt='vdp.compute_metric(ref, test)', globals=globals())

#             print("Torch Benchmark:")
#             print(bench.timeit(100))


# class TestLoss(torch.nn.Module):
#     def __init__(self, display_model, device, epsilon):
        
#         super(TestLoss, self).__init__()

#         self.display_model = display_model
#         self.device        = device
#         self.pix_per_deg   = self.display_model.get_ppd(0.0)
#         self.epsilon       = epsilon

#         mode = "blur2_adapt"

#         self.mode          = mode

#         if mode == "blur1_adapt":
#             self.imgaussfilt   = ImGaussFilt(1.0 * self.pix_per_deg, self.device)
#         elif mode == "blur2_adapt":
#             self.imgaussfilt   = ImGaussFilt(0.5 * self.pix_per_deg, self.device)



#         if mode is None:
#             logging.error("Error: Need mode")
#             sys.exit(1)

#         logging.info("Test loss mode: %s" % mode)

#     def forward(self, output, target):
#         N,C,D,H,W = target.shape

#         lossvals = torch.zeros((D,))

#         for ff in range(D):
#             T = target[:,:,ff,...]
#             R = output[:,:,ff,...]

#             if   self.mode == "no_adapt":    L_bkg = torch.ones_like(R)
#             elif self.mode == "mean_adapt":  L_bkg = torch.clamp(torch.ones_like(R) * torch.mean(R), min=self.epsilon)
#             elif self.mode == "blur1_adapt": L_bkg = torch.clamp(self.imgaussfilt.run(R),            min=self.epsilon)
#             elif self.mode == "blur2_adapt": L_bkg = torch.clamp(self.imgaussfilt.run(R),            min=self.epsilon)
#             elif self.mode == "mape_adapt":  L_bkg = torch.clamp(R,                                  min=self.epsilon)
#             elif self.mode == "smape_adapt": L_bkg = torch.clamp((R+T) * 0.5,                        min=self.epsilon)

#             lossvals[ff] = torch.mean(torch.abs(R/L_bkg - T/L_bkg))

#         return torch.mean(lossvals), None, None
