
import torch
import torch.nn.functional as Func
import numpy as np 
import scipy.io as spio
import os
import sys
import torch.autograd.profiler as profiler

class hdrvdp_lpyr_dec():

    def __init__(self, W, H, ppd, device):
        self.device = device
        self.ppd = ppd
        self.min_freq = 0.5
        self.W = W
        self.H = H

        max_levels = int(np.floor(np.log2(min(self.H, self.W))))-1

        bands = np.concatenate([[1.0], np.power(2.0, -np.arange(0.0,14.0)) * 0.3228], 0) * self.ppd/2.0 

        # print(max_levels)
        # print(bands)
        # sys.exit(0)

        invalid_bands = np.array(np.nonzero(bands <= self.min_freq)) # we want to find first non0, length is index+1

        if invalid_bands.shape[-2] == 0:
            max_band = max_levels
        else:
            max_band = invalid_bands[0][0]

        # max_band+1 below converts index into count
        self.height = np.clip(max_band+1, 0, max_levels) # int(np.clip(max(np.ceil(np.log2(ppd)), 1.0)))
        self.band_freqs = np.array([1.0] + [0.3228 * 2.0 **(-f) for f in range(self.height-1)]) * self.ppd/2.0



    def get_freqs(self):
        return self.band_freqs

    def get_band_count(self):
        return self.height+1

    def get_band(self, bands, band):
        if band == 0 or band == (len(bands)-1):
            band_mul = 1.0
        else:
            band_mul = 2.0

        return bands[band] * band_mul

    def set_band(self, bands, band, data):
        if band == 0 or band == (len(bands)-1):
            band_mul = 1.0
        else:
            band_mul = 2.0

        bands[band] = data / band_mul

    def get_gband(self, gbands, band):
        return gbands[band]

    # def get_gband_count(self):
    #     return self.height #len(gbands)

    # def clear(self):
    #     for pyramid in self.P:
    #         for level in pyramid:
    #             # print ("deleting " + str(level))
    #             del level

    def decompose(self, image): 
        # assert len(image.shape)==4, "NCHW (C==1) is expected, got " + str(image.shape)
        # assert image.shape[-2] == self.H
        # assert image.shape[-1] == self.W

        # self.image = image

        return self.laplacian_pyramid_dec(image, self.height+1)

    def reconstruct(self, bands):
        img = bands[-1]

        for i in reversed(range(0, len(bands)-1)):
            img = self.gausspyr_expand(img, [bands[i].shape[-2], bands[i].shape[-1]])
            img += bands[i]

        return img

    def laplacian_pyramid_dec(self, image, levels = -1, kernel_a = 0.4):
        gpyr = self.gaussian_pyramid_dec(image, levels, kernel_a)

        height = len(gpyr)
        if height == 0:
            return []

        lpyr = []
        for i in range(height-1):
            layer = gpyr[i] - self.gausspyr_expand(gpyr[i+1], [gpyr[i].shape[-2], gpyr[i].shape[-1]], kernel_a)
            lpyr.append(layer)

        lpyr.append(gpyr[height-1])

        # print("laplacian pyramid summary:")
        # print("self.height = %d" % self.height)
        # print("height      = %d" % height)
        # print("len(lpyr)   = %d" % len(lpyr))
        # print("len(gpyr)   = %d" % len(gpyr))
        # sys.exit(0)

        return lpyr, gpyr

    def interleave_zeros(self, x, dim):
        z = torch.zeros_like(x, device=self.device)
        if dim==2:
            return torch.cat([x,z],dim=3).view(x.shape[0], x.shape[1], 2*x.shape[2],x.shape[3])
        elif dim==3:
            return torch.cat([x.permute(0,1,3,2),z.permute(0,1,3,2)],dim=3).view(x.shape[0], x.shape[1], 2*x.shape[3],x.shape[2]).permute(0,1,3,2)

    def gaussian_pyramid_dec(self, image, levels = -1, kernel_a = 0.4):

        default_levels = int(np.floor(np.log2(min(image.shape[-2], image.shape[-1]))))

        if levels == -1:
            levels = default_levels
        if levels > default_levels:
            raise Exception("Too many levels (%d) requested. Max is %d for %s" % (levels, default_levels, image.shape))

        res = [image]

        for i in range(1, levels):
            res.append(self.gausspyr_reduce(res[i-1], kernel_a))

        return res

    def get_guass_red_kernel_nchw(self, kernel_a):
        K1d = torch.tensor([0.25 - kernel_a/2.0, 0.25, kernel_a, 0.25, 0.25 - kernel_a/2.0], device=self.device)
        K = torch.reshape(K1d, (1, 1, K1d.shape[0], 1))
        return K

    def sympad(self, x, padding, axis):
        if padding == 0:
            return x
        else:
            beg = torch.flip(torch.narrow(x, axis, 0,        padding), [axis])
            end = torch.flip(torch.narrow(x, axis, -padding, padding), [axis])

            return torch.cat((beg, x, end), axis)

    def gausspyr_reduce(self, x, kernel_a = 0.4):
        K  = self.get_guass_red_kernel_nchw(kernel_a)
        Kt = K.permute(0, 1, 3, 2)

        y_a = Func.conv2d(self.sympad(  x, padding=2, axis=-2), K)
        y_a = y_a[:, :, ::2, :]
        y   = Func.conv2d(self.sympad(y_a, padding=2, axis=-1), Kt)
        
        y = y[:, :, :, ::2]

        return y

    def gausspyr_expand_pad(self, x, padding, axis):
        if padding == 0:
            return x
        else:
            beg = torch.narrow(x, axis, 0,        padding)
            end = torch.narrow(x, axis, -padding, padding)

            return torch.cat((beg, x, end), axis)

    def gausspyr_expand(self, x, sz = None, kernel_a = 0.4):
        if sz is None:
            sz = [x.shape[-2]*2, x.shape[-1]*2]

        K  = 2.0 * self.get_guass_red_kernel_nchw(kernel_a)
        Kt = K.permute(0, 1, 3, 2)
        y_a = self.interleave_zeros(x, dim=2)[...,0:sz[0],:]
        y_a = self.gausspyr_expand_pad(y_a, padding=2, axis=-2)
        y_a = Func.conv2d(y_a, K)

        y   = self.interleave_zeros(y_a, dim=3)[...,:,0:sz[1]]
        y   = self.gausspyr_expand_pad(  y,   padding=2, axis=-1)
        y   = Func.conv2d(y, Kt)

        return y


if __name__ == '__main__':

    device = torch.device('cpu')

    torch.set_printoptions(precision=2, sci_mode=False, linewidth=300)

    image = torch.tensor([
        [ 1,  2,  3,  4,  5,  6,  7,  8],
        [11, 12, 13, 14, 15, 16, 17, 18],
        [21, 22, 23, 24, 25, 26, 27, 28],
        [31, 32, 33, 34, 35, 36, 37, 38],
        [41, 42, 43, 44, 45, 46, 47, 48],
        [51, 52, 53, 54, 55, 56, 57, 58],
        [61, 62, 63, 64, 65, 66, 67, 68],
        [71, 72, 73, 74, 75, 76, 77, 78],
        ], dtype=torch.float32)

    image = torch.cat((image, image, image), axis = -1)
    image = torch.cat((image, image, image), axis = -2)


    lp = hdrvdp_lpyr_dec(image, 4.0, device)

    print("----Gaussian----")
    for gi in range(lp.get_gband_count()):
        print(lp.get_gband(gi))

    print("----Laplacian----")
    for li in range(lp.get_band_count()):
        print(lp.get_band(li))

