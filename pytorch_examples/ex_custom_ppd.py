# This example shows how to create own fvvdp_display_geometry with a custom computation of pixels-per-degree
# This example is based on "ex_foveated_video.py"
import os
import time
import torch
import numpy as np
import ex_utils as utils
from PIL import Image

import pyfvvdp

class custom_display_geometry( pyfvvdp.fvvdp_display_geometry ):
    # Get the number of pixels per degree 
    #
    # ppd = R.get_ppd()
    # ppd = R.get_ppd(view_dir)
    #
    # Without any arguments, the function returns ppd at the centre of the screen. 
    # When view_dir is provided, the function returns ppd for a given set of view directions. 
    # view_dir is a tensor [2 x height x width] containing horizontal and vertical angles in visual degrees. 
    # The central pixel have both coordinates equal to 0. The view_dir has both coordinates negative for pixels in the left-top cornert of the screen. 
    # pixel coordinates can be transformed to view_dir with the pix2view_direction method. 
    def get_ppd(self, view_dir = None):
                        
        if view_dir is None:
            return self.ppd_centre
        else:
            view_angle = torch.sqrt(torch.sum((view_dir)**2, dim=0, keepdim=False ))        
            # let's assume a display in which ppd decreases as we move away from the centre            
            ppd = self.ppd_centre / (view_angle/20. + 1.)
            return ppd


I_ref = pyfvvdp.load_image_as_array(os.path.join('example_media', 'wavy_facade.png'))
ar = 1440/1600; # the aspect ratio of HTC Vive Pro (width/height)
h, w, _ = I_ref.shape
crop_pix = int(np.floor((w - h*ar)/2))
I_ref = I_ref[:,crop_pix-1:-crop_pix] # Crop to the aspect ratio of HTC Vive Pro

# Torch does not support uint16
# I_ref = next(utils.uint16tofp32(I_ref[np.newaxis]))
I_ref = next(utils.uint16to8(I_ref[np.newaxis]))

# The below line needs to be replaced. PIL forces uint8 which is
# inconsistent with Matlab
I_ref = np.array(Image.fromarray(I_ref).resize((1440, 1600), resample=3)) # 3 is bicubic

N = 60 # The number of frames
fps = 30 # Frames per second

V_ref = np.repeat(I_ref[...,np.newaxis], N, axis=3) # Reference video (in colour). 
noise_amplitude = 0.02
V_test_noise = utils.imnoise(V_ref, noise_amplitude)

# The gaze will move from the top-left to the bottom-right corner
# We are pasing [N 2] matrix with the fixation points as [x y], where x
# goes from 0 to width-1.
# If the gaze position is fixed, pass [x y] vector. 
# If you ommit 'fixation_point' option, the fixation will be set to the
# centre of the image.
gaze_pos = np.stack((np.linspace(0, V_ref.shape[1]-1, N),
                     np.linspace(0, V_ref.shape[0]-1, N))).T

disp_geom = custom_display_geometry(resolution=(1440, 1600), distance_m=3, fov_diagonal=110.0 )

# The metric will use display_model.json to initialize photometric model (for 'htc_vive_pro'), but it 
# will use the custom class for the geometric model
fv = pyfvvdp.fvvdp(display_name='htc_vive_pro', display_geometry=disp_geom, heatmap=None, foveated=True)

start = time.time()
Q_JOD_dynamic, stats_dynamic = fv.predict(V_test_noise, V_ref,
                               dim_order="HWCF", frames_per_second=fps,
                               fixation_point=gaze_pos)
end = time.time()

print( 'Quality for dynamic noise: {:.3f} JOD (took {:.4f} secs to compute)'.format(Q_JOD_dynamic, end-start) )
