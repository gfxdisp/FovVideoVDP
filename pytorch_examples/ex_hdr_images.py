import os
import numpy as np
import matplotlib.pyplot as plt
import ex_utils as utils

from pyfvvdp.fvvdp import fvvdp
from pyfvvdp.fvvdp_display_model import fvvdp_display_photo_absolute
from pyfvvdp.video_source_file import load_image_as_array

I_ref = load_image_as_array(os.path.join('example_media', 'nancy_church.hdr'))

noise_fname = os.path.join('example_media', 'wavy_facade_noise.png')
L_peak = 4000   # Peak luminance of an HDR display

# HDR images are often given in relative photometric units. They MUST be
# mapped to absolute amount of light emitted from the display. For that, 
# we map the peak value in the image to the peak value of the display,
# then we increase the brightness by 2 stops (*4):
I_ref = I_ref/I_ref.max() * L_peak * 4

# Add Gaussian noise of 20% contrast
# I_test_noise = I_ref + I_ref.*randn(size(I_ref))*0.3;
std = 0.3/L_peak
I_test_noise = utils.imnoise(I_ref, std, peak=L_peak*4)

I_test_blur = utils.imgaussblur(I_ref, 2)

# We use geometry of SDR 4k 30" display, but ignore its photometric
# properties and instead tell that we pass absolute colorimetric values. 
# Note that many HDR images are in rec709 color space, so no need to
# specify rec2020. 
disp_photo = fvvdp_display_photo_absolute(L_peak)
fv = fvvdp(display_name='standard_hdr', display_photometry=disp_photo, heatmap='threshold')

# predict() method can handle numpy ndarrays or PyTorch tensors. The data
# type should be float32, int16 or uint8.
# Channels can be in any order, but the order must be specified as a dim_order parameter. 
# Here the dimensions are (Height,Width,Colour)
Q_JOD_noise, stats_noise = fv.predict( I_test_noise, I_ref, dim_order="HWC" )
noise_str = f'Noise - Quality: {Q_JOD_noise:.3f} JOD'
print( noise_str )

Q_JOD_blur, stats_blur = fv.predict( I_test_blur, I_ref, dim_order="HWC" )
blur_str = f'Blur - Quality: {Q_JOD_blur:.3f} JOD'
print( blur_str )

f, axs = plt.subplots(1, 2)
axs[0].imshow( stats_noise['heatmap'][0,:,0,:,:].permute([1,2,0]).cpu().numpy() )
axs[0].set_xticks([])
axs[0].set_yticks([])
axs[0].set_title(noise_str)
axs[1].imshow( stats_blur['heatmap'][0,:,0,:,:].permute([1,2,0]).cpu().numpy() )
axs[1].set_xticks([])
axs[1].set_yticks([])
axs[1].set_title(blur_str)

f.show();
plt.waitforbuttonpress()
