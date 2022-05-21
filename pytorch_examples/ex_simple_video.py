import os
from matplotlib.pyplot import axis
import torch
import numpy as np
from PIL import Image, ImageFilter
from pyfvvdp import fvvdp
import time

def load_image_as_array(imgfile):
    img = np.array(Image.open(imgfile).convert("RGB"))
    return img

# The frame to use for the video
I_ref = load_image_as_array( os.path.join( 'matlab', 'examples', 'wavy_facade.png' ) )

N = 60 # The number of frames
fps = 30 # Frames per second

V_ref = np.repeat(I_ref[:,:,:,np.newaxis], N, axis=3) # Reference video (in colour). 
max_v = 255
N_amplitude = 0.07; # Amplitude of the noise (in gamma encoded values, scale 0-1)
V_dynamic_noise = (V_ref.astype('int16') + (np.random.randn(I_ref.shape[0],I_ref.shape[1],I_ref.shape[2],N)*N_amplitude*max_v).astype('int16')).clip(0,255).astype('uint8')

# Static Gaussian noise
V_static_noise = (V_ref.astype('int16') + np.repeat((np.random.randn(I_ref.shape[0],I_ref.shape[1],I_ref.shape[2],1)*N_amplitude*max_v).astype('int16').clip(0,255), N, axis=3)).astype('uint8') 

fv = fvvdp(display_name='standard_4k', heatmap=None)

start = time.time()
Q_JOD_static, stats_static = fv.predict( V_static_noise, V_ref, dim_order="HWCF", frames_per_second=fps )
end = time.time()

print( 'Quality for static noise: {:.3f} JOD (took {:.4f} secs to compute)'.format(Q_JOD_static, end-start) )

start = time.time()
Q_JOD_dynamic, stats_dynamic = fv.predict( V_dynamic_noise, V_ref, dim_order="HWCF", frames_per_second=fps )
end = time.time()

print( 'Quality for dynamic noise: {:.3f} JOD (took {:.4f} secs to compute)'.format(Q_JOD_dynamic, end-start) )
