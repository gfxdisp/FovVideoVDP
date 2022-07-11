# Example showing how to run FovVideoVDP on a numpy array with a video
# The video is autogenerated in this example. See ex_aliasing.py for an example in which video is loaded from .mp4 files

import os, sys
import numpy as np
import ex_utils as utils

sys.path.append('..')
from pyfvvdp import fvvdp
import time


# The frame to use for the video
I_ref = utils.imread(os.path.join('..', 'matlab', 'examples', 'wavy_facade.png'))

N = 60 # The number of frames
fps = 30 # Frames per second

V_ref = np.repeat(I_ref[...,np.newaxis], N, axis=3) # Reference video (in colour). 
N_amplitude = 0.07; # Amplitude of the noise (in gamma encoded values, scale 0-1)
V_static_noise = utils.imnoise(V_ref, N_amplitude, static=True)
V_dynamic_noise = utils.imnoise(V_ref, N_amplitude)

# Torch does not support uint16
V_ref, V_static_noise, V_dynamic_noise = utils.uint16to8((V_ref, V_static_noise, V_dynamic_noise))


fv = fvvdp(display_name='standard_4k', heatmap=None)

start = time.time()
Q_JOD_static, stats_static = fv.predict( V_static_noise, V_ref, dim_order="HWCF", frames_per_second=fps )
end = time.time()

print( 'Quality for static noise: {:.3f} JOD (took {:.4f} secs to compute)'.format(Q_JOD_static, end-start) )

start = time.time()
Q_JOD_dynamic, stats_dynamic = fv.predict( V_dynamic_noise, V_ref, dim_order="HWCF", frames_per_second=fps )
end = time.time()

print( 'Quality for dynamic noise: {:.3f} JOD (took {:.4f} secs to compute)'.format(Q_JOD_dynamic, end-start) )
