import os, sys
import numpy as np
import ex_utils as utils

sys.path.append('..')
from pyfvvdp import fvvdp, fvvdp_display_photo_gog

import matplotlib.pyplot as plt


I_ref = utils.imread(os.path.join('..', 'matlab', 'examples', 'wavy_facade.png'))
std = np.sqrt(0.001)
I_test_noise = utils.imnoise(I_ref, std)

# Torch does not support uint16
I_ref, I_test_noise = utils.uint16to8((I_ref, I_test_noise))

# Measure quality on displays of different brightness
disp_peaks = np.logspace(np.log10(1), np.log10(1000), 5)

# Display parameters
contrast = 1000   # Display contrast 1000:1
gamma = 2.2       # Standard gamma-encoding
E_ambient = 100   # Ambient light = 100 lux
k_refl = 0.005    # Reflectivity of the display

Q_JOD = []
for dd, Y_peak in enumerate(disp_peaks):
    disp_photo = fvvdp_display_photo_gog(Y_peak, contrast, gamma, E_ambient, k_refl)
    fv = fvvdp(display_name='standard_4k', display_photometry=disp_photo, heatmap='threshold')
    
    q, stats = fv.predict(I_test_noise, I_ref, dim_order="HWC")
    Q_JOD.append(q.cpu())

plt.plot(disp_peaks, Q_JOD, '-o')
plt.grid(which='major', linestyle='-')
plt.grid(which='minor', linestyle='--')
plt.xscale('log')
plt.xlabel('Display peak luminance [cd/m^2]')
plt.ylabel('Quality [JOD]')

plt.show()
