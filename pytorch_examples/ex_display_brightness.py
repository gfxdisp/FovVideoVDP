import os
import numpy as np
import matplotlib.pyplot as plt
import ex_utils as utils

from pyfvvdp.fvvdp import fvvdp
from pyfvvdp.fvvdp_display_model import fvvdp_display_photo_gog
from pyfvvdp.video_source_file import load_image_as_array


I_ref = load_image_as_array(os.path.join('example_media', 'wavy_facade.png'))
std = np.sqrt(0.001)
I_test_noise = utils.imnoise(I_ref, std)

# Torch does not support uint16
I_ref, I_test_noise = utils.uint16tofp32((I_ref, I_test_noise))

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
