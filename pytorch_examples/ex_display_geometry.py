import os
import numpy as np
import ex_utils as utils
import matplotlib.pyplot as plt

import pyfvvdp

I_ref = pyfvvdp.load_image_as_array(os.path.join('example_media', 'wavy_facade.png'))
std = np.sqrt(0.005)
I_test_noise = utils.imnoise(I_ref, std)

# Measure quality at several viewing distances
distances = np.linspace(0.5, 2, 5)

Q_JOD = []
for dd, dist in enumerate(distances):
    # 4K, 30 inch display, seen at different viewing distances
    disp_geo = pyfvvdp.fvvdp_display_geometry((3840, 2160), diagonal_size_inches=30, distance_m=dist)
    fv = pyfvvdp.fvvdp(display_name='standard_4k', display_geometry=disp_geo, heatmap='threshold')
    
    q, stats = fv.predict(I_test_noise, I_ref, dim_order="HWC")
    Q_JOD.append(q.cpu())

plt.plot(distances, Q_JOD, '-o')
plt.grid(which='major', linestyle='-')
plt.grid(which='minor', linestyle='--')
plt.xlabel('Viewing distance [m]')
plt.ylabel('Quality [JOD]')

plt.show()
