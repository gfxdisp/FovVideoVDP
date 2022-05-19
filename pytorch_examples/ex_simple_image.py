import os
import torch
import numpy as np
from PIL import Image
from pyfvvdp import fvvdp

import matplotlib.pyplot as plt

#from gfxdisp.pfs.pfs_torch import pfs_torch

def img2np(img):
    return np.array(img, dtype="float32") * 1.0/255.0


def load_image_as_tensor(imgfile, device, frames=1):
    raw_tensor = torch.tensor(img2np(Image.open(imgfile).convert("RGB"))).to(device)

    # # add batch and frame dimensions
    raw_tensor = raw_tensor.permute(2, 0, 1).unsqueeze(dim=0).unsqueeze(dim=2)
    return raw_tensor

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

I_ref = load_image_as_tensor( os.path.join( 'matlab', 'examples', 'wavy_facade.png' ), device )
I_test_noise = load_image_as_tensor( os.path.join( 'matlab', 'examples', 'wavy_facade_noise.png' ), device )
I_test_blur = load_image_as_tensor( os.path.join( 'matlab', 'examples', 'wavy_facade_blur.png' ), device )

#I_test_noise = I_ref.clone() + torch.randn_like(I_ref)*0.003

# I_test_noise = imnoise( I_ref, 'gaussian', 0, 0.003 );
# I_test_blur = imgaussfilt( I_ref, 2 );

fv = fvvdp(display_name='standard_4k', heatmap='threshold')

Q_JOD_noise, stats_noise = fv.predict( I_test_noise, I_ref )
noise_str = 'Noise; Quality: {:.3f} JOD'.format( Q_JOD_noise )
print( noise_str )


Q_JOD_blur, stats_blur = fv.predict( I_test_blur, I_ref )
blur_str = 'Blur; Quality: {:.3f} JOD'.format( Q_JOD_blur )
print( blur_str )


f, axs = plt.subplots(2, 2)
# axs[0][0].imshow( I_test_noise.cpu().numpy() )
# axs[0][1].imshow( I_test_blur.cpu().numpy() )
axs[1][0].imshow( stats_noise['heatmap'][0,:,0,:,:].permute([1,2,0]).cpu().numpy() )
axs[1][0].set_xticks([])
axs[1][0].set_yticks([])
axs[1][0].set_title(noise_str)
axs[1][1].imshow( stats_blur['heatmap'][0,:,0,:,:].permute([1,2,0]).cpu().numpy() )
axs[1][1].set_xticks([])
axs[1][1].set_yticks([])
axs[1][1].set_title(blur_str)

f.show();
plt.waitforbuttonpress()






