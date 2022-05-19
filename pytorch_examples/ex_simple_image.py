import os
import torch
import numpy as np
from PIL import Image
from pyfvvdp import fvvdp


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

#%I_test_noise = I_ref.clone() + torch.randn_like(I_ref)*0.003

# I_test_noise = imnoise( I_ref, 'gaussian', 0, 0.003 );
# I_test_blur = imgaussfilt( I_ref, 2 );

#% imwrite( I_test_noise, 'wavy_facade_noise.png' );
#% imwrite( I_test_blur, 'wavy_facade_blur.png' );

#[Q_JOD_noise, diff_map_noise] = fvvdp( I_test_noise, I_ref, 'display_name', 'standard_4k', 'heatmap', 'threshold' );
#[Q_JOD_blur, diff_map_blur] = fvvdp( I_test_blur, I_ref, 'display_name', 'standard_4k', 'heatmap', 'threshold' );

fv = fvvdp(display_name='standard_4k', heatmap='threshold')

Q_JOD_noise, stats_noise = fv.predict( I_test_noise, I_ref )
print( 'Image with noise - Q_JOD = {:.3f}'.format( Q_JOD_noise ) )

Q_JOD_blur, stats_blur = fv.predict( I_test_blur, I_ref )
print( 'Image with blur - Q_JOD = {:.3f}'.format( Q_JOD_blur ) )





