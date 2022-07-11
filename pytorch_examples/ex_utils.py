import png
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter

# Wrapper for reading images
def imread(filename, bits=8):
    assert bits in (8, 16, 32), 'Unsupported bit-depth'
    if filename.endswith('.png'):
        pngdata = png.Reader(filename).read_flat()
        img = np.array(pngdata[2]).reshape((pngdata[1], pngdata[0], -1))
    else:
        img = np.array(Image.open(filename).convert("RGB"))
    return img


# Add 0 mean Gaussian noise
def imnoise(clean, std, static=False):
    dtype = clean.dtype
    if dtype == np.uint8:
        peak = 2**8-1
    elif dtype == np.uint16:
        peak = 2**16-1
    elif dtype == np.float32:
        peak = 1
    assert peak >= clean.max()

    if static:
        # Constant noise for all frames
        h, w, c, N = clean.shape    # axis=-1 is frame axis
        noise = np.repeat((np.random.randn(h, w, c, 1)*std), N, axis=-1)
    else:
        noise = np.random.randn(*clean.shape)*std
    noisy = clean.astype(np.float32)/peak + noise
    noisy = (noisy.clip(0, 1)*peak).astype(dtype)
    return noisy


# Blur RGB image by applying 2d Gaussian kernel
def imgaussblur(clean, sigmas):
    if clean.ndim == 3:    # Handle single input image
        clean = clean[...,np.newaxis]

    if np.isscalar(sigmas):
        sigmas = np.repeat(sigmas, clean.shape[-1])
    assert sigmas.shape[0] == clean.shape[-1]

    blur = np.zeros_like(clean)
    for ff, sigma in enumerate(sigmas): # for each frame
        for cc in range(3):              # for each color
            blur[...,cc,ff] = gaussian_filter(clean[...,cc,ff], sigma,
                                              mode='nearest', truncate=2.0)

    return blur.squeeze()


# Convert array of uint16 images to uint8
uint16to8 = lambda imgs: (np.floor(im/256).astype(np.uint8) for im in imgs)
