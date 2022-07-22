import numpy as np
from scipy.ndimage import gaussian_filter


# Add 0 mean Gaussian noise
# std: Standard deviation in normalized units
# static: Set to True if same noise should be added to all frames
# peak: Intensity of brightest pixel
def imnoise(clean, std, static=False, peak=None):
    dtype = clean.dtype

    if peak is None:
        peak = 1 if dtype.kind == 'f' else np.iinfo(dtype).max

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


# Convert array of images to different datatypes
uint16to8 = lambda imgs: (np.floor(im/256).astype(np.uint8) for im in imgs)
# uint16toint16 = lambda imgs: (im.astype(np.int16) for im in imgs)
# uint16tofp32 = lambda imgs: (im.astype(np.float32)/(2**16 - 1) for im in imgs)
