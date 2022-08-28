# FovVideoVDP: A visible difference predictor for wide field-of-view video

<img src="https://www.cl.cam.ac.uk/research/rainbow/projects/fovvideovdp/teaser.png"></img>

FovVideoVDP is a full-reference visual quality metric that predicts the perceptual difference between pairs of images and videos. Similar to popular metrics like PSNR and SSIM, it is aimed at comparing a ground truth reference video against a distorted (e.g. compressed, lower framerate) version.

However, unlike traditional quality metrics, FovVideoVDP works for videos in addition to images, accounts for peripheral acuity, works with SDR and HDR content. We model the response of the human visual system to changes over time as well as across the visual field, so we can predict temporal artifacts like flicker and judder, as well as spatiotemporal artifacts as perceived at different degrees of peripheral vision. Such a metric is important for head-mounted displays as it accounts for both the dynamic content, as well as the large field of view.

FovVideoVDP has both a PyTorch and MATLAB implementations. The usage is described below.

The details of the metric can be found in:

Mantiuk, Rafał K., Gyorgy Denes, Alexandre Chapiro, Anton Kaplanyan, Gizem Rufo, Romain Bachy, Trisha Lian, and Anjul Patney. “FovVideoVDP : A Visible Difference Predictor for Wide Field-of-View Video.” ACM Transaction on Graphics 40, no. 4 (2021): 49. https://doi.org/10.1145/3450626.3459831.

The paper, videos and additional results can be found at the project web page: https://www.cl.cam.ac.uk/research/rainbow/projects/fovvideovdp/

If you use the metric in your research, please cite the paper above. 

## PyTorch quickstart

Install with PyPI `pip install pyfvvdp` and run directly from the command line:

```bash
fvvdp --test test_file --ref ref_file --gpu 0 --display standard_fhd
```
The test and reference files can be images or videos. `--display` specified a display on which the conrent is viewed. See [fvvdp_data/display_models.json](https://github.com/gfxdisp/FovVideoVDP/blob/main/fvvdp_data/display_models.json) for the available displays.

See [Command line interface](#command-line-interface) for further details. FovVideoVDP can be also run directly from Python - see [Low-level Python interface](#low-level-python-interface). 

**Table of contents**
- [Display specification](#display-specification)
    - [Custom specification](#custom-specification)
    - [Reporting metric results](#reporting-metric-results)
    - [Predicting quality scores](#predicted-quality-scores)
- [PyTorch](#pytorch)
    - [Command line interface](#command-line-interface)
    - [Low-level Python interface](#low-level-python-interface)
- [MATLAB](#matlab)
    - [Low-level MATLAB interface](#low-level-matlab-interface)
- [Differences between MATLAB and PyTorch versions](#differences-between-matlab-and-pytorch-versions)
- [Release notes](#release-notes)

## Display specification

Unlike most image quality metrics, FovVideoVDP needs physical specification of the display (e.g. its size, resolution, peak brightness) and viewing conditions (viewing distance, ambient light) to compute accurate predictions. The specifications of the displays are stored in `fvvdp_data/display_models.json`. You can add the exact specification of your display to this file, however, it is unknown to you, you are encouraged to use one of the standard display specifications listed on the top of that file, for example `standard_4k`, or `standard_fhd`. If you use one of the standard displays, there is a better chance that your results will be comparable with other studies. 

You specify the display by passing `--display` argument to the PyTorch code, or `display_name` parameter to the MATLAB code. 

Note the the specification in `display_models.json` is for the display and not the image. If you select to use `standard_hdr` with the resolution of 3840x2160 for your display and pass a 1920x1080 image, the metric will assume that the image occupies one quarter of that display (the central portion). 


### Custom specification

The display photometry and geometry is typically specified by passing `display_name` parameter to the metric. Alternatively, if you need more flexibility in specifying display geometry (size, fov, viewing distance) and its colorimetry, you can instead pass objects of the classes `fvvdp_display_geometry`, `fvvdp_display_photo_gog` for most SDR displays, and `fvvdp_display_photo_absolute` for HDR displays. You can also create your own subclasses of those classes for custom display specification. 

### Reporting metric results

When reporting the results of the metric, please include the string returned by the metric, such as:
`"FovVideoVDP v1.1, 75.4 [pix/deg], Lpeak=200, Lblack=0.5979 [cd/m^2], non-foveated, (standard_4k)"`
This is to ensure that you provide enough details to reproduce your results. 

### Predicted quality scores

FovVideoVDP reports image/video quality in the JOD (Just-Objectionable-Difference) units. The highest quality (no difference) is reported as 10 and lower values are reported for distorted content. In case of very strong distortion, or when comparing two unrelated images, the quality value can drop below 0. 

The main advantage of JODs is that they (a) should be linearly related to the perceived magnitude of the distortion and (b) the difference of JODs can be interpreted as the preference prediction across the population. For example, if method A produces a video with the quality score of 8 JOD and method B gives the quality score of 9 JOD, it means that 75% of the population will choose method B over A. The plots below show the mapping from the difference between two conditions in JOD units to the probability of selecting the condition with the higher JOD score (black numbers on the left) and the percentage increase in preference (blue numbers on the right). For more explanation, please refer to Section 3.9 and Fig. 9 in the main paper.

The differences in JOD scores can be converted to the percentage increase in preference (or the probability selecting A over B) using the MATLAB function `fvvdp_preference`.

<table>
  <tr>
    <td>Fine JOD scale</td>
    <td>Coarse JOD scale</td>
  </tr>
  <tr>
    <td><img width="512" src="https://github.com/gfxdisp/FovVideoVDP/raw/webpage/imgs/fine_jod_scale.png"></img></td>
    <td><img width="512" src="https://github.com/gfxdisp/FovVideoVDP/raw/webpage/imgs/coarse_jod_scale.png"></img></td>
  </tr>
</table>

## PyTorch

### Command line interface
The main script to run the model on a set of images or videos is [run_fvvdp.py](https://github.com/gfxdisp/FovVideoVDP/blob/main/pyfvvdp/run_fvvdp.py), from which the binary `fvvdp` is created . Run `fvvdp --help` for detailed usage information.

<img src="https://github.com/gfxdisp/FovVideoVDP/raw/main/example_media/aliasing/ferris-ref.gif"></img>

For the first example, the above video was downsampled (4x4) and upsampled (4x4) by different combinations of Bicubic and Nearest filters. To predict quality, you can run:

```bash
fvvdp --test example_media/aliasing/ferris-*-*.mp4 --ref example_media/aliasing/ferris-ref.mp4 --gpu 0 --display standard_fhd --heatmap supra-threshold
```

<table>
  <tr>
    <td>Bicubic &#8595; Bicubic &#8593; (4x4)</td>
    <td>Bicubic &#8595; Nearest &#8593; (4x4)</td>
    <td>Nearest &#8595; Bicubic &#8593; (4x4)</td>
    <td>Nearest &#8595; Nearest &#8593; (4x4)</td>
  </tr>
  <tr>
    <td>6.6277</td>
    <td>6.4803</td>
    <td>6.0446</td>
    <td>5.9450</td>
  </tr>
  <tr>
    <td><img src="https://github.com/gfxdisp/FovVideoVDP/raw/main/example_media/aliasing/diff_maps/ferris-bicubic-bicubic_diff_map_viz.gif"></img></td>
    <td><img src="https://github.com/gfxdisp/FovVideoVDP/raw/main/example_media/aliasing/diff_maps/ferris-bicubic-nearest_diff_map_viz.gif"></img></td>
    <td><img src="https://github.com/gfxdisp/FovVideoVDP/raw/main/example_media/aliasing/diff_maps/ferris-nearest-bicubic_diff_map_viz.gif"></img></td>
    <td><img src="https://github.com/gfxdisp/FovVideoVDP/raw/main/example_media/aliasing/diff_maps/ferris-nearest-nearest_diff_map_viz.gif"></img></td>
  </tr>
</table>

The second row of the table shows predicted quality and the GIFs show difference maps for various conditions.

### Low-level Python interface
FovVideoVDP can also be run through the Python interface by instatiating the `pyfvvdp.fvvdp.fvvdp` class. This example shows how to predict the quality of images degraded by Gaussian noise and blur.

```python
from pyfvvdp.fvvdp import fvvdp
from pyfvvdp.video_source_file import load_image_as_array
import ex_utils as utils

I_ref = load_image_as_array(os.path.join('example_media', 'wavy_facade.png'))
fv = fvvdp(display_name='standard_4k', heatmap='threshold')

# Gaussian noise with variance 0.003
I_test_noise = utils.imnoise(I_ref, np.sqrt(0.003))
Q_JOD_noise, stats_noise = fv.predict( I_test_noise, I_ref, dim_order="HWC" )

# Gaussian blur with sigma=2
I_test_blur = utils.imgaussblur(I_ref, 2)
Q_JOD_blur, stats_blur = fv.predict( I_test_blur, I_ref, dim_order="HWC" )
```

<img src="https://github.com/gfxdisp/FovVideoVDP/raw/main/example_media/simple_image_diff_map.png"></img>

More examples can be found in these [example scripts](https://github.com/gfxdisp/FovVideoVDP/blob/main/pytorch_examples).


## MATLAB

MATLAB code for the metric can be found in `matlab/fvvdp.m`. The full documentation of the metric can be shown by typing `doc fvvdp`.

The best starting point is the examples, which can be found in `matlab/examples`. For example, to measure the quality of a noisy image and display the difference map, you can use the code:

```
I_ref = imread( 'wavy_facade.png' );
I_test_noise = imnoise( I_ref, 'gaussian', 0, 0.001 );

[Q_JOD_noise, diff_map_noise] = fvvdp( I_test_noise, I_ref, 'display_name', 'standard_phone', 'heatmap', 'threshold' );

clf
imshow( diff_map_noise );

```

By default, FovVideoVDP will run the code on a GPU using `gpuArray`s, which require functioning CUDA on your computer. If you do not have GPU with CUDA support (e.g. you are on Mac), the code will automatically fallback to the CPU, which will be much slower. 
 
### Low-level MATLAB interface

`fvvdp` function is the suitable choice for most cases. But if you need to run metric on large datasets, you can use a low-level function `fvvdp_core`. It requires as input an object of the class `fvvdp_video_source`, which supplies the metric with the frames. Refer to the documentation of that class for further details. 

## Differences between MATLAB and Pytorch versions

* Both versions are implementation of the same metric, but due to differences in the video loaders, you can expect to see small differences in their predictions - typically up to 0.05 JOD.
* PyTorch version is a bit faster when running on a GPU. 

## Release notes

* v1.1.1 - 28 August 2022
  * We found a small inconsistency in eccentricity calculations. After fixing this, the metric has been retrained on the same datasets as described in the paper. FovVideoVDP v1.1 will return JOD values that are different than v1.0. For that reason, it is important to mention the version number when reporting the results. 
  * Python interface has been thoroughly redesigned and make more consistent with Matlab's conterpart. Now it should be much easier to call the metric from Python. 
  * All Matlab examples has been ported to Python. 
  * Python version is now faster. 
  * Published as a PIP repository. 

* v1.0 - 21 April 2021
  * The original FovVideoVDP release, released with the paper.

The detailed list of changes can be found in [ChangeLog.md](https://github.com/gfxdisp/FovVideoVDP/blob/main/ChangeLog.md).
