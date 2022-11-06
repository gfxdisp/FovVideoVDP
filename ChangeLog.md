# v1.1.4 - ?
 - [Python] Added option "--display-models" to specify an external file with display models
 - [Python] When passed "--display ?", a list of available display models will be printed

# v1.1.3 - 18/10/2022
 - [Python] Added "raw" heatmap type to the command line
 - [Python] Changed the way pyfvvdp classes are imported  to avoid clash between the file and class names (see updated pytortch_examples)
 - [Python] Installation of PyEXR is now optional (caused problems on some operating systems)

# v1.1.2 - 23/09/2022
 - [Python] Updated Python dependencies - now works with earlier versions of PyTorch, Numpy and SciPy

# v1.1.1 - 28/08/2022
- [Python] Thoroughly redesided the interface so that the code can be easily run from python code.
- [Python] All code examples are now both in python and Matlab.
- [Python] Optimized code for the Laplacian pyramid decomposition - faster predictions. There is a small (<0.02) difference in the predicted JODs.
- [Python] Improved compatibility with pytorch 1.12 + better performance
- [Python] PIP package
- [Matlab] Improved computation of eccentricity for wide-field-of-view displays
- [Matlab] Added new option 'ignore_boundary_pixels'
- [Matlab] Fixed the cortical magnificatioon bug in the CSF

# 19/05/2022

- Updated examples in README.md
- [Python] Fixed temporal convolution issue, which could cause problems on some versions of PyTorch. The code should be also a bit faster. 

# v1.0 - 21/04/2021

- Version 1.0 is explained in the SIGGRAPH Asia 2021 paper (see README.md).
- [Matlab] Updated calibration: added parameter on cortical magnification and 
  optimized for the weight of the transient channel
- [Matlab] Added support for reading videos from files (as in the python version)
- [Matlab] Metric now reverts to CPU of no CUDA CPU available
- [Matlab] added "foveated" parameter to 'fvvdp' (instead of an option)
- [Matlab] fvvdp is now more verbose about its photometric and geometric settings
- [Matlab] added an option to list available displays
- [Matlab] extended the documentation and fixed a few typos
- [Python] added the same "image" mode as Matlab version for much faster processing
- [Python] added "foveated" parameter. Non-foveated is the default (used to be the opposite)
- [Python] improved error reporting
- [Python] added ambient light reflection to the display model (to sync with Matlab)
- [All] raised ambient light levels for most displays to 250 lux (recommended office illumination)

# v0.2 - 26/02/2021

- [Matlab] Added `examples/ex_foveated_video.m`
- [Matlab] Fixed the issue with singularity when eccentricity was >90 deg
- [Matlab] Updated documentation throughout the code

# v0.1 - 13/02/2021

- The first release of the metric