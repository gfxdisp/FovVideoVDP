# v1.0 - 21/04/2021

- Version 1.0 is explained in the SIGGRAPH Asia 2021 paper (see README.md).
- [Matlab] Updated calibration: added parameter on cortical magnification and 
  optimized for the weight of the transient channel
- [Matlab] Added support for reading videos from files (as the python version)
- [Matlab] Metric now reverts to CPU of no CUDA CPU available
- [Matlab] added "foveated" parameter to 'fvvdp' (instead of an option)
- [Matlab] raised ambient light levels for most displays to 250 lux (recommended office illumination)
- [Matlab] fvvdp is now more verbose about its photometric and geometric settings
- [Matlab] added an option to list available displays
- [Matlab]extended the documentation and fixed a few typos

# v0.2 - 26/02/2021

- [Matlab] Added `examples/ex_foveated_video.m`
- [Matlab] Fixed the issue with singularity when eccentricity was >90 deg
- [Matlab] Updated documentation throughout the code

# v0.1 - 13/02/2021

- The first release of the metric