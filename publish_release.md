# Steps for Pip
### Prerequisites
- Install twine: `pip install twine`
- Register at https://pypi.org/account/register/

### Publish release
- Summarize changes in the [changelog](ChangeLog.md).
- Increment version number [here](pyfvvdp/fvvdp_data/fvvdp_parameters.json) and [here](setup.py).
- [Optional] Remove previous versions `rm dist/*`
- Create the new package: `python setup.py sdist`
- Upload the package: `twine upload dist/*`. You will be asked to enter your credentials for every update.

# Steps for Conda
You can build separate packages for different architectures and python versions. This is a guide for python 3.9 on a Linux machine.
### Prerequisites
- Install conda-build: `conda install conda-build`
- Install conda client: `conda install anaconda-client`
- Register at https://anaconda.org/account/login

### Publish release
- Navigate to a different folder: `TMP=$(mktmp -d); cd $TMP`
- Initialize with pip skeleton: `conda skeleton pypi pyfvvdp`
- Open the file `pyfvvdp/meta.yaml` and make the following changes:
  1. replace all occurances of "torch" with "pytorch". Pip and conda reference PyTorch under different names.
  2. replace "pyfvvdp/third_party" with "pyfvvdp.third_party". Not sure why conda initializes this field incorrectly, need to investigate further.
  3. update the field "recipe-maintainers".
  4. [Temporary] Update "summary" to use single quotes around FovVideoVDP. This should be fixed in next pip release.
- Build the conda package: `conda-build -c conda-forge -c pytorch --python 3.9 pyfvvdp`. You can build multiple packages, one for each python version to be supported.
- The last few lines of the output will indicate where the build files are. For example `Source and build intermediates have been left in /auto/homes/pmh64/miniconda3/envs/fov/conda-bld`. Let this be `$BLD`
- Convert the build to other architectures: `conda convert --platform win-64 $BLD/linux-64/pyfvvdp-1.1.1-py39_0.tar.bz2 -o $BLD`
- Upload all the packages to conda: `anaconda upload $BLD/*/pyfvvdp-*.tar.bz2`
- Logout from the client: `anaconda logout`
- [Optional] Clean intermediate build files: `conda build purge`
