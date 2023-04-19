# Steps for Pip
### Prerequisites
- Install twine: `pip install twine`
- Register at https://pypi.org/account/register/

### Publish release
- Summarize changes in the [changelog](ChangeLog.md).
- Increment version number [here](pyfvvdp/fvvdp_data/fvvdp_parameters.json) and [here](setup.py).
- [Optional] Remove previous versions `rm dist/*`
- Create the new package: `python setup.py sdist`
- [Optiona] Upload to the testpypi: `twine upload --repository testpypi dist/*`
- Upload the package: `twine upload dist/*`. 
  - You need to create a token on the pypi.org web page, then enter `__token__` as username and paste the token password as the password. 

# Steps for Conda
You can build separate packages for different architectures and python versions. This is a guide for python 3.9 on a Linux machine.
### Prerequisites
- First make sure the pip package is updated.
- Create a fresh conda environment and activate it.
- Install some conda packages: `conda install conda-build conda-verify`
- Install conda client: `conda install anaconda-client`
- Register at https://anaconda.org/account/login

### Publish release
- Navigate to a different folder: `TMP=$(mktemp -d); cd $TMP`
- Initialize with pip skeleton: `conda skeleton pypi pyfvvdp`
- Open the file `pyfvvdp/meta.yaml` and make the following changes:
  1. `sed -i 's/torch/pytorch/g' pyfvvdp/meta.yaml`: replace all occurances of "torch" with "pytorch". Pip and conda reference PyTorch under different names.
  2. `sed -i 's/pyfvvdp\/third_party/pyfvvdp.third_party/g' pyfvvdp/meta.yaml`: replace "pyfvvdp/third_party" with "pyfvvdp.third_party". Not sure why conda initializes this field incorrectly, need to investigate further.
  3. update the field "recipe-maintainers".
  4. add requirement `libiconv` (under host and run)
- Build the conda package: `conda-build -c conda-forge --python 3.9 pyfvvdp`. You can build multiple packages, one for each python version to be supported. **Important** Do not use multiple conda channels. This will result in broken packages.
- Test the build by installing pyfvvdp `conda install -c ${CONDA_PREFIX}/conda-bld/ -c conda-forge pyfvvdp`
- Convert the build to other architectures: E.g., `conda convert --platform all ${CONDA_PREFIX}/conda-bld/linux-64/pyfvvdp-1.1.1-py39_0.tar.bz2 -o ${CONDA_PREFIX}/conda-bld`
- Upload all the packages to conda: `anaconda upload ${CONDA_PREFIX}/conda-bld/*/pyfvvdp-*.tar.bz2`
- [Optional] Logout from the client: `anaconda logout`
- [Optional] Clean intermediate build files: `conda build purge`
