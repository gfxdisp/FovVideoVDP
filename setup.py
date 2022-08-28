from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='pyfvvdp',
    version='1.0.0',
    description='Pytorch code accompanying "FovVideoVDP": a full-reference' \
                'visual quality metric that predicts the perceptual' \
                'difference between pairs of images and videos.',
    long_description=long_description,
    url='https://github.com/gfxdisp/FovVideoVDP',
    long_description_content_type='text/markdown',
    author='Rafał K. Mantiuk',
    author_email='mantiuk@gmail.com',
    license='Creative Commons',
    packages=['pyfvvdp', 'pyfvvdp/third_party'],
    package_data={'pyfvvdp': ['csf_cache/*.mat', 'fvvdp_data/*.json']},
    include_package_data=True,
    install_requires=['numpy==1.23.1',
                      'scipy==1.8.1',
                      'ffmpeg-python==0.2.0',
                      'torch==1.12.0',
                      'ffmpeg==1.4',
                      'imageio==2.19.5'],

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],

    entry_points={
        'console_scripts': [
            'fvvdp=pyfvvdp.run_fvvdp:main'
        ]
    }
)
