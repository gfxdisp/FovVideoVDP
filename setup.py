from setuptools import setup

setup(
    name='pyfvvdp',
    version='0.1.0',    
    description='Pytorch code accompanying "FovVideoVDP": a full-reference' \
                'visual quality metric that predicts the perceptual' \
                'difference between pairs of images and videos.',
    url='https://github.com/gfxdisp/pyfvvdp',
    author='Rafał K. Mantiuk',
    author_email='mantiuk@gmail.com',
    license='Creative Commons',
    packages=['pyfvvdp'],
    install_requires=['pytorch_msssim==0.2.1',
                      'scikit_video==1.1.11',
                      'numpy==1.23.1',
                      'scipy==1.8.1',
                      'graphviz==0.16',
                      'ffmpeg-python==0.2.0',
                      'torch==1.11.0',
                      'natsort==6.0.0',
                      'ffmpeg==1.4',
                      'imageio==2.19.5'],

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Creative Commons License',  
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)