from setuptools import setup

setup(
    name='pyfvvdp',
    version='1.0.0',
    description='Pytorch code accompanying "FovVideoVDP": a full-reference' \
                'visual quality metric that predicts the perceptual' \
                'difference between pairs of images and videos.',
    url='https://github.com/gfxdisp/FovVideoVDP',
    author='Rafał K. Mantiuk',
    author_email='mantiuk@gmail.com',
    license='Creative Commons',
    packages=['pyfvvdp', 'pyfvvdp/third_party'],
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
        'Operating System :: POSIX :: Linux',
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
