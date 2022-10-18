from setuptools import setup

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

setup(
    name='pyfvvdp',
    version='1.1.3',
    description='PyTorch code for \'FovVideoVDP\': a full-reference' \
                'visual quality metric that predicts the perceptual' \
                'difference between pairs of images or videos.',
    long_description=long_description,
    url='https://github.com/gfxdisp/FovVideoVDP',
    long_description_content_type='text/markdown',
    author='Rafał K. Mantiuk',
    author_email='mantiuk@gmail.com',
    license='Creative Commons Attribution-NonCommercial 4.0 International Public License',
    packages=['pyfvvdp', 'pyfvvdp/third_party'],
    package_data={'pyfvvdp': ['csf_cache/*.mat', 'fvvdp_data/*.json']},
    include_package_data=True,
    install_requires=['numpy>=1.17.3',
                      'scipy>=1.7.0',
                      'ffmpeg-python>=0.2.0',
                      'torch>=1.8.2',
                      'ffmpeg>=1.4',
                      'imageio>=2.19.5'],

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],

    entry_points={
        'console_scripts': [
            'fvvdp=pyfvvdp.run_fvvdp:main'
        ]
    }
)
