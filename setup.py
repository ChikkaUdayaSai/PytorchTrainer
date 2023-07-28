from setuptools import setup, find_packages
from version import __version__

setup(
    name='PytorchTrainer',
    version=__version__,
    packages=find_packages(),
    install_requires=[
        'torch==2.0.1',
        'torchvision==0.15.2',
        'matplotlib==3.7.1',
        'torchsummary==1.5.1',
        'albumentations==1.2.1',
        'torchinfo==1.8.0',
        'tqdm== 4.65.0',
        'numpy==1.22.4',
        'opencv-python==4.7.0.72',
        'torch-lr-finder==0.2.1',
        'grad-cam==1.4.8',
        'ttach==0.0.3'
    ],
)
