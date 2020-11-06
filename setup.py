from setuptools import find_packages, setup

setup(
    name='ppi-data-parser',
    packages=find_packages(),
    version='0.1.0',
    description='Data parser for protein-protein interactions.',
    author='Patricia Suriana',
    license='MIT',
    install_requires=[
        'atom3-py3',
        'easy-parallel',
        'numpy',
        'pandas',
    ],
)
