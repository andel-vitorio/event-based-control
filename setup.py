from setuptools import setup, find_packages

setup(
    name='event_based_control',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy',
        'pandas',
        'cvxpy',
        'ipython',
    ],
)
