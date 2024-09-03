from setuptools import setup

setup(
    name='CR_Analyze',
    version='0.0.2',
    packages=['CR_Analyze'],
    install_requires=['numpy','matplotlib']
    extras_require={
            "illustristng": ["h5py","illustris_python"]
        }
)
