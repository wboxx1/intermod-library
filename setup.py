import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="intermod_library",
    version="0.0.1",
    author="Will Boxx",
    author_email="wboxx1@gmail.com",
    description="A library of tools for finding and viewing intermodulation signals and harmonics",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/wboxx1/intermod-library",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License:: OSI Approved:: GNU General Public License v3(GPLv3)",
        "Operating System :: OS Independent",
    ],
)
