from setuptools import find_packages
from setuptools import setup

setup(
    name="RL_Connect4",
    version="0.0.1",
    maintainer="niv",
    description=f"Local folders of RL_CONNECT4 as Python module",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
)