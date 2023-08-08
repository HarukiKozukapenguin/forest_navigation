import os
from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

__pkgname__ = 'forest_navigation'

setup_args = generate_distutils_setup(
    packages=[__pkgname__],
    package_dir={'': 'src'},
)

setup(**setup_args)