"""
Teaching-annotated gpSLDS modules.
Adapted from: https://github.com/lindermanlab/gpslds

This package contains core modules from the official gpSLDS implementation,
with detailed comments explaining mathematical foundations and design choices.
"""

from .kernels import *
from .likelihoods import *
from .em import *
from .utils import *

__version__ = "0.1.0"
__author__ = "Adapted for teaching from lindermanlab/gpslds"
