from setuptools import setup, find_packages
import sys
from pathlib import Path
# from iblvideo.__init__ import __version__

CURRENT_DIRECTORY = Path(__file__).parent.absolute()

CURRENT_PYTHON = sys.version_info[:2]
REQUIRED_PYTHON = (3, 7)
if CURRENT_PYTHON < REQUIRED_PYTHON:
    sys.stderr.write("""
==========================
Unsupported Python version
==========================
This version of iblvideo requires Python {}.{}, but you're trying to
install it on Python {}.{}.
""".format(*(REQUIRED_PYTHON + CURRENT_PYTHON)))
    sys.exit(1)

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
    name='iblvideo',
    version="0.0.0",  #__version__,
    python_requires='>={}.{}'.format(*REQUIRED_PYTHON),
    description='DLC applied to IBL data',
    license="MIT",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='IBL Staff',
    url="https://www.internationalbrainlab.com/",
    packages=find_packages(),
    include_package_data=True,
)
