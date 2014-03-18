import sys
import os.path as op
#from distutils.core import setup
from setuptools import setup, find_packages

NAME = 'pyparty'

# Python >= 2.7 
# -------
#user_py = sys.version_info
#if user_py < (2, 7):
#    raise SystemExit('%s requires python 2.7 or higher (%s found).' % \
#                    (NAME, '.'.join(str(x) for x in user_py[0:3])))
    
# For now, most of these are in here for testing.  Dependency version 
#requirements can probably be relaxed, especially chaco.
setup(
    name = NAME,
    version = '0.2.0', 
    author = 'Adam Hughes',
    maintainer = 'Adam Hughes',
    maintainer_email = 'hughesadam87@gmail.com',
    author_email = 'hughesadam87@gmail.com',
    packages = find_packages(),

    # include .tex and .ipynb files
    package_data={   
      'pyparty.examples.Notebooks':['*.ipynb'],
      'pyparty.bundled':['*.css'],
      'pyparty.data':['*'],
                 },
       
#    entry_points = {'console_scripts': [
#                       'gwuspec = pyuvvis.scripts.gwu_script.gwuspec:main',
#                       'gwureport = pyuvvis.scripts.gwu_script.gwureport:main']
#                    },
    
    url = 'http://pypi.python.org/pypi/pyparty/',
    download_url = 'https://github.com/hugadams/pyparty',
    license = 'LICENSE.txt',
    description = 'Tools for patterning 2d-shapes on ndarrays',
    
     # REMOVED DUE TO ENCODING ISSUES
 #   long_description = open(op.join(sys.path[0], 'README.txt'), 'rb').read(),
    
    classifiers = [
          'Development Status :: 2 - Pre-Alpha',
          'Environment :: Console',
          'Intended Audience :: Science/Research',
          'Intended Audience :: Developers',
          'Natural Language :: English',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX',
          'Operating System :: Unix',
          'Operating System :: MacOS',          
          'Programming Language :: Python :: 2',
          'Topic :: Scientific/Engineering :: Visualization',
          'Topic :: Scientific/Engineering :: Physics',
          ],
)
