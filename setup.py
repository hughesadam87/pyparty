import sys
import os.path as op
#from distutils.core import setup
from setuptools import setup, find_packages

NAME = 'pyparty'

# Python >= 2.7 ?
# -------

#user_py = sys.version_info
#if user_py < (2, 7):
#    raise SystemExit('%s requires python 2.7 or higher (%s found).' % \
#                     (NAME, '.'.join(str(x) for x in user_py[0:3])))

#XXX Get current module path and do path.join(requirements)
#with open(op.join(sys.path[0], 'requirements.txt')) as f:
with open('requirements.txt', 'r') as f:
    required = f.readlines()
    required = [line.strip() for line in required]
    
# For now, most of these are in here for testing.  Dependency version 
#requirements can probably be relaxed, especially chaco.
setup(
    name = NAME,
    version = '0.1.2-1',
    author = 'Adam Hughes',
    maintainer = 'Adam Hughes',
    maintainer_email = 'hughesadam87@gmail.com',
    author_email = 'hughesadam87@gmail.com',
    packages = find_packages(),

    # include .tex and .ipynb files
    package_data={   
      'pyuvvis.examples.Notebooks':['*.ipynb'],
#      'pyuvvis.scripts':['*.png']
                 },
       
#    entry_points = {'console_scripts': [
#                       'gwuspec = pyuvvis.scripts.gwu_script.gwuspec:main',
#                       'gwureport = pyuvvis.scripts.gwu_script.gwureport:main']
#                    },
    
    url = 'http://pypi.python.org/pypi/PyUvVis/',
    download_url = 'https://github.com/hugadams/pyparty',
    license = 'LICENSE.txt',
    description = 'Tools for patterning 2d-shapes on ndarrays',
    long_description = open(op.join(sys.path[0], 'README.rst')).read(),
    install_requires = [
        # Setup.py install actually downloads and installs current version. 
        # IS THAT DESIRABLE? Or not cool to put that here?
        "scikit-image", 
     	"traits"
		],
    classifiers = [
          'Development Status :: 2 - Pre-Alpha',
          'Environment :: Console',
          'Intended Audience :: Science/Research',
          'Intended Audience :: Developers',
          'Natural Language :: English',
#          'Operating System :: OSIndependent',  #This is limited only by 
          'Programming Language :: Python :: 2',
          'Topic :: Scientific/Engineering :: Visualization',
          'Topic :: Scientific/Engineering :: Physics',
		  # ADD MORE?
          ],
)