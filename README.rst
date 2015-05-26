.. image:: pyparty/data/coverimage.png
   :height: 100px
   :width: 200 px
   :scale: 50 %
   :alt: alternate text
   :align: left

==========
What's New
==========
Check out this study on `sickle cells`_ using ``pyparty`` and ``skimage`` from Clay Mortin.

``pyparty`` is now `published`_.  

Check out `Object Hunter`_, a ``pyparty`` script for identifying, summarizing and plotting 
groups of objects in an image.

   .. _`Object Hunter` : http://nbviewer.ipython.org/urls/raw.github.com/hugadams/pyparty/master/examples/Notebooks/objecthunt_tutorial.ipynb?create=1
   .. _`published` : http://dx.doi.org/10.5334/jors.bh

   .. _`sickle cells` : https://github.com/hugadams/pyparty/blob/master/clay.html


======================================
pyparty: Python (py) particles (party) 
======================================

``pyparty`` is a small library for drawing, labeling, patterning and manipulating 
particles in 2d images.  ``pyparty`` was built primarily over the excellent
image processing library, scikit-image_.

   .. _scikit-image: http://scikit-image.org


Getting Started
===============

Videos 
``````

Check out Zhaowen Liu's `great screencast`_ on using Ilastik to segregate AuNP types on SEM images.

   .. _`great screencast` : https://www.youtube.com/watch?v=YzylgLw4iTA

The current documentation is a series of example notebooks 
(`IPython Notebook`_), which cover most of the basics. These have been linked below:

- **TUTORIALS**:
   - `Intro to Canvas: Basic Operations`_ 
   - `Intro to Shapes`_
   - `Intro to Grids`_
   - `Intro to MultiCanvas`_
   
- **LABELS FROM IMAGES**:
   - `Intro to Labeling`_
   - `Labeling Nanoparticle Species`_

- **MISCELLANEOUS**:
   - `Matplotlib Color Maps`_
   - `Watershedding Example Adapted`_

- **ARTIFICIAL IMAGES**:
   - `Basic Artificial SEM Images and Noise`_
   - `Simple Images and Labels for JORS`_

- **SCRIPTS**:
   - `Object Hunter`_

- **COMING SOON**:
   - *Advanced artificial SEM/TEM images*
   
   .. _`Intro to Canvas: Basic Operations`: http://nbviewer.ipython.org/github/hugadams/pyparty/blob/master/examples/Notebooks/basictests.ipynb?create=1
   .. _`Intro to Shapes`: http://nbviewer.ipython.org/github/hugadams/pyparty/blob/master/examples/Notebooks/shapes.ipynb?create=1
   .. _`Intro to Grids` : http://nbviewer.ipython.org/github/hugadams/pyparty/blob/master/examples/Notebooks/grids.ipynb?create=1
   .. _`Intro to MultiCanvas` : http://nbviewer.ipython.org/github/hugadams/pyparty/blob/master/examples/Notebooks/multi_tutorial.ipynb?create=1
   .. _`Intro to Labeling`: http://nbviewer.ipython.org/github/hugadams/pyparty/blob/master/examples/Notebooks/Analyze_Particles.ipynb?create=1
   .. _`Labeling Nanoparticle Species` :  http://nbviewer.ipython.org/github/hugadams/pyparty/blob/master/examples/Notebooks/groups_of_labels.ipynb?create=1
   .. _`Basic Artificial SEM Images and Noise` : http://nbviewer.ipython.org/github/hugadams/pyparty/blob/master/examples/Notebooks/making_noise.ipynb?create=1
   .. _`Matplotlib Color Maps` : http://nbviewer.ipython.org/github/hugadams/pyparty/blob/master/examples/Notebooks/gwu_maps.ipynb?create=1
   .. _`Watershedding Example Adapted` : http://nbviewer.ipython.org/github/hugadams/pyparty/blob/master/examples/Notebooks/watershed.ipynb?create=1
   .. _`Simple Images and Labels for JORS` : http://nbviewer.ipython.org/github/hugadams/pyparty/blob/master/examples/Notebooks/JORS_data.ipynb?create=1
   .. _`Object Hunter` : http://nbviewer.ipython.org/urls/raw.github.com/hugadams/pyparty/master/examples/Notebooks/objecthunt_tutorial.ipynb?create=1

Notebooks were initialized with ``pylab``:

   ipython notebook --pylab=inline
   
Having trouble viewing/editing notebooks?  Consider using `Enthought
Canopy`_, which has a notebook kernel builtin, as well as a graphical package manager. 
For simple viewing, paste the github url of each notebook into the IPython Notebook viewer_. 
 
   .. _documentation: http://hugadams.github.com/pyparty/
   .. _`IPython Notebook`: http://ipython.org/notebook.html?utm_content=buffer83c2c&utm_source=buffer&utm_medium=twitter&utm_campaign=Buffer
   .. _`Enthought Canopy`: https://www.enthought.com/products/canopy/
   .. _viewer: http://nbviewer.ipython.org/   

**These notebooks are free for redistribution.  If referencing in publication, please cite as:**
	- Hughes, A 2014 pyparty: Blob Detection, Drawing and Manipulation in Python. Journal of Open Research Software, 2: e26, DOI: http://dx.doi.org/10.5334/jors.bh

Support
=======

For help, please write to our group, **pyparty@googlegroups.com**

Have a feature request, or want to report a bug?  Please fill out a github
issue_ with the appropriate label.	

.. _issue : https://github.com/hugadams/pyparty/issues


License
=======

3-Clause Revised BSD_

   .. _BSD : https://github.com/hugadams/pyuvvis/blob/master/LICENSE.txt

Overview and Features
=====================

``pyparty`` provides a simple API for particle analysis in 2d images, while streamlining 
common operations in the image processing pipeline.  

*Some key features include*:

1. Pythonic **ParticleManager** for storing and manipulating particles from image 
   labels OR builtin shapes.  Some highlights of **Particles** include:
       - A common datastructure for array operations like rotations and 
         translations.
       - ``skimage`` descriptors_ as primary attributes on all particles.
       - Filtering and mapping based with numpy logical indexing syntax. 
         
2. A **Grid** system for patterning particles, as well as mesh utilities for creating 
   image backgrounds.

3. A **Canvas** to easily integrate *Grids*, *Particles* and flexible *Backgrounds*. 
   In addition, Canvas also provides simplified interfaces for:
      - binarization / thresholding
      - plotting
      - slicing and other pythonic container operations

4. A plotting API based on matplotlib.imshow() that generally supports 
    rasterizaztions AND `matplotlib patches`_.

5. Flexible color designations ('red', (1,0,0), 00FF00), and strict typing
   to ensure consistency in data and plots.

6. General ndarray operations such as rotations and translations supported by ALL particle types.

7. API for adding **Noise** to images.

   .. _descriptors : http://scikit-image.org/docs/dev/api/skimage.measure.html#regionprops
   .. _`matplotlib patches` : http://matplotlib.org/examples/api/patch_collection.html

What are some use cases, and will pyparty help to me?
=====================================================

Tasked well-suited for ``pyparty`` include:

1. Filtering and characterization of cells based on descriptors like
eccentricit and area.

2. Patterning a grid of particles over a shadowed background to compare performance
   of thresholding algorithms.

3. Manipulating particles in a *pythonic* manner:

   - delete all particles that have area > 50 pixels.
   - sort and color ellipses in order of increasing eccentricity.
   - dilate all particles appearing in bottom half of an image

4. Scripting without leaving Python.

5. Plot particles as rasterizations or matplotlib patches side-by-side.

In short, you may consider using ``pyparty`` if you are doing image analysis and find 
generating, managing or labeling particles as a bottleneck.  

   .. _patchcollection : http://matplotlib.org/examples/api/patch_collection.html

License
=======

3-Clause Revised BSD_

   .. _BSD : https://github.com/hugadams/pyparty/blob/master/LICENSE.txt

Dependencies
============
``pyparty`` requires ``scikit-image``, ``Traits`` and their dependencies, which
include many core packages such as ``numpy`` and ``matplotlib``.  If you are new
to Python for scientific computing, consider downloading a packaged distribution_.

   .. _distribution :  https://www.enthought.com/products/canopy/

``pyparty`` uses Traits_ because it is well-suited for writing clean, type-checked
object-oriented classes. You will not need to understand or use ``Traits``
unless you develop for ``pyparty``; *it is not used in the public API*, and may be 
removed in future installments after the core functionality is stable.

   .. _Traits : http://code.enthought.com/projects/traits/
   
Installation
============

I would recommend using `Enthought Canopy`_ and installing ``Traits`` and 
``scikit-image`` through the package manager; however, ``pyparty`` is also 
registered on PyPi_.

   .. _PyPi : https://pypi.python.org/pypi/pyparty

Pip Install
-----------

Make sure you have pip installed:

    sudo apt-get install python-pip
    
Then:
   
    pip install pyparty
    
To install all of the dependencies (scikit-image, traits and their various dependencies), download ``pyparty`` from github, navigate
to the base directory and type:

    pip install -r requirements.txt


Installation from source
------------------------

In the ``pyparty`` base directory run:

    python setup.py install

The developmental version can be cloned from github:

    git clone https://github.com/hugadams/pyparty.git
    
This will not install any dependencies.
    

Testing 
-------

To quickly test your installation, open python and type:

    from pyparty import *

If this results in no errors, the installation probably went smoothly.

While a proper nosetests platform is still under development, there is a 
quasi-regression test suite in *pyparty/testing/REGRESSION.ipynb*.  This 
will run all of the available pyparty ipython notebooks located in *pyparty/examples/Notebooks*,
and capture the output.  If any of the operations in these notebooks raises an error,
it will be reported back to the REGRESSION notebook.  This requires **ipython 3.0.0** to run!

A static version of the test suite may be viewed here_.

   .. _here: http://nbviewer.ipython.org/github/hugadams/pyparty/blob/master/testing/REGRESSION.ipynb


Archive
-------

`pyparty` is archived on Zenodo (DOI 10.5281/zenodo.11194)

http://dx.doi.org/10.5281/zenodo.11194


    
Related Libraries
=================
Interested in the Python ecosystem?   Check out some of these related libraries:

   - NumPy_ (Fundamental vectorized numerics in Python)
   - SciPy_ (Collection of core, numpy-based scientific libraries)
   - scikit-image_ (Scipy image processing suite)
   - matplotlib_ (De facto static plotting in Python)
   - pandas_ (Data analysis library : inspired ``pyparty`` ParticleManager API)
   - ilastik_ (Interactive Learning and Segmentation Tool)
   - Pillow_ (Python Image Library)

   .. _Pillow: http://python-imaging.github.io/
   .. _NumPy: http://www.numpy.org/
   .. _pandas: http://pandas.pydata.org/
   .. _SciPy: http://scipy.org/
   .. _matplotlib : http://matplotlib.org/
   .. _ilastik : http://www.ilastik.org/
   
Coming Soon
===========
   - More multi-particle types.
   - Better control of color shading of labels.
   - More examples.
   
Have a feature request, or want to report a bug?  Please fill out a github
issue_ with the appropriate label.	

.. _issue : https://github.com/hugadams/pyparty/issues

History
=======
``pyparty`` originally began at the George Washington University (2013) in an 
effort to generate test data for SEM and AFM images of gold nanoparticles on glass substrates.
We really enjoyed scikit-image_ for image processing and sought to implement it in generating test data.  
We sought to provide an API for managing labeled particles from real images.  Scikit-image draw and measure
modules laid the groundwork to the core functionality that ``pyparty`` attempts to streamline. 

I should also note that some of the inspiration came from the excellent ``Analyze Particles`` features
in ImageJ_.

   .. _ImageJ : http://rsbweb.nih.gov/ij/


About the Author
================

I'm a PhD student at GWU (check me out on researchgate_, Linkedin_ or twitter_)
and former Enthought intern. I work in biomolecule sensing and plasmonics.   

   .. _researchgate : https://www.researchgate.net/profile/Adam_Hughes2/?ev=hdr_xprf
   .. _Linkedin : http://www.linkedin.com/profile/view?id=121484744&goback=%2Enmp_*1_*1_*1_*1_*1_*1_*1_*1_*1_*1_*1&trk=spm_pic
   .. _twitter : https://twitter.com/hughesadam87

Acknowledgements
================
Thank you scikit-image team for their patience and assistance with us on the 
mailing list, and for putting together a great library for the community.

Thank you countless developers who have patiently answered hundreds of 
my questions on too many mailing lists and sites to list.

Thank you `Zhaowen Liu`_ for all of your help with this project, our 
other projects and for your unwaivering encouragement (and for the panda).

    .. _`Zhaowen Liu` : https://github.com/EvelynLiu77
