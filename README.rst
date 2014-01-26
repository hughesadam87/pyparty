.. image:: pyparty/data/coverimage.png
   :height: 100px
   :width: 200 px
   :scale: 50 %
   :alt: alternate text
   :align: left
   
   
======================================
pyparty: Python (py) particles (party) 
======================================

``pyparty`` is a library for drawing, labeling, patterning and manipulating 
particles in 2d images.  ``pyparty`` was built primarily over the excellent
image processing library, scikit-image_.

   .. _scikit-image: http://scikit-image.org
   
Overview and Features
---------------------

``pyparty`` provides a simple API for particle analysis in 2d images, while streamlining
several other facets of image processing.  While it is designed with particles in mind, 
many of its features may have broader appeal.

*Some key features include*:

1. Pythonic **ParticleManager** for storing and manipulating particles from image 
   labels OR builtin geometric shapes.
       - All **Particle** types share quick access to descriptors as well as
         general operations like rotations and translations.

2. A **Grid** system for patterning particles, as well as 2d functions for 
   image backgrounds.

3. A **Canvas** to easily integrate *Grids*, *Particles* and flexible *Backgrounds*. 
   In addition, Canvas also provides simplified interfaces for:
      - binarization / thresholding
      - plotting
      - pain-free access to colored, gray and binary image representations

4. A plotting API that supports both *matplotlib.imshow* AND `matplotlib patches`_.

5. Flexible color designations ('red', (1,0,0), 00FF00), and strict typing
   to ensure consistency in data and plots.

6. General ndarray operations such as rotations and translations supported by ALL particle types.

7. iPython Notebook tutorials.

In essence, ``pyparty`` integrates pre-existing plotting and image processing tools 
in a way that hopefully will help simplify many common operations, 
especially in regard to particle analysis.

   .. _`matplotlib patches` : http://matplotlib.org/examples/api/patch_collection.html

What are some use cases, and will pyparty help to me?
-----------------------------------------------------

Some operations that pyparty would be particularly suited for would be:

1. Counting cells in an image and measuring their eccentricity.

2. Patterning a grid of particles over a shadowed background to compare performance
   of thresholding algorithms.

3. Manipulating particles based on descriptors.  For example:

   - delete all particles that have area > 50 pixels.
   - sort and color ellipses in order of increasing eccentricity.
   - dilate all particles appearing in bottom half of an image

4. Plot particles as masks, or matplotlib patches side-by-side.

In short, you may consider using ``pyparty`` if you are doing image analysis and find 
generating, managing or labeling particles as a bottleneck.  

   .. _patchcollection : http://matplotlib.org/examples/api/patch_collection.html

Documentation
=============

The current documtation/testing suite is a series of example notebooks 
(`iPython Notebook`_), which cover most of the basics. These have been linked below:

   - `Basic Operations: Primary Tutorial`_ 
   - `Intro to Shapes`_
   - `Intro to Grids`_
   - `Particles from Image Labels`_
   - `Watershedding Example Adapted`_
   - `Matplotlib Color Maps`_
   
   .. _`Basic Operations: Primary Tutorial`: http://nbviewer.ipython.org/github/hugadams/pyparty/blob/master/examples/Notebooks/basictests.ipynb?create=1
   .. _`Intro to Shapes`: http://nbviewer.ipython.org/github/hugadams/pyparty/blob/master/examples/Notebooks/shapes.ipynb?create=1
   .. _`Intro to Grids` : http://nbviewer.ipython.org/github/hugadams/pyparty/blob/master/examples/Notebooks/grids.ipynb?create=1
   .. _`Particles from Image Labels`: http://nbviewer.ipython.org/github/hugadams/pyparty/blob/master/examples/Notebooks/Analyze_Particles.ipynb?create=1
   .. _`Matplotlib Color Maps` : http://nbviewer.ipython.org/github/hugadams/pyparty/blob/master/examples/Notebooks/gwu_maps.ipynb?create=1
   .. _`Watershedding Example Adapted` : http://nbviewer.ipython.org/github/hugadams/pyparty/blob/master/examples/Notebooks/watershed.ipynb?create=1

Notebooks were initialized with ``--pylab inline``; that is:

   ipython notebook --pylab=inline
   
Having trouble viewing/editing notebooks?  Consider using `Enthought
Canopy`_, which has a notebook kernel builtin, as well as a graphical package manager. 
For simple viewing, paste the github url of each notebook into the iPython Notebook viewer_. 
 
   .. _documentation: http://hugadams.github.com/pyparty/
   .. _`iPython Notebook`: http://ipython.org/notebook.html?utm_content=buffer83c2c&utm_source=buffer&utm_medium=twitter&utm_campaign=Buffer
   .. _`Enthought Canopy`: https://www.enthought.com/products/canopy/
   .. _viewer: http://nbviewer.ipython.org/

History
=======
``pyparty`` originally began at the George Washington University (2013) in an 
effort to generate test data for SEM and AFM images of gold nanoparticles on glass substrates.
We really enjoyed scikit-image_ for image processing and sought to implement it in generating test data.  
We sought to provide an API for managing labeled particles from real images.  Scikit-image draw and measure
modules laid the groundwork to the core functionaly that ``pyparty`` attempts to streamline. 

I should also note that some of the inspiration can from the excellent ``Analyze Particles`` features
in ImageJ_.

   .. _ImageJ : http://rsbweb.nih.gov/ij/

License
=======

BSD_

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
registered in pip. (Checkit it out on PyPi_)

   .. _PyPi : https://pypi.python.org/pypi/pyparty

Pip Install
-----------

Make sure you have pip installed:

    sudo apt-get install python-pip
    
Then:
   
    pip install pyparty
    
To install all of the dependencies, download ``pyparty`` from github, navigate
to the base directory and type:

    pip install -r requirements.txt


Installation from source
------------------------

In the ``pyparty`` base directory run:

    python setup.py install

The developmental version can be cloned from github:

    git clone https://github.com/hugadams/pyparty.git
    
This will not install any dependencies (see above)
    
    
Related Libraries
=================
Interested in the Python ecosystem?   Check out some of these related libraries:

   - SciPy_ (Collection of core scientific libraries)
   - NumPy_ (Fundemental vectorized numerics package in Python)
   - matplotlib_ (Defacto static plotting in Python)
   - pandas_ (Data analysis library : inspired ``pyparty``` ParticleManager API)
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
   - Individual tile operations on grids.
   - More real-world examples.

About the Author
================

I'm a PhD student at GWU.  I work in biomolecule sensing and nanophotonics; you 
can check me out on researchgate_.  Last summer, I interened at Enthought and 
really enjoy software design.  Like any PhD student, my time is apportioned across
many project.  As such, my source code is messy at times, and a test suite hasn't been
developed yet.  I know this is cardinal sin uno, but developing
the iPython notebooks alongside the code helped served as a basic test platform.  
If anyone wants to assist in this effort, I'd be greatly indebted to you.

   .. _researchgate : https://www.researchgate.net/profile/Adam_Hughes2/?ev=hdr_xprf
