.. image:: pyparty/data/coverimage.png
   :height: 100px
   :width: 200 px
   :scale: 50 %
   :alt: alternate text
   :align: left
   
   
=============================================
pyparty: Python (py) particle (party) manager
=============================================

``pyparty`` is a library for creating, managing and measuring 
particles in 2d images.  This library was built mainly over the excellent
image processing library, scikit-image_.

   .. _scikit-image: http://scikit-image.org


What does pyparty do, and will it be useful to me?
--------------------------------------------------

In short, you may consider using our package if you require any of the following:
 
   1. The ability to draw shapes/patterns on an arbitrary image.  This can be very
      helpful for creating *intricate* test data.  Some background generation tools
      are also provided.
   2. Indexing/manipulating by particle attribute.  For example:
       - return all particles that have area > 50 pixels.
       - color ellipses in order of increasing eccentricity.
       - remove all particles appearing in bottom half of image
   3. Manage and manipulated labeled particles from scipy.ndimage.label, and
      perform subsequent analysis in Python.  IO for ImageJ_ / ilastik_ particles
      is forthcoming.

   .. _ImageJ: http://rsb.info.nih.gov/ij/
   .. _ilastik: http://www.ilastik.org/


Background
----------
``pyparty`` originally began at the George Washington University (2013) in an 
effort to generate test data for SEM and AFM images of gold nanoparticles on the
surface of optical fibers.  We really enjoyed the design of scikit-image_ for image processing 
and sought to implement it in generating test data.  We also wanted to provide an API for managing
labeled particles from real images.  All of the tools already existed in scikit-image and scipy.ndimage;
``pyparty`` merely streamlines some of the functionality.  

I should also note that some of the inspiration can from the excellent ``Analyze Particles`` features
in ImageJ_.


Documentation
=============

The official documentation_ doesn't exist yet.  Instead, we provide a series of example notebooks 
(`iPython Notebook`_), which cover most of the basics. Please see **pyparty.examples.Notebooks**
for the current tutorials gallery.  For convienence, these have been linked below:

   - `Basic Operations`_ (Primary Tutorial)
   - `Manage Particles on Image`_
   - `pyparty shapes`_
   
   .. _`Basic Operations`: http://nbviewer.ipython.org/github/hugadams/pyparty/blob/master/examples/Notebooks/Analyze_Particles.ipynb?create=1
   .. _`Manage Particles on Image`: http://nbviewer.ipython.org/github/hugadams/pyparty/blob/master/examples/Notebooks/basictests.ipynb?create=1
   .. _`pyparty shapes`: http://nbviewer.ipython.org/github/hugadams/pyparty/blob/master/examples/Notebooks/basictests.ipynb?create=1

Notebooks were initialized with pylab inline IE:

   ipython notebook --pylab=inline
   
Having trouble viewing/editing notebooks?  Consider using `Enthought
Canopy`_, which has a notebook kernel builtin, as well as a graphical package manager. 
For simply viewing, paste the github url of each notebook into the iPython Notebook viewer_. 
 
   .. _documentation: http://hugadams.github.com/pyparty/
   .. _`iPython Notebook`: http://ipython.org/notebook.html?utm_content=buffer83c2c&utm_source=buffer&utm_medium=twitter&utm_campaign=Buffer
   .. _`Enthought Canopy`: https://www.enthought.com/products/canopy/
   .. _viewer: http://nbviewer.ipython.org/

Please post a github issue if you notice a dead link.  I will update them periodically.   

License
=======

BSD_

   .. _BSD : https://github.com/hugadams/pyparty/blob/master/LICENSE.txt

Dependencies
============
``pyparty`` requires **scikit-image**, **Traits** and their dependencies, which
include many core packages such as ``numpy`` and ``matplotlib``.  

``pyparty`` uses Traits_ because it is well-suited for writing clean, type-checked
object-oriented classes. You will not need to understand or use **Traits**
unless you develop for ``pyparty``; *it is not used in the public API*.  (The **Traits** dependency may be removed in future installments after the 
core functionality is stable.)

   .. _Traits: http://code.enthought.com/projects/traits/
   
Installation
============

I would recommend using `Enthought Canopy`_ to install ``Traits`` and ``scikit-image``; however,
``pyparty`` can also imports pip.  

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
   
Coming Soon
===========

About the Author
================

I'm a PhD student at GWU.  I work in biomolecule sensing and nanophotonics; you can check me out here_.  Last summer, I interened at Enthought and really enjoy software design.  As with any PhD, my time is fairly limited, especially in pursuing sofftware ventures.  As such, you may not find my code documentation up to par, and ``pyparty`` doesn't have any nosetests yet.  I know this is cardinal sin uno, but developing the iPython notebooks alongside the code helped served as a basic test platform.  If anyone feels so compelled to assist in this effort, I'd be forever in your debt.

   .. _here : https://www.researchgate.net/profile/Adam_Hughes2/?ev=hdr_xprf
