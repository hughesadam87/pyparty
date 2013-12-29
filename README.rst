=============================================
pyparty: Python (py) particle (party) manager
=============================================

``pyparty`` is an object-oriented library for creating, managing and measuring 
particles/inclusion in images.  This library was built mainly over the excellent
image processing library, scikit-image_.

   .. _scikit-image: http://scikit-image.org

License
=======

BSD

Documentation
=============

The official documentation_ does not yet exist.  Please see **pyparty.examples.Notebooks**
for example gallery.  Having trouble viewing/editing notebooks?  Consider using `Enthought
Canopy`_, which has an `iPython Notebook`_ kernel builtin.  For simply viewing,
paste the github url of each notebook into the iPython Notebook viewer_. 
 
   .. _documentation: http://hugadams.github.com/pyparty/
   .. _`iPython Notebook`: http://ipython.org/notebook.html?utm_content=buffer83c2c&utm_source=buffer&utm_medium=twitter&utm_campaign=Buffer
   .. _`Enthought Canopy`: https://www.enthought.com/products/canopy/
   .. _viewer: http://nbviewer.ipython.org/

Goals and Background
====================

``pyparty`` originally began at the George Washington University (2013) in an 
effort to generate test data for SEM and AFM images of gold nanoparticles on the
surface of optical fibers.  We really enjoyed the power of scikit-image_ and sought
to implement it in generating test data, as well as supplant the ``Analyze Particles``
feature in ImageJ_.  While all of this is possible in scikit-image_, we needed 
better separation between *particles* and *image*.  

In short, you may consider using our package if you require any of the following:
 
   1. The ability to draw shapes/patterns on an arbitrary image.  This can be very
      helpful for creating test data.
   2. Indexing/manipulating by particle attribute.  For example:
       - return all particles that have area > 50 pixels.
       - color ellipses in order of increasing eccentricity.
       - remove all particles appearing in bottom half of image
   3. Import particle data from ImageJ_ or ilastik_ (other sources coming soon) and
      perform subsequent image analysis in Python.

   .. _ImageJ: http://rsb.info.nih.gov/ij/
   .. _ilastik: http://www.ilastik.org/

Dependencies
============
``pyparty`` requires **scikit-image** and **Traits** and their dependencies.  

I developed in Traits_ because it is very suitable for writing clean, type-checked
object-oriented classes.  For example, I didn't want to muck around ensuring that 
particle dimensions were entered as ints (pixels) instead of floats, and Traits_
validation takes care of this nicely.  You will not need to understand or use **Traits**
unless you develop for ``pyparty``; *it is not used in the public API*.  

   .. _Traits: http://code.enthought.com/projects/traits/

Installation
============

In the ``pyparty`` directory (same one where you found this file), execute::

    python setup.py install

The developmental version can be cloned from github:

    git clone https://github.com/hugadams/pyparty.git
 
