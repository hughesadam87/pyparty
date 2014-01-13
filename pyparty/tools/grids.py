from __future__ import division
from traits.api import *
import numpy as np
from math import sqrt as msqrt
from pyparty.utils import mem_address
from pyparty.config import _PAD
from pyparty.tools.arraytools import array2sphere, column_array, translate, \
     rotate, astype_rint


def rint(x): return int(round(x,0))

class GridError(Exception):
    """ """

def ztrig(xx, yy):
    """ Example of returning a trig function applied on grid """
    return np.cos(xx) + np.sin(yy)

def fade(xx, yy, style='h', reverse = False, fadefcn=None):
    """ Mesh with a constant fade.  Can be hoirzonal, vertical or diagonal.
    If reverse, fade direction flips (ie right to left).  Optional fadefcn
    will apply a function AFTER the the fading."""

    if reverse:
        xx, yy = -xx, -yy
    if style == 'h' or style == 'horizontal':
        out = xx
    elif style == 'v' or style == 'vertical':
        out = yy
    elif style == 'd' or style == 'diagonal':
        out = xx + yy
    else:
        raise GridError('invalid style "%s".  '
            'Choose form ("side", "top", "corner".')
    if fadefcn:
        return fadefcn(out)
    return out

def unzip_array(ndarray):
    """ Inverse operation of arraytools.unzip_array.  Make sure that is not in
    use and then replace"""
    return tuple(ndarray.T)

class Grid(HasTraits):
    """ Wraps np.meshgrid; main purpose in pyparty is to create and map functions
    to images.  I would not use this for quantitative 2d functions as I've chosen
    initial parameters based on if they produce the correct behavior when plotted
    with imshow(); this may not mean they are numerically veritable as imshow()
    is somewhat liberal in its own normalization schema."""
    
    # 0 should work with these!
    xstart = Int(1)
    ystart = Int(1)
    
    xend = Int(512)
    yend = Int(512)
    
    xpoints = Int(10)
    ypoints = Int(10)
    
    # Avoids intermediate values _xspainc
    xspacing = Property()
    yspacing = Property()

    # REMOVE THIS AND MAKE POLAR GRID ITS OWN THING
    polar = Bool(False)
    
    # MUST RETURN (N X N) array.  IE xx + yy, or xx.  
    zfcn = Function()

    def __init__(self, *args, **kwargs):

        # Still bad; need a general way to pop fcn args!
        zfcn = kwargs.pop('zfcn', fade) #Default is left fade
        if zfcn == 'fade':
            style = kwargs.pop('style', None)
            reverse = kwargs.pop('reverse', None)
            fadefcn = kwargs.pop('fadefcn', None)
            
            def _fade(xx, yy, style=style, reverse=reverse, fadefcn=fadefcn):
                return fade(xx, yy, style, reverse, fadefcn)
            zfcn = _fade
        
        self.zfcn = zfcn
        super(Grid, self).__init__(*args, **kwargs)
            

    def empty_bool(self):
        """ Return False boolean array; used in many related methods """
        return np.zeros(self.shape).astype(bool)

    @property
    def xspacing(self):
        return (self.xend - self.xstart) / self.xpoints
    
    @property
    def yspacing(self):
        return (self.yend - self.ystart) / self.ypoints   
    
    @property
    def diagonalspacing(self):
        return msqrt(self.xspacing**2 + self.yspacing**2)
    
    @xspacing.setter
    def xspacing(self, xsp):
        self.xpoints = rint( (self.xend - self.xstart) / xsp )
    
    @yspacing.setter
    def yspacing(self, ysp):
        self.ypoints = rint( (self.yend - self.ystart) / ysp )
   
    # This would be good property to caches
    @property
    def mesh(self):
        """ [XX, YY] (each 512, 512)"""
        x = np.linspace(self.xstart, self.xpoints, self.xend)
        y = np.linspace(self.ystart, self.ypoints, self.yend)
        if not self.polar:
            return np.meshgrid(x,y) #XX, YY      
        

    def rotate(self, theta):
        """ Theta in degrees.  Problem is that rotating grid result in negative 
        indicies in rr, cc.  Therefore, don't want to do image[rr,cc].  Returns 
        (2,N) indicies."""
        newgrid = rotate(column_array(self.grid), theta, center=self.midpoint)
        logger.warn("Rotations will lead to astry indicies")
        return unzip_array(astype_rint(newgrid))        
    
        
    @property
    def meshindex(self):
        """ Indicies of meshgrid: not sure if useful atm """
        x = np.linspace(self.xstart, self.xpoints, self.xend)
        y = np.linspace(self.ystart, self.ypoints, self.yend)
        return np.meshgrid(x,y, indexing='ij')   
    
        
    @property
    def xx(self):          
        return self.mesh[0]
   
    @property
    def yy(self):
        return self.mesh[1]
        
    @property
    def zz(self):
        return self.zfcn(self.xx, self.yy)
    
    @property
    def shape(self):
        """ For convienence; forget to always check zz"""
        return self.zz.shape     
    
    @property
    def midpoint(self):
        """ (xmid, ymid) ; useful for meancentering operations """
        return ( rint( (self.xend-self.xstart) / 2), 
                 rint( (self.yend-self.ystart) / 2) 
               ) 
    
    @property
    def gradient(self):
        """ X and Y components of gradient.  gx+gy for net grad field """
        gx, gy = np.gradient(self.zz)
        return gx, gy
    
    
    def __repr__(self):
        """   
        Grid (512 X 512) at 0x359da10:
           X --> 10 divisions (51.1 pixels / div)
           Y --> 10 divisions (51.1 pixels / div)
         """
        address = mem_address(super(Grid, self).__repr__())
        outstring = "%s (%s X %s) at %s:\n" % (self.__class__.__name__, 
            self.shape[0], self.shape[1], address)

        outstring += "%sX --> %s divisions (%s pixels / div)\n" % (
            _PAD, self.xpoints, round(self.xspacing, 1) )

        outstring += "%sY --> %s divisions (%s pixels / div)" % (
            _PAD, self.ypoints, round(self.yspacing, 1) )       
        
        return outstring
        
    
class TiledGrid(Grid):
    """ Provide tiled grid by rounding reslution to integers.  Stores boolean 
    array corresponding to edges, corners and centers.
    
    Notes
    -----
    Edges, corners, centers are returned as len(2) tuples of indicies, the same
    primitive as rr_cc in Particles.  This facilitates easy indexing 
    (eg image[corners]=1) as well as compatibility with utils and arraytools.
    """
    
     
    @property
    def xx(self):          # 0, XPOINTS, RES  (MAYBE RENAME XPONITS)
        return astype_rint(self.mesh[0])
   
    @property
    def yy(self):
        return astype_rint(self.mesh[1])
     
    @property
    def hlines(self):
        return self.gradient[0].astype(bool)
    
    @property
    def vlines(self):
        return self.gradient[1].astype(bool)
    
    @property
    def grid(self):
        """ X and Y edges as boolean"""
        return np.where(self.hlines + self.vlines)
    
    @property
    def quadrants(self):
        """ Return 4-qudrants of grid """
        cx, cy = self.midpoint
        # Is it possible to do this all in one step?
        boolarray = self.empty_bool()  
        boolarray[cx, :] = True
        boolarray[:, cy] = True        
        return np.where(boolarray)
    
    
    @property
    def corners(self):
        """ Default corners: bottom-left corners """
        return np.where(self.hlines * self.vlines)
    
    @property
    def centers(self):
        """ For now, returns indicies (IE TUPLE) instead of matrix.  Still is
        indexable with Image[cx, cy] """
        return self.grid_points(0)
    
    @property
    def gmap(self, where='centers'):
        """ Map a function to each point in the grid"""
        raise NotImplementedError
    
    @property
    def tiles(self):
        area = self.zz.shape[0] * self.zz.shape[1]
        # ROUNDS DOWN
        squaresize = self.xspacing * self.yspacing
        numsquares = rint(area / squaresize)
        raise NotImplementedError
    
    def grid_points(self, which=0):
        """ 0 = center, 1 = bottom left corner, 2 = top left corner, 
        3 = top left corner, 4 = bottom right corner.  Returns integer grid
        in form tuple(2, N)
        
        Notes
        -----
        GRID IS INTENDED TO BE PLOTTED WITH NEGATIVE Y-AXIS!  Thus, top right
        and bottom right are actually reversed so that plotting looks right.
        
        Everything is computed relative to BOTTOM LEFT CORNER, which is the bottom
        right corner of each tile."""
        

        if which == 1 or which == 'tl': 
            return self.corners
        
        elif which == 0 or which == 'c':
            r = .5 * self.diagonalspacing
            theta = 45.0

        elif which == 2 or which == 'bl':
            r = self.yspacing
            theta = 90.0            

        elif which ==  3 or which == 'br':
            r = self.diagonalspacing
            theta = 45.0
            
        elif which == 4 or which == 'tr':
            r=self.xspacing
            theta = 0.0
    
        else:
            raise GridError('Grid points must be 0-4, or "c" or "center" etc...'
                ' %s is invalid' % which)
        
        # Turn into column array; translate, then return to 2,N tuple
        cornerpairs = column_array(self.corners)
        translated = translate(cornerpairs, r, theta)
        return unzip_array(astype_rint(translated))


class CartesianGrid(TiledGrid):
    """ A tiled grid with a left-right fade mesh.  This results in a gradient
    that is constant; hence, it's easy to make a cartesian grid."""

    ## Setting zfcn is not allowed in tiled grid as it is used for gradient
    def __init__(self, *args, **kwargs):
        if 'zfcn' in kwargs:
            raise GridError("Zfcn is fixed for CartesianGrid; use TiledGrid")

        kwargs.update({'zfcn':'fade', 'style':'d'})
        super(CartesianGrid, self).__init__(*args, **kwargs)     

    
if __name__ == '__main__':
    t=CartesianGrid()
    print t
    
#    print t.grid, t.hlines
    import matplotlib.pyplot as plt


    image = np.zeros((512,512))
    #image[t.grid_points(0)]=.2
    #image[t.grid_points(1)]=.4
    #image[t.grid_points(2)]=.6
    #image[t.grid_points(3)]=.8
    image[t.grid_points(4)]=1
    

    image[t.grid]=1
    plt.imshow(image, plt.cm.gray)
#    plt.show()
  #  print t.centers[0], t.corners[0]
  
  
# XXX SCRAP
#-----------

# Polar grid attempts; probably best to make it another class
    
    ## Not working either... owkring w/ matplotlib apparanelt
    #azimuths = np.radians(np.linspace(0, 360,20))
    #zeniths = np.arange(0, 70, 10)            
    #return np.meshgrid(zeniths, azimuths)            
    
    #pairs = np.array(zip(x,y))
    #out =array2sphere(pairs)
    #r, theta = out.T
#    return np.meshgrid(r, theta)