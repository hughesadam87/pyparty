from __future__ import division
import logging
import numpy as np
import math
import functools
from operator import itemgetter

from traits.api import HasTraits, Int, Bool
from matplotlib.patches import Path, PathPatch
from matplotlib.collections import PatchCollection, PathCollection
from pyparty.utils import mem_address, rgb2uint, crop
from pyparty.config import _PAD
from pyparty.tools.arraytools import array2sphere, column_array, translate, \
     rotate, astype_rint

logger = logging.getLogger(__name__) 

def rint(x): return int(round(x,0))

class GridError(Exception):
    """ """

# XXX : Should these subclass from multiple parents?
class GridImportError(GridError):
    """ """

class GridAttributeError(GridError):
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
                        'Choose form ("side", "top", "corner")')
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

    xdiv = Int(10)
    ydiv = Int(10)

    # REMOVE THIS AND MAKE POLAR GRID ITS OWN THING
    polar = Bool(False)

    def __init__(self, *args, **kwargs):
        """ Smooth grid; object to wrap ZZ 2d fucntion.  Default function
        is a smooth gradient (fadefcn) that can be easily accessed through
        keywords.

        Parameters
        ----------
        rez : (int, int)
            Shortcut for setting grid start, end values.  Note that to be
            consistent with plotting, rez[0] sets yend.

        negative_y : bool (True)
            If rez passed, this will reverse xend and yend to be consistent
            with imshow().  If y-axis looks backwards, make this False.

        fadefcnargs 
            Valid arguments to fade().  Eg reverse, style.  Only use if 
            Grid is to be instantied with a fade/gradient 

        Notes
        -----
        Only fade() is supported at initalization.  Other functions can be
        set using Grid.set_zfcn().
        """

        style = kwargs.pop('style', 'h')
        reverse = kwargs.pop('reverse', False)
        fadefcn = kwargs.pop('fadefcn', None)

        rez = kwargs.pop('rez', None)
        negative_y = kwargs.pop('negative_y', True)
        if rez:
            kwargs['xstart']=0
            kwargs['yend']=0
            # REVERSED
            if negative_y:
                kwargs['xend']= rez[1]
                kwargs['yend']= rez[0]
            else:
                kwargs['xend']= rez[0]
                kwargs['yend']= rez[1]                

        super(Grid, self).__init__(*args, **kwargs)
        self._set_zfcn(fade, style=style, reverse=reverse, fadefcn=fadefcn)

    def set_zfcn(self, fcn_or_string, *args, **kwargs):
        return self._set_zfcn(fcn_or_string, *args, **kwargs)

    def _set_zfcn(self, fcn_or_string, *args, **kwargs):
        """ Private method to allow overloading of set_zfcn"""
        if isinstance(fcn_or_string, str):
            if fcn_or_string != 'fade':
                raise GridError('"fade" is the only implicit zfcn passable via'
                                'string.  Please pass a function instead.')
            # Setup a dictionary like threshfcn soon
            self._zfcn = functools.partial(fade, *args, **kwargs)
            self._zfcntype = fcn_or_string
        else:
            self._zfcn = functools.partial(fcn_or_string, *args, **kwargs)
            self._zfcntype = fcn_or_string.__name__

    @property
    def zfcn(self):
        return self._zfcntype

    @zfcn.setter
    def zfcn(self, val):
        raise GridAttributeError('Please refer to "set_zfcn()')


    def empty_bool(self):
        """ Return False boolean array; used in many related methods """
        return np.zeros(self.shape).astype(bool)

    @property
    def div(self):
        """ X and y div; setter is more useful """
        return (self.xdiv, self.ydiv)

    @div.setter
    def div(self, div):
        """ Return x and y divisions """
        self.xdiv = div
        self.ydiv = div

    @property
    def xspacing(self):
        return (self.xend - self.xstart) / self.xdiv

    @property
    def yspacing(self):
        return (self.yend - self.ystart) / self.ydiv   

    @property
    def diagonalspacing(self):
        return math.sqrt(self.xspacing**2 + self.yspacing**2)

    @xspacing.setter
    def xspacing(self, xsp):
        self.xdiv = rint( (self.xend - self.xstart) / xsp )

    @yspacing.setter
    def yspacing(self, ysp):
        self.ydiv = rint( (self.yend - self.ystart) / ysp )

    # This would be good property to caches
    @property
    def mesh(self):
        """ [XX, YY] (each 512, 512)"""
        x = np.linspace(self.xstart, self.xdiv, self.xend)
        y = np.linspace(self.ystart, self.ydiv, self.yend)
        if not self.polar:
            return np.meshgrid(x,y) #XX, YY      


    def rotate(self, theta):
        """ Theta in degrees.  Problem is that rotating grid result in negative 
        indicies in rr, cc.  Therefore, don't want to do image[rr,cc].  Returns 
        (2,N) indicies."""
        newgrid = rotate(column_array(self.grid), theta, center=self.midpoint)
        logger.warn("Rotations will lead to astry indicies")
        return unzip_array(astype_rint(newgrid))        


    def blend_bg(self, bg, weight=0.5):
        if weight > 1 or weight < 0:
            raise GridError("weight must be between 0 and 1")

        if bg.ndim == 3:
            bg = rgb2uint(bg)
            logger.warn('3-channel image convert to 8-bit uint')

        if bg.shape != self.shape:
            xf, yf = self.shape
            try:
                bg = crop(bg, 0, 0, xf, yf)
            except Exception as E:
                raise GridError('bg shape %s did not match grid shape %s and '
                                'cropping failed with: %s' % (bg.shape, self.shape, E.message))
            else:
                logger.warn("bg image cropped to grid dimensions %s" % self.shape)

        # Normalize the values to balance weight sum
        scale = bg.max() / self.zz.max()
        zzout = self.zz * scale 
        alpha = 1-weight

        return (weight * zzout + ( alpha  * bg) ) #/ scale 

    @property
    def meshindex(self):
        """ Indicies of meshgrid: not sure if useful atm """
        x = np.linspace(self.xstart, self.xdiv, self.xend)
        y = np.linspace(self.ystart, self.ydiv, self.yend)
        return np.meshgrid(x,y, indexing='ij')   


    @property
    def xx(self):          
        return self.mesh[0]

    @property
    def yy(self):
        return self.mesh[1]

    @property
    def zz(self):
        return self._zfcn(self.xx, self.yy) 

    @property
    def shape(self):
        """ For convienence; forget to always check zz"""
        return self.zz.shape     

    @property
    def midpoint(self):
        """ (xmid, ymid) ; useful for meancentering operations """
        return ( rint( (self.xend-self.xstart) / 2), 
                 rint( (self.yend-self.ystart) / 2) ) 

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

        outstring += "%sX --> %s divisions (%.1f pixels / div)\n" % (
            _PAD, self.xdiv, self.xspacing )

        outstring += "%sY --> %s divisions (%.1f pixels / div)" % (
            _PAD, self.ydiv, self.yspacing )       

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

    def __init__(self, *args, **kwargs):
        """ DO NOT REMOVE OR SUBLCASSES WILL NOT INITIALIZE CORRECTLY"""
        return super(TiledGrid, self).__init__(*args, **kwargs)


    @property
    def xx(self):          # 0, XPOINTS, RES  (MAYBE RENAME XPONITS)
        return self.mesh[0].astype(int)

    @property
    def yy(self):
        return self.mesh[1].astype(int)

    @property
    def hlines(self):
        gx = self.gradient[0]
        out = np.zeros(gx.shape)
        # Mask every other column

        # if odd, iterate one more
        itermax = gx.shape[0]
        if itermax % 2:  #IF EVEN PASS; IF ODD, ADD ONE
            for i in range(0, itermax, 2):
                out[i, :] = gx[i, :]

        else:
            for i in range(1, itermax, 2):
                out[i, :] = gx[i, :]            

        out[0,:] = 1.0
        return out.astype(bool)       

    @property
    def vlines(self):
        gy = self.gradient[1]
        out = np.zeros(gy.shape)

        # if even, iterate one more
        itermax = gy.shape[1]
        if itermax % 2:
            for i in range(0, itermax, 2):
                out[:, i] = gy[:, i]
        else:
            for i in range(1, itermax, 2):
                out[:, i] = gy[:, i]

        #FILL LEFT EDGE
        out[:, 0] = 1.0                
        return out.astype(bool)

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
        indexable with Image[cx, cy].  Takes atan2(x,y) relative to top left
        corner, result in a vector facing down, right.  Instead, we translate 
        opposite this (up, left), and then cut out centers that are negative.
        In short, we compute the vector from topleft corner to center, then 
        translate in opposite direction."""

        rxmax, rymax = self.shape

        # Vector from top left corner facing down/right to center
        thetadiag = math.degrees(math.atan(self.xspacing/self.yspacing))
        r = .5 * self.diagonalspacing
        # Translate towards up/left direction.
        thetadiag = thetadiag + 180

        cornerpairs = column_array(self.corners)
        translated = translate(cornerpairs, r, thetadiag)
        rr_cen, cc_cen = unzip_array(astype_rint(translated))

        # THIS IS HOW TO DEFINE 2D MASK PROPERLY
        mask = (rr_cen >= 0) & (rr_cen < rxmax) & (cc_cen > 0) & (cc_cen < rymax)
        return (rr_cen[mask], cc_cen[mask])


    def _diags(self, style='tiles'):
        """ Returns tiles, diagonals or negative diagonals.  Boilerplate 
        reduction; all are based on center coordinates."""

        zz = self.zz
        xhalf = self.xspacing / 2.
        yhalf = self.yspacing / 2.

        out = []
        for idx, (cx, cy) in enumerate( sorted(self.pairs('centers')) ):
            xl, xr = int(cx - xhalf), int(cx + xhalf)
            yl, yr = int(cy - yhalf), int(cy + yhalf)

            xs = range(xl,xr) 

            if style == 'dlines':
                ys = range(yl,yr)

            elif style == 'dlines_neg':
                ys = range(yl,yr)[::-1]

            else:                
                raise GridError('%s style not understood' % style)

            out.append( (xs, ys) )

        return out     


    @property
    def tiles(self):
        return self.as_tiles(key=None, sort=False)


    def as_tiles(self, key=None, sort=True):
        """ Return tiles as list of xs, ys or as dictionary.

        Attributes
        ----------
        key : None, True, 'flat', '2d'
            If None, list is returns (same as self.tiles)
            If True or 'flat', dicionary keyed by index corresponding
            to iterating over self.centers
            If '2d', dict keyed by (I, J) indicies, corresponding to 2d 
            index pairs in range (0-self.xdiv, 0-self.ydiv)

        sort : bool
            On dictionary return, OrderedDict will be returned

        """

        validkeys = [None, True, 'flat', '2d']
        if key not in validkeys:
            raise GridError('as_tiles() key must be "%s";'
                            ' recieved %s' % ('","'.join(validkeys), key) )        

        zz = self.zz
        xhalf = self.xspacing / 2.
        yhalf = self.yspacing / 2.


        tiles = []
        for (cx, cy) in sorted(self.pairs('centers')):

            xl, xr = int(cx - xhalf), int(cx + xhalf)
            yl, yr = int(cy - yhalf), int(cy + yhalf)

            tiles.append( np.meshgrid( range(xl,xr), range(yl,yr) ) )

        if not key:
            return tiles

        tiles = enumerate(tiles)

        if sort:
            try:
                from collections import OrderedDict
            except ImportError:
                raise GridImportError('Tile sorting requires OrderedDict form '
                                      'python.collection package; package is standard in 2.7 and '
                                      'higher')

            tiledict = OrderedDict(tiles)
            tiles2d = OrderedDict()

        else:
            tiledict = dict(tiles)
            toles2d = {}

        # Key == flat
        if key == True or key == 'flat':
            return tiledict

        # Key == 2d
        idx = 0
        for i in range(self.xdiv):
            for j in range(self.ydiv):

                tiles2d[(i,j)] = tiledict.pop(idx)
                idx += 1

        return tiles2d

    @property
    def dlines(self):
        return self._diags('dlines')

    @property
    def dlines_neg(self):
        return self._diags('dlines_neg')    


    def pairs(self, attr):
        """ Return pairs of attributes that are typically stored as len(2) 
        tuple, indicies.  Valid attr include corners, centers, etc..."""
        valid = ['grid', 'centers', 'corners', 'hlines', 'vlines']
        if attr not in valid:
            raise GridError('Invalid grid argument, "%s".  Choose from:  '
                            'True, "grid", "centers", "corners", "xlines", "vlines"' )

        rr_attr, cc_attr = getattr(self, attr)
        # REVERSED X, Y ORDERING
        return zip(*(cc_attr, rr_attr))



    def as_patch(self, **kwds):
        """ matplotlib pathpatch for hlines and vlines

        **pathkwds : any valid PathCollection keyword (linestyle, edgecolor)

        Notes
        -----
        First, finds coordinates of hlines, vlines.  Then, it it finds
        the x, y coords of each line; it extracts the points corresponding to
        each isoline, and makes a path for each of these."""

        hlines = zip(*np.where(self.hlines))
        vlines = zip(*np.where(self.vlines))

        xunique = set([x for x, y in hlines])
        yunique = set([y for x, y in vlines])

        hpaths = []
        vpaths = []

        # REVERSE PATHS FOR SAKE OF PLOT (y,x) !!
        for xun in xunique:
            hpaths.append(Path([(y, x) for x,y in hlines if x==xun]))

        for yun in yunique:
            vpaths.append(Path([(y, x) for x,y in vlines if y==yun]))            

        return PathCollection(hpaths + vpaths, **kwds)


class CartesianGrid(TiledGrid):
    """ A tiled grid with a left-right fade mesh.  This results in a gradient
    that is constant; hence, it's easy to make a cartesian grid."""

    ## Setting zfcn is not allowed in tiled grid as it is used for gradient
    def __init__(self, *args, **kwargs):  
        """ Force diagonal style fadefcn """
        kwargs.update({'style':'d', 
                       'reverse':False, 
                       'fadefcn':None})
        super(CartesianGrid, self).__init__(*args, **kwargs)

    def set_zfcn(fcn_or_str, *args, **kwargs):
        raise GridError("Zfcn is fixed for CartesianGrid; use TiledGrid")