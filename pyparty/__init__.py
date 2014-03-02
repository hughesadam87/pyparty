#Promote the most useful functions, but no system in place
import os.path as op

from pyparty.tools import Canvas, concat_canvas
from pyparty.tools.grids import Grid, CartesianGrid
from pyparty.utils import showim, any2rgb, crop, splot, to_normrgb
from pyparty.multi import MultiCanvas

#  Borrowing API from skimage
pkg_dir = op.abspath(op.dirname(__file__))
data_dir = op.join(pkg_dir, 'data')
bundled_dir = op.join(pkg_dir, 'bundled')