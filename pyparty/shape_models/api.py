from pyparty.shape_models.basic import Circle, BezierCurve, Ellipse, \
     Polygon, Line
from pyparty.shape_models.multi import Dimer, Trimer, Square

PARTICLETYPES= \
    {
     'circle': Circle,
     'bezier' : BezierCurve,
     'ellipse' : Ellipse,
     'line' : Line, 
     'polygon' : Polygon,
     'dimer' : Dimer,
     'trimer' : Trimer,
     'square' : Square
     }

# Eventually, break these up between basic, multi etc...