from pyparty.shape_models.basic import Circle, BezierCurve, Ellipse, \
     Polygon, Line, Rectangle, Square
from pyparty.shape_models.multi import Dimer, Trimer, Tetramer

GROUPEDTYPES= \
    {
    'simple':
        {
        'circle': Circle,
        'bezier' : BezierCurve,
        'ellipse' : Ellipse,
        'line' : Line, 
        'polygon' : Polygon,
        'rectangle' : Rectangle,
        'square' : Square,
         },

    'circle_multi':
        {
        'dimer' : Dimer,
        'trimer' : Trimer,
        'tetramer' : Tetramer            
        }
    }

ALLTYPES = dict(GROUPEDTYPES['simple'].items() + 
                GROUPEDTYPES['circle_multi'].items() )

# Eventually, break these up between basic, multi etc...