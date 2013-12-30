from pyparty.shape_models.basic import Circle, BezierCurve, Ellipse, \
     Polygon, Line
from pyparty.shape_models.multi import Dimer, Trimer, Square

GROUPEDTYPES= \
    {
    'simple':
        {
        'circle': Circle,
        'bezier' : BezierCurve,
        'ellipse' : Ellipse,
        'line' : Line, 
        'polygon' : Polygon
         },

    'multi':
        {
        'dimer' : Dimer,
        'trimer' : Trimer,
        'square' : Square            
        }
    }

ALLTYPES = dict(GROUPEDTYPES['simple'].items() + 
                GROUPEDTYPES['multi'].items() )

# Eventually, break these up between basic, multi etc...