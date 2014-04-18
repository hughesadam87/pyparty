###########################################################################
## PLEASE COPY CONTENTS OF THIS FILE INTO A NEW CONFIG FILE IF OVERWRITING
## OTHERWISE DEFAULT SCRIPT CONFIGURATION WILL BE ERASED
## THIS FILE CAN BE REDOWNLOADED AT: ???
###########################################################################
import inspect
import os.path as op

from traits.api import HasTraits


class ParametersError(Exception):
    """ """

class Parameters(HasTraits):
    """ Stores parameters for the run.
    
    Notes
    -----
    HasTraits subclass allows any parameter to be overwritten without
    defining an __init__().  Useful for overwritting arguments
    directly from command line.
    """

    
    # Multicanvas Options
    # -------------------
   
    multidir = 'multi'   # Directory name where multicanvas plots saved

    storecolors = True     # Object colors from image are retained
    ignore = 'white'
    mapper = [
        ('patch1','black'),
        ('patch2','red'),
        ('patch3', (0,1,0) ),
        ('patch4', 'magenta')
        ]
    
    # Histogram 
    multihist = 'hist.png' 
    multihistkwds = {
        'stacked':True,
         }

    # Pie chart
    multipie = 'pie.png'
    multipiekwds = {}

    multishow = 'multiplot.png'
    multishowkwds = {'nolabel': True,  'names':True}


    # Canvas Options
    # --------------
    
    canvasdir = 'canvas' # Directory where canvas plots/summary saved    
                         # if None, no canvas is created
                         
    grayimage = 'grayimage.png'
    binaryimage = 'binaryimage.png'
    scatter = [
              ('circularity', 'minor_axis_length'),
              ('area', 'equivalent_diameter')
              ]
    
    

    # Methods (Don't mess with these)
    # ---------------------------------
    def __init__(self, *args, **kwargs):
        """ Prevent setting of instance attr that are not class attr. """
        
        cname = self.__class__.__name__
        modulename = op.splitext(op.basename(__file__))[0]
        
        # Make sure no keywords are in class attributes prior to initialization
        for key in kwargs:
            if key not in dict(self._class_attributes()):
                raise ParametersError('Invalid parameter "%s" is not a class '
                   'attribute of %s.%s' % (key, modulename, cname) )

        super(Parameters, self).__init__(*args, **kwargs)

    
    def _class_attributes(self):    
        """ If called before init, just class attributes.  Otherwise, instance
        attributes are included as well.  Returns GENERATOR! """

        attributes = inspect.getmembers(self, lambda a:not(inspect.isroutine(a)))
        return (a for a in attributes if not(a[0].startswith('__') 
                            and a[0].endswith('__')))
            
    
    def parameter_stream(self, delim='\t'):
        """ Summarize all parameters into a stream """
        return [k+':'+delim+str(v) for k,v in tuple(self._class_attributes())]
            
    

if __name__ == '__main__':
#    x=Parameters()
    y=Parameters(storecolors=False, foo='bar')
    test = y._class_attributes()
    print type(test), test.keys()
    
 #   print x.storecolors, y.storecolors