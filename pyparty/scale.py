import pprint
import numpy as np

class ConverterError(Exception):
    pass

class Converter(object):
    """
    blah blah blah

    Parameters
    ----------
    foo : int
        The starting value of the sequence.

    Raises
    ------
    ConverterError
       If condition foo is met or not met...
       blah blah ...

    Notes
    -----
    blah blah blah
    blah blah blah

    See Also
    --------
    baz : description of baz
    
    Examples
    --------
    >>> np.linspace(2.0, 3.0, num=5)
        array([ 2.  ,  2.25,  2.5 ,  2.75,  3.  ])
    >>> np.linspace(...)
        ...

    """
    
    def __init__(self, unit=None, scale=None, baseunit = 'pixel'):
        ''' Initialize Converter with unit/scale '''
                   

        if baseunit:
            self.baseunit = baseunit
        else:
            self._baseunit = None

        if scale:
            self.scale = scale
        else:
            self._scale = None

        if unit:
            self.unit = unit
        else:
            self._unit = None


    @staticmethod
    def r2(value):
        ''' Return string of value rounded to two digits.'''
        return str(round(value, 2))
            
    @property
    def baseunit(self):
        return self._baseunit
    
    @baseunit.setter
    def baseunit(self, value):
        try:
            self._baseunit = str(value)
        except Exception:
            raise ConverterError('baseunit could not be converted to string')

            
    @property
    def unit(self):
        return self._unit
    
    @unit.setter
    def unit(self, value):
        try:
            self._unit = str(value)
        except Exception:
            raise ConverterError('unit could not be converted to string')
        
    @property
    def scale(self):
        return self._scale
    
    @scale.setter
    def scale(self, value):
        try:
            self._scale = float(value)
        except Exception:
            raise ConverterError('scale could not be converted to float')            


    def convert(self, value):
        ''' converts a single or an array. '''

        # OK, JUST DO A MAPPINT!
        if hasattr(value, '__iter__'):
            intype = type(value)
            if intype == np.ndarray:
                return self._convert_array(value)

            else:
                scale_float = lambda x: float(x) * self.scale
                return intype(map(scale_float, value))
        
        try:
            float(value)
        except Exception:
            raise ConverterError('value could not be converted to float') 
        
        return self.scale * float(value)
    
    def _convert_array(self, value_iterable):
        values = np.asarray(value_iterable, dtype='float')
        return values * self.scale
        
    
    def show_convert(self, value):
        if hasattr(value, '__iter__'):
            raise ConverterError('single value required; got iterable of '
                'length %s' % len(value))
                                 
        outval = self.convert(value)
        print '%s %s  [ %s %s X %s ]' % (self.r2(outval), self.unit, 
            self.r2(value), self.baseunit, self.scalestring)
        return outval

    @property
    def scalestring(self):
        ''' String of scale value with units ie 50 (nm/pixels)'''

        return '%s (%s / %s)' % (round(self.scale,2), self.unit, self.baseunit)
        
    def show(self):
        INDENT = 4
        JUSTIFY = 15
        gap = ' ' * INDENT

        attrs = ['baseunit','unit']
        
        out = '\n'.join( (gap + k.ljust(JUSTIFY) + gap + str(getattr(self, k)) ) \
                        for k in attrs)
        
        out += '\n' + gap + 'scale'.ljust(JUSTIFY) + gap + self.scalestring
        
        print 'Converts %s to %s:\n%s' % (self.baseunit, self.unit, out)


    def invert(self):
        """
        Create a new instance from an epd platform string (e.g. 'win-32')
        """
        unit = self.baseunit
        baseunit = self.unit
        scale = 1.0/self.scale

        return Converter(unit=unit, scale=scale, baseunit=baseunit)

        
if __name__ == '__main__':
    x=Converter(unit='nm')
    x.scale=50.5
    x.show()  
    x.invert().show()
    y=x.invert()
    print y.show_convert(10)
        
    