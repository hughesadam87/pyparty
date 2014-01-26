from enthought.traits.api import BaseInt

class IntOrNone(BaseInt):
    """ Traits whose value is an integer or None"""
    default_value = None

    # Describe the trait type
    info_text = 'an integer or None'

    def validate(self, object, name, value):
        if value is None:
            return value        

        try:
            return int(value)
        except ValueError:
            self.error( object, name, value )