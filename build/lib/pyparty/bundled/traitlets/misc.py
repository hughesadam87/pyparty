# Inspired by numpy's skipif, but uses the full apply_wrapper utility to
# preserve function metadata better and allows the skip condition to be a
# callable.
def skipif(skip_condition, msg=None):
    ''' Make function raise SkipTest exception if skip_condition is true

    Parameters
    ----------
    skip_condition : bool or callable.
        Flag to determine whether to skip test.  If the condition is a
        callable, it is used at runtime to dynamically make the decision.  This
        is useful for tests that may require costly imports, to delay the cost
        until the test suite is actually executed.
    msg : string
        Message to give on raising a SkipTest exception

   Returns
   -------
   decorator : function
       Decorator, which, when applied to a function, causes SkipTest
       to be raised when the skip_condition was True, and the function
       to be called normally otherwise.

    Notes
    -----
    You will see from the code that we had to further decorate the
    decorator with the nose.tools.make_decorator function in order to
    transmit function name, and various other metadata.
    '''

    def skip_decorator(f):
        # Local import to avoid a hard nose dependency and only incur the
        # import time overhead at actual test-time.
        import nose

        # Allow for both boolean or callable skip conditions.
        if callable(skip_condition):
            skip_val = skip_condition
        else:
            skip_val = lambda : skip_condition

        def get_msg(func,msg=None):
            """Skip message with information about function being skipped."""
            if msg is None: out = 'Test skipped due to test condition.'
            else: out = msg
            return "Skipping test: %s. %s" % (func.__name__,out)

        # We need to define *two* skippers because Python doesn't allow both
        # return with value and yield inside the same function.
        def skipper_func(*args, **kwargs):
            """Skipper for normal test functions."""
            if skip_val():
                raise nose.SkipTest(get_msg(f,msg))
            else:
                return f(*args, **kwargs)

        def skipper_gen(*args, **kwargs):
            """Skipper for test generators."""
            if skip_val():
                raise nose.SkipTest(get_msg(f,msg))
            else:
                for x in f(*args, **kwargs):
                    yield x

        # Choose the right skipper to use when building the actual generator.
        if nose.util.isgenerator(f):
            skipper = skipper_gen
        else:
            skipper = skipper_func

        return nose.tools.make_decorator(f)(skipper)

    return skip_decorator

