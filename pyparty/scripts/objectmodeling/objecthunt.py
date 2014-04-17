#!/usr/bin/env python
# Author:   --<Adam Hughes>
# Purpose: Compute statistics from labeled image
# Created: 4/15/2014

from __future__ import division
import sys
import shutil
import imp
import os
import os.path as op
import argparse #http://docs.python.org/dev/library/argparse.html
import skimage.io as io
import datetime as dt
import matplotlib.pyplot as plt

from pyparty.logging_utils import configure_logger, log, LogExit, logclass
import logging

from pyparty import MultiCanvas


# DEFAULT PARAMETERS 
# ------------------
SCRIPTNAME = 'objecthunt'
DEF_OUTROOT = './output'
DEF_CONFIG = './_configobjhunt.py'
LOGNAME = '%s.py' % SCRIPTNAME # How this file is referred to by logger
LOGFILE = 'runlog.txt' 
PARAMSFILE = 'runparams.txt'
SUMMARYFILE = 'summary.txt'


# CUSTOM ERRORS 
# -------------
class ScriptError(Exception):
    """ """

class ParserError(ScriptError):
    """ """

# Utilities 
# ---------   
def ext(afile): 
    ''' get file extension'''
    return op.splitext(afile)[1]

def timenow():
    return dt.datetime.now()    

def _parse_tfnone(dic):
    """ Replace dictionary values of 'none', 'true' and 'false' by their 
    python types.  """

    formatter = {'none':None, 'true':True, 'false':False}
    out = {}
    for k,v in dic.items():
        try:
            v=formatter[v.lower()]
        except KeyError:
            pass
        out[k] = v
    return out


def logging_decorator(func):
    def wrapper():
        wrapper.count += 1
        print "The function I modify has been called {0} times(s).".format(
              wrapper.count)
        func()
    wrapper.count = 0
    return wrapper


def continue_on_fail(func):
    """ Failure in function will not break script """
    def wrapper():
        try:
            func()
        except LogExit:
            pass
    return wrapper


@logclass(log_name=LOGNAME , public_lvl='info', skip=[])
class ObjectHunter(object):

    def start_run(self):
        tstart =timenow()
        
        self.ARGS = self.parse()        
        self.LOGGER = configure_logger(screen_level=self.ARGS.verbosity, 
                                  name='%s' % LOGNAME, 
                                  mode='w',
                                  logfile=op.join(self.ARGS.outroot, LOGFILE))  
        self.PARAMS = self.load_parameters()
        self.run_main()
        
        tend = timenow()
        self.LOGGER.info('COMPLETE (runtime=%s) see logfile "%s"' % 
                    ((tend-tstart), LOGFILE) )      
        

    # SUBTRACT UP TO ROOT DIR IF CAN DIG UP HOW TO DO THAT FROM SOMEWHERE ELSE
    def logstream(self, fullpath, stream, mode='w'):
        """ Save a file to outdir, default root is outroot """
        self.LOGGER.info('Saving file: %s' % fullpath)
        with open(fullpath, mode) as f:
            f.write(stream)
            
    def logsavefig(self, fullpath):
        self.LOGGER.info('Saving figure: %s' % fullpath)
        plt.savefig(fullpath, dpi=self.ARGS.dpi)
        plt.clf()
    


    def logmkdir(self, fullpath):
        """ Makes directory path/folder, and logs."""
        self.LOGGER.info('Making directory: %s.' % fullpath)
        os.mkdir(fullpath)
    
    # MAIN ROUTINE
    # ------------
    def run_main(self):
        """ Define a series of sub-functions so logging/traceback easier """
                
        def mkfromrootdir(dirname):
            """ Make a directory relative to self.ARGS.root; log it """
            fullpath = op.join(self.ARGS.outroot, dirname)
            self.logmkdir(fullpath)
            return fullpath
     
        @continue_on_fail
        def MHIST():
            if self.PARAMS.multihist:
                mc.hist(**self.PARAMS.multihistkwds)
                self.logsavefig(op.join(multidir, self.PARAMS.multihist))   
    
        @continue_on_fail
        def MPIE():
            if self.PARAMS.multipie:
                mc.pie(**self.PARAMS.multipiekwds)
                self.logsavefig(op.join(multidir, self.PARAMS.multipie))        
            
    
        @continue_on_fail
        def MSHOW():
            if self.PARAMS.multishow:
                mc.show(**self.PARAMS.multishowkwd)
                self.logsavefig(op.join(multidir, self.PARAMS.multishow))        
            
    
        summary = open(op.join(self.ARGS.outroot, SUMMARYFILE), 'a') #make method 

        stream = self.PARAMS.parameter_stream()
        paramspath = op.join(self.ARGS.outroot, PARAMSFILE)
        self.logstream(paramspath, '\n\n'.join(stream))
        
        # MULTICANVAS OPERATIONS
        #-----------------------
        self.LOGGER.info("Creating MultiCanvas...")
        if self.PARAMS.mapper is None:
            self.PARAMS.mapper = []
        mc = MultiCanvas.from_labeled(self.ARGS.image, 
                                      storecolors=self.PARAMS.storecolors, 
                                      ignore=self.PARAMS.ignore,
                                      mapper = self.PARAMS.mapper
                                      )
        
        def sumwrite(string, newlines='\n\n'):
            if newlines:
                summary.write(string + newlines)
            else:
                summary.write(string)
        
        sumwrite(mc.__repr__())
        
        multidir = mkfromrootdir(self.PARAMS.multidir)
    
        MHIST()
        MPIE()
        MSHOW()
    
    
        # CANVAS OPERATIONS
        #-----------------------        
        #Early exit if not canvas operations
        if not self.PARAMS.canvasdir:
            return
        
        canvasdir = mkfromrootdir(self.PARAMS.canvasdir)
        self.LOGGER.info("Creating Canvas...")
        canvas = mc.to_canvas(mapcolors=True)
        sumwrite(canvas.__repr__())
        sumwrite("### STATS ###")
#        sumwrite(
        
        summary.close()
    
    
    def load_parameters(self):
        """ Load Parameters object from config file using imp package:
        http://stackoverflow.com/questions/4970235/importing-a-module-dynamically-using-imp
            
        Notes:
        -----
        Also inspects the user-set params against the class attributes.  If they
        are invalid, raises error.
        """
        modulename = op.splitext(op.basename(self.ARGS.config))[0]
        try:
            source = imp.load_source(modulename, self.ARGS.config)
        except Exception as exc:
            raise ParserError("Failed to load parameter model from config file:\n %s" % exc)
        return source.Parameters(**self.ARGS.params)
    
    
    def parse(self):
        """ Returns arguments in parser.  All validation is done in separate
        method so that self.LOGGER can be first set in __main__()."""
                        
        parser = argparse.ArgumentParser(
            prog=SCRIPTNAME,
            usage='%s <image> <outdir> --options' % SCRIPTNAME,
            description = 'ADD SHORT DESCRIPTION HERE',
            epilog = 'ADD EPILOG HERE'
            )   
        
        parser.add_argument("-t", "--trace", help='Explict traceback in logging', 
                            action='store_true')
    
        parser.add_argument("image", 
                            help='Path to image')
    
        parser.add_argument("outroot", 
                            metavar='out directory', 
                            help='Path to outdirectory')
        
        parser.add_argument("-f", "--force",
                            action='store_true',
                            help='Overwrite outdirectory if it exists.  WARNING:'
                            ' this will overwrite and rename materials at will!')
    
        parser.add_argument('-c', '--config', 
                            default = DEF_CONFIG,  
                            metavar = '', 
                            help = 'Path to config file.  '
                                'Defaults to "%s"' % DEF_CONFIG)   
    
        parser.add_argument('-v', '--verbosity', 
                            nargs='?',
                            default='warning', 
                            const='info',
                            metavar = '', #For printout
                            help='Set screen logging.  If no argument, defaults to'
                                 ' info' )
        
        parser.add_argument('-p','--params', 
                            nargs='*',
                            metavar='',
                            help='Overwrite config parameters manually in form '
                            'k="foo value" (e.g. ?????)')
    
        parser.add_argument('--dpi', 
                            type=int, #is None if not passed
                            help='Plotting resolution (dots per inch)')
    
        
    
        args = parser.parse_args()
    
        # Parse --params
        if not args.params:
            args.params = {}
        else:
            args.params = dict (zip( [x.split('=', 1 )[0] for x in args.params],
                                    [x.split('=', 1)[1] for x in args.params] ))
        args.params = _parse_tfnone(args.params)
        
        # Validate directories; cannot log any of this because logging has to be
        # configured AFTER outdir has been updated.  LEAVE AS IS
        def _validatepath(path, create=False):
            """ Return absolute path; create new directory for outroot. """
            path = op.abspath(path)
            if not op.exists:
                raise ParserError("Path not found: %s" % path)
            return path
    
        # Validate config/image paths
        args.config = _validatepath(args.config)
        args.image = _validatepath(args.image)
    
        #Load image
        args.image = io.imread(args.image)
    
        # Validate/overwirte outdirectory
        outroot = op.abspath(args.outroot)
        if op.exists(outroot):
            if args.force:
                shutil.rmtree(outroot)
            else:
                raise ParserError("Outdirectory %s exists.  Use -f to "
                                  "fully overwrite directory." % outroot)
        os.mkdir(outroot)
        args.outroot = outroot
        return args


def main(*args, **kwargs):
    hunter = ObjectHunter(*args, **kwargs)
    hunter.start_run()    
        
if __name__ == '__main__':
    main()
    
