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
from _configobjhunt import Parameters

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

_ROUND = 2 # round to how many units
_INDENT = ' ' * 3


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
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
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
        
        def getPARAM(attr, null=None):
            """ Geattr on self.PARAMS with optional null to return empty lists
            or dicts instead of None """
            return getattr(self.PARAMS, attr, null)
                
        def mkfromrootdir(dirname):
            """ Make a directory relative to self.ARGS.root; log it """
            fullpath = op.join(self.ARGS.outroot, dirname)
            self.logmkdir(fullpath)
            return fullpath
     
               
        @continue_on_fail
        def MPLOT(mcattr, mcpltfcn, mcpltkwds, outdirectory):
            """ boilerplate reduction; if multiplot parameter, runs the plot."""
            
            if getPARAM(mcattr):
                mcpltfcn, mcpltkwds = getattr(mc, mcpltfcn), getPARAM(mcpltkwds, null={})
                mcpltfcn(**mcpltkwds)                    
                self.logsavefig(op.join(outdirectory, getPARAM(mcattr) ))                  
            
    
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
        
        def sumwrite(string, newlines='\n\n', indent=''):
            summary.write(indent + string + newlines)
        
        sumwrite(mc.__repr__())
        
        multidir = mkfromrootdir(self.PARAMS.multidir)
    
        MPLOT('multihist', 'hist', 'multihistkwds', multidir)
        MPLOT('multipie', 'pie', 'multipiekwds', multidir)
        MPLOT('multishow', 'show', 'multishowkwds', multidir)

    
        # CANVAS OPERATIONS
        #-----------------------        
        #Early exit if not canvas operations
        if not getattr(self.PARAMS, 'canvasdir', None):
            return


        @continue_on_fail
        def canvas_stats(canvas, name='STATS'):
            """ Writes a text stream of various canvas parameters. """
            head = '### %s ###'% name            
            sumwrite(head, newlines='\n')
            sumwrite('-' * len(head))

            sumwrite(canvas.__repr__())
            sumwrite("Image Resolution: %s"% str(canvas.rez), indent=_INDENT)
            sumwrite("Particle coverage: %.2f%%" % 
                     round(100*canvas.pixarea, _ROUND), indent=_INDENT)

            for attr in getPARAM('summary_attr'):
                val = getattr(canvas, attr)
                xmin, xmax = min(val), max(val)
                sumwrite("%s (min, max):   (%.2f - %.2f)" % (attr, xmin, xmax), 
                         indent=_INDENT)
            sumwrite('')

        @continue_on_fail
        def canvas_hist(canvas, attr, savepath=None, **histkwds):
            """ Histogram of canvas attribute """

            attr_array = getattr(canvas, attr)
            plt.hist(attr_array, **histkwds)
            plt.xlabel(attr)
            plt.ylabel('counts')
            if savepath:
                self.logsavefig(savepath)

        
        self.LOGGER.info("Creating Canvas List")

        # List of new canvas and canvas-by-canvas breakdown
        total_canvas = mc.to_canvas(mapcolors=True)  

        #X Don't mess w/ order, net canvas must be first (see below idx==0)
        ALLITEMS = [(getPARAM('canvasdir', null='Net Canvas'), total_canvas)]
        if getPARAM('canvas_by_canvas'):
            ALLITEMS.extend(mc.items())
            canbycandir = mkfromrootdir('canvas_by_canvas')
      

        # Stats of each canvas pairwise
        for idx, (name, canvas) in enumerate(ALLITEMS):
            if idx == 0:
                workingcanvasdir = mkfromrootdir(name)
            else:
                workingcanvasdir = mkfromrootdir('%s/%s' % (canbycandir, name))

            autocolor = None
            if getPARAM('autocolor'):
                if idx != 0:
                    autocolor = mc._request_plotcolors()[idx-1]

            # Set canvas background
            if getPARAM('canvas_background'):
                canvas = canvas.set_bg(getPARAM('canvas_background'))

            canvas_stats(canvas, name)
            
            # Color/ Gray/ Binary image  #Maybe refactor in future boilerplate
            if getPARAM('colorimage'):
                colorkwds = getPARAM('showkwds', null={})
                canvas.show(**colorkwds)
                self.logsavefig(op.join(workingcanvasdir, '%s_colored.png' % name))
                
            if getPARAM('grayimage'):
                graykwds = getPARAM('graykwds', null={})
                if 'cmap' not in graykwds:
                    graykwds['cmap'] = 'gray'
                canvas.show(**graykwds)
                self.logsavefig(op.join(workingcanvasdir, '%s_gray.png' % name))

                
            if getPARAM('binaryimage'):
                binarykwds = getPARAM('binarykwds', null={})
                binarykwds.update({'cmap':'pbinary'})
                canvas.show(**binarykwds)
                self.logsavefig(op.join(workingcanvasdir, '%s_binary.png' % name))


            # Scatter plots
            for (x,y) in getPARAM('scatter', null=[]):

                scatterkwds = getPARAM('scatterkwds')
                
                if autocolor and 'color' not in scatterkwds:       #Don't use update         
                    canvas.scatter(attr1=x, attr2=y, color=autocolor, **scatterkwds)
                else:
                    canvas.scatter(attr1=x, attr2=y, **scatterkwds)
                self.logsavefig(op.join(workingcanvasdir, '%s_scatter.png' % name))
                

            # Generate histograms
            
            for attr in getPARAM('summary_attr'):
                savepath = op.join(workingcanvasdir, '%s_hist.png'%name)
                histkwds = getPARAM('histkwds', null={})

                if autocolor and 'color' not in histkwds:
                    canvas_hist(canvas, attr, color=autocolor, 
                                savepath=savepath, **histkwds)
                else:
                    canvas_hist(canvas, attr, savepath=savepath, **histkwds)

        
        summary.close()
        
    
    def load_parameters(self):
        """ Load Parameters object from config file using imp package:
        http://stackoverflow.com/questions/4970235/importing-a-module-dynamically-using-imp
            
        Notes:
        -----
        Also inspects the user-set params against the class attributes.  If they
        are invalid, raises error.
        """
        #modulename = op.splitext(op.basename(self.ARGS.config))[0]
        #try:
            #source = imp.load_source(modulename, self.ARGS.config)
        #except Exception as exc:
            #raise ParserError("Failed to load parameter model from config file:\n %s" % exc)
        #return source.Parameters(**self.ARGS.params)
        return Parameters(**self.ARGS.params)
    
    
    def parse(self):
        """ Returns arguments in parser.  All validation is done in separate
        method so that self.LOGGER can be first set in __main__()."""
                        
        parser = argparse.ArgumentParser(
            prog=SCRIPTNAME,
            usage='%s <image> <outdir> --options' % SCRIPTNAME,
            description = 'Measure and separate objects in image based on color.',
            epilog = 'Please consult tutorial: http://nbviewer.ipython.org/github/hugadams/pyparty/blob/master/examples/Notebooks/objecthunt_tutorial.ipynb'
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
                            ' this will remove entire directory tree!')
    
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
    
