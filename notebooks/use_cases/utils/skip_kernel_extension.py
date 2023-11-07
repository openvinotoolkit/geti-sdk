# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 00:07:39 2017
@author: Robbe Sneyders
"""

def skip(line, cell=None):
    '''Skips execution of the current line/cell.'''
    if eval(line):
        return
        
    get_ipython().ex(cell)
    
def load_ipython_extension(shell):
    '''Registers the skip magic when the extension loads.'''
    shell.register_magic_function(skip, 'line_cell')

def unload_ipython_extension(shell):
    '''Unregisters the skip magic when the extension unloads.'''
    del shell.magics_manager.magics['cell']['skip']
