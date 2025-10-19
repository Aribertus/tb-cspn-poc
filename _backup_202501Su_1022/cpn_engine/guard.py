# -*- coding: utf-8 -*-
"""
Created on Wed May 28 17:49:20 2025

@author: botto
"""

class Guard:
    
    def __init__(self, function):
        
        self._function  = function
        
    
    def getFunction(self):
        
        return self._function
        
        
    def executeCheck(self, token):
                
        return self.getFunction()(token)