# -*- coding: utf-8 -*-
"""
Created on Sat May 31 17:14:53 2025

@author: botto
"""

class ComputeToken:
    
    def __init__(self, function):
        
        self._function = function
        
    def getFunction(self):
        
        return self._function
        
        
    def execute(self, foundTokens):
                        
        return self.getFunction()(foundTokens)