# -*- coding: utf-8 -*-
"""
Created on Wed May 28 18:23:18 2025

@author: botto

"""

class PetriNet:
    
    def __init__(self):
        
        self._transitions = []
        self._places = []
        
    
    def addTrans(self, trans):
        
        self._transitions.append(trans)
        
        
    def addPlace(self, place):
        
        self._places.append(place)
        
        
    def getPlaces(self):
        
        return self._places
            
    
    def getTransitions(self):
        
        return self._transitions
        

    def fireNet(self):
                
        for trans in self.getTransitions():
                        
            if trans.isFireable():
                                
                trans.fire()
                return True
            
        return False
                
                
    def process(self):
        
        flag = True
        
        while flag:
            
            flag = flag and self.fireNet()
            
            