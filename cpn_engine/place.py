# -*- coding: utf-8 -*-
"""
Created on Wed May 28 12:01:03 2025

@author: botto
"""



class Place:
        
    nextPlace = 0
    
    
    def __init__(self, name, topic, guard):
        
        self._name = name
        self._topic = topic
        self._guard = guard
        self._pool = None
        self._id = Place.nextPlace    
        Place.nextPlace += 1
        
    def __hash__(self):       
        return self._id
    
    
    def getName(self):
        
        return self._name
    
    
    def getId(self):
        
        return self._id
    
    
    def getTopic(self):
        
        return self._topic

   
    def getContent(self):
        
        return self._pool
    
    def getGuard(self):
        
        return self._guard
    
    
    def checkPresence(self, token):
        
        return token == self.getContent()
    
        
    def insert(self, token):       
        self._pool = token
        
        
    def empty(self):
        self._pool = None

    
            