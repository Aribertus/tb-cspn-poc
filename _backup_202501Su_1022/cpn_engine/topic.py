# -*- coding: utf-8 -*-
"""
Created on Fri May 30 12:54:06 2025

@author: botto
"""


class Topic:
    
    
    topicFactory = dict()
    
    def __init__(self, name, topicType, relevance = 0.5):
        
        if name not in Topic.topicFactory:
                   
            self._name = name
            self._relevance = relevance
            
            if isinstance(topicType, type):
                self._dsType = topicType
                                         
            elif topicType != None:
                self._dsType = type(topicType)
                
            else:
                raise Exception(f'It is not possible recover a type from argument {topicType}')
                
            Topic.topicFactory[name] = self
            
        else:
            raise Exception(f'Topic with name {name} is already existing')
        
            
            
        def __eq__(self, o):
            
            return self.getName() == o.getName() and self.getTopicType() == o.getTopicType()
            

    def getName(self):
            
        return self._name
    
    
    def getRelevance(self):
        
        return self._relevance
    
    
    def getTopicType(self):
        
        return self._dsType
    
    def modifyRelevance(self, newRel):
        
        self._relevance = newRel
    
