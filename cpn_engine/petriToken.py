# -*- coding: utf-8 -*-
"""
Created on Wed May 28 17:37:29 2025

@author: botto
"""

class Token:
    
    def __init__(self, topic, value):
        
        self._topic = topic
        
        if type(value) == topic.getTopicType():          
            self._value = value
                        
        else:          
            self._value = None
            
        
    def __eq__(self, token):
        
        if token != None:
        
            myTopic, tokTopic = self.getTopic(), token.getTopic()
        
            return  myTopic.getName() == tokTopic.getName() and self.getValue() == token.getValue()

        else:
            return False
    
    def getTopic(self):
        
        return self._topic
    
    
    def getValue(self):
        
        return self._value

def __str__(self):
    return f"Token(topic={self.getTopic().getName()}, value={self.getContent()})"

    