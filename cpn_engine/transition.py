# -*- coding: utf-8 -*-
"""
Created on Wed May 28 12:06:16 2025

@author: botto
"""

class Transition:
    
    def __init__(self, name, globalPre, pre = set(), post = dict(), thresholds = dict()):
    
        self._name = name
        self._globalPre = globalPre
        self._preConditions = pre
        self._postConditions = post
        self._relevanceThresholds = thresholds
        
    
    def getName(self):
        
        return self._name
    
        
    def getPreConditions(self):
        
        return self._preConditions
    
    
    def getGlobalPre(self):
        
        return self._globalPre
        

    def getPostConditions(self):
        
        return self._postConditions
    
            
    def getThresholds(self):
        
        return self._relevanceThresholds


    def addToPre(self, place):
        
        self.getPreConditions().add(place)
        
    
    def setGlobalCheck(self, globalCheck):

        self._globalPre = globalCheck
        

    def updatePost(self, place, postAction):
        
        self.getPostConditions().update({place : postAction})
        
    
    def updateThreshold(self, topic, threshold):
        
        self.getThresholds().update({topic : threshold})

        
    def checkPrePlace(self, place):
        
        return (place.getContent() != None) and self.getThresholds().get(place.getTopic().getName()) <=  place.getTopic().getRelevance()
    
        
    def checkAllRequiredPlaces(self):
        
        flag = True
        places = dict()
        
        for place in self.getPreConditions():
            
            flag = flag and self.checkPrePlace(place)
            
            if flag:
                places.update({place : place.getContent()})
                
            else:
                return dict()
            
        return places
    
    
    def checkGlobalTokens(self, places):
                
        return self.getGlobalPre()(places)
   
   
    def isFireable(self):
        
        return self.checkAllRequiredPlaces()
        
        # if places:
            
        #     return self.checkGlobalTokens(places)
        
        # else:
            
        #     return False
        
        
    def prepareRepresentationPre(self, preDict):
        
            representationPre = ['Topic: ' + pl.getTopic().getName() + ' with value: ' +  str(preDict[pl].getValue()) + ' from Place: ' + pl.getName() + '\n' for pl in preDict]
            
            return representationPre

    
    def prepareRepresentationPost(self, postDict):  
        
        representationPost = []

        representationPost = ['Topic: ' + pl.getTopic().getName() + ' with value: ' + str(pl.getContent().getValue()) + ' in Place: ' + pl.getName() + '\n' for pl in postDict]
                 
        return representationPost


    def printRepresentation(self, header, representation):
        
        print (header)
            
        for info in representation:
            
            print(info)
            

    def fire(self):
        
        if self.isFireable():
                        
            places = dict()
        
            for placePre in self.getPreConditions():
                                
                places.update({placePre : placePre.getContent()})
                placePre.empty()
                                
            self.printRepresentation(f'Transition {self.getName()} firing with tokens: \n', self.prepareRepresentationPre(places))
           
                
            postConds = self.getPostConditions()
                        
            for placePost in postConds:
                                                
                placePost.insert(postConds.get(placePost).execute(places))
                                        
            self.printRepresentation(f'Transition {self.getName()} has produced tokens: \n', self.prepareRepresentationPost(postConds))

        return True
            