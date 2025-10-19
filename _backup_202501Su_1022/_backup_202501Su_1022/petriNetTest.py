# -*- coding: utf-8 -*-
"""
Created on Sat May 31 17:45:37 2025

@author: botto

PetriNet text
"""
from cpn_engine.computeToken import ComputeToken
from cpn_engine.guard import Guard
from cpn_engine.place import Place
from cpn_engine.petriNet import PetriNet
from cpn_engine.petriToken import Token
from cpn_engine.topic import Topic
from cpn_engine.transition import Transition

def goodGains(token : Token):
        
    return token.getValue() >= 800


def containedLosses(token : Token):
        
    return token.getValue() < 500

def checkBuy(token):
    
    return token.getContent() == 'buy'

def dummyCheck(token):
    
    return True

def globalEvaluationToBuy(places):
    
    flag = True
                
    for key in places:
                
        flag = key.getGuard().executeCheck(places.get(key))
            
    return flag


def decideOnBuy(places):
                
    return Token(topic3, "buy" if  globalEvaluationToBuy(places) else "don't buy")



topic1 = Topic('gains', float, 0.7)
topic2 = Topic('losses', float, 0.8)
topic3 = Topic('decision', str)

token1 = Token(topic1, 1000.0)
token2 = Token(topic2, 400.0)


guard1 = Guard(goodGains)
guard2 = Guard(containedLosses)
guard3 = Guard(checkBuy)

place1 = Place('ForGains', topic1, guard1)
place2 = Place('ForLosses', topic2, guard2)
place3 = Place('ForBuyDecision', topic3, guard3)

place1.insert(token1)
place2.insert(token2)

action1 = ComputeToken(decideOnBuy)

transitionOnBuy = Transition('Decider', action1)
decision_token = place3.getContent()

transitionOnBuy.addToPre(place1)
transitionOnBuy.addToPre(place2)
transitionOnBuy.setGlobalCheck(dummyCheck)
transitionOnBuy.updatePost(place3, action1)
transitionOnBuy.updateThreshold('gains', 0.5)
transitionOnBuy.updateThreshold('losses', 0.5)

pn1 = PetriNet()
pn1.addPlace(place1)
pn1.addPlace(place2)
pn1.addPlace(place3)
pn1.addTrans(transitionOnBuy)

print("\n[TEST] Executing Petri Net Transition...\n")

if transitionOnBuy.fire():
    decision_token = place3.getContent()
    print(f"[RESULT] Final decision: topic = {decision_token.getTopic().getName()}, value = {decision_token.getValue()}")
else:
    print("[RESULT] Transition did not fire.")

