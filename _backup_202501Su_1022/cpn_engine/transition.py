# -*- coding: utf-8 -*-
"""
Created on Wed May 28 12:06:16 2025

@author: botto
"""

from tb_cspn_observe.logger import open_jsonl
from pathlib import Path

# Ensure the runs/ folder exists for logs
Path("runs").mkdir(exist_ok=True)

# Initialize a global logger for this module
try:
    OBS_LOG
except NameError:
    OBS_LOG = open_jsonl("runs/obs.jsonl")

THREAD_ID = "run-001"


class Transition:
    def __init__(self, name, globalPre, pre=set(), post=dict(), thresholds=dict()):
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
        self.getPostConditions().update({place: postAction})

    def updateThreshold(self, topic, threshold):
        self.getThresholds().update({topic: threshold})

    def checkPrePlace(self, place):
        return (
            (place.getContent() is not None)
            and self.getThresholds().get(place.getTopic().getName())
            <= place.getTopic().getRelevance()
        )

    def checkAllRequiredPlaces(self):
        flag = True
        places = dict()

        for place in self.getPreConditions():
            flag = flag and self.checkPrePlace(place)
            if flag:
                places.update({place: place.getContent()})
            else:
                return dict()

        return places

    def checkGlobalTokens(self, places):
        return self.getGlobalPre()(places)

    def isFireable(self):
        # Note: returns a dict of places if all checks pass, else {}
        # Used truthily in `fire()` below.
        return self.checkAllRequiredPlaces()

        # If you want a strict boolean instead, use:
        # return bool(self.checkAllRequiredPlaces())

    def prepareRepresentationPre(self, preDict):
        representationPre = [
            "Topic: "
            + pl.getTopic().getName()
            + " with value: "
            + str(preDict[pl].getValue())
            + " from Place: "
            + pl.getName()
            + "\n"
            for pl in preDict
        ]
        return representationPre

    def prepareRepresentationPost(self, postDict):
        representationPost = [
            "Topic: "
            + pl.getTopic().getName()
            + " with value: "
            + str(pl.getContent().getValue())
            + " in Place: "
            + pl.getName()
            + "\n"
            for pl in postDict
        ]
        return representationPost

    def printRepresentation(self, header, representation):
        print(header)
        for info in representation:
            print(info)

    def fire(self):
        # Single span for the whole transition firing
        sid = OBS_LOG.log(
            type="transition",
            thread_id=THREAD_ID,
            node=self.getName(),
            payload={"phase": "try"},
        )

        if self.isFireable():
            places = {}
            for placePre in self.getPreConditions():
                places.update({placePre: placePre.getContent()})
                placePre.empty()

            # BEFORE: print + log
            pre_repr = self.prepareRepresentationPre(places)
            self.printRepresentation(
                f"Transition {self.getName()} firing with tokens: \n", pre_repr
            )
            OBS_LOG.log(
                type="transition",
                thread_id=THREAD_ID,
                node=self.getName(),
                span_id=sid,  # reuse same span
                payload={"phase": "before", "pre": pre_repr},
            )

            # produce into post-places
            postConds = self.getPostConditions()
            for placePost in postConds:
                placePost.insert(postConds.get(placePost).execute(places))

            # AFTER: print + log
            post_repr = self.prepareRepresentationPost(postConds)
            self.printRepresentation(
                f"Transition {self.getName()} has produced tokens: \n", post_repr
            )
            OBS_LOG.log(
                type="transition",
                thread_id=THREAD_ID,
                node=self.getName(),
                span_id=sid,  # reuse same span
                payload={"phase": "after", "post": post_repr},
            )

            return True
        else:
            # Not fireable
            OBS_LOG.log(
                type="error",
                thread_id=THREAD_ID,
                node=self.getName(),
                span_id=sid,  # reuse same span
                payload={"reason": "not_fireable"},
            )
            return False

