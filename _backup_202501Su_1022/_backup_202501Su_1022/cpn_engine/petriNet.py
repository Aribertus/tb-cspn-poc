# -*- coding: utf-8 -*-
"""
Created on Wed May 28 18:23:18 2025

@author: botto
"""
from pathlib import Path
from tb_cspn_observe.logger import open_jsonl

from .transition import Transition
from .place import Place

# Ensure the runs/ folder exists for logs
Path("runs").mkdir(exist_ok=True)

# Initialize a global logger for this module
try:
    OBS_LOG
except NameError:
    OBS_LOG = open_jsonl("runs/obs.jsonl")

THREAD_ID = "run-001"


class PetriNet:
    def __init__(self):
        self._transitions: list[Transition] = []
        self._places: list[Place] = []

    def addTrans(self, trans: Transition):
        self._transitions.append(trans)

    def addPlace(self, place: Place):
        self._places.append(place)

    def getPlaces(self):
        return self._places

    def getTransitions(self):
        return self._transitions

    def fireNet(self):
        # Return True if at least one transition fired; otherwise False
        fired_any = False

        OBS_LOG.log(type="start", thread_id=THREAD_ID, node="petriNet.run")

        for trans in self.getTransitions():
            try_name = getattr(trans, "getName", lambda: str(trans))()
            OBS_LOG.log(
                type="transition",
                thread_id=THREAD_ID,
                node=try_name,
                payload={"phase": "try_from_runner"},
            )

            if trans.isFireable():
                trans.fire()
                fired_any = True

        OBS_LOG.log(type="end", thread_id=THREAD_ID, node="petriNet.run")
        return fired_any

    def process(self):
        flag = True
        while flag:
            flag = flag and self.fireNet()
        return flag

            