import json
from _datetime import datetime

from assets import Incident
from assets.graph import IncidentsGraph

def main():
    incident = Incident(
            id= "INC930000",
            created_at= datetime.fromisoformat("2025-09-01T08:00:00"),
            short_description= "Users report intermittent errors accessi",
            description= "Users report intermittent errors accessing the service",
            service= "user-service",
            impact= 2,
            state= "in progress"
    )

    g = IncidentsGraph(llm_call=True)
    out = g.run(incident=incident)
    print(out)

if __name__=="__main__":
    main()