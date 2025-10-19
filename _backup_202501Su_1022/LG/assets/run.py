from datetime import date
from typing import Dict, List

from loguru import logger

from assets.custom_obj import BaseLog
from assets.graph import IncidentsGraph
from assets.utils import set_environment_variables, upload_json_incidents, upload_topics


def process_input(llm_call: bool = False, n_items: int = 50, temperature: float = 0.5, model: str = "gpt-4o-mini") -> List[Dict[str, Dict[str, List[BaseLog]]]]:
    logs: List[Dict[str, Dict[str, List[BaseLog]]]] = []
    set_environment_variables(f"incidents_analyzer_{date.today()}")
    incidents = upload_json_incidents()
    n_items = len(incidents) if n_items > len(incidents) else n_items
    for i, inc in enumerate(incidents[:n_items] or []):
        topics = set(upload_topics())
        agent_graph = IncidentsGraph(topics=topics, llm_call=llm_call)
        response = agent_graph.run(inc)
        logger.debug(response)
        logs.append({f"Inc{i}": response.nodes_logs})
        log_str = "*"*35 + f"INC {i} ANALYZED" + "*"*35
        logger.info(log_str)
        logger.info(" - "*30)
    return logs

