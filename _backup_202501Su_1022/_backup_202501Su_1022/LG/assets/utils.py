import csv
import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Union, Dict, List, Tuple

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.output_parsers import StrOutputParser

from decouple import config
from langchain_core.runnables import RunnableSerializable

from assets.custom_obj import AgentState, WorkerLog

from langgraph.types import Command
from loguru import logger

from assets.helper.costants import (
    RESTART_WORKER_NAME,
    NOTIFY_TEAM_WORKER_NAME,
    DIAGNOSTIC_WORKER_NAME,
    LOG_WORK_NOTE_WORKER_NAME,
)

AgentLike = Union[AgentExecutor, RunnableSerializable[dict, Any]]
def create_agent(llm: BaseChatModel, tools: list, system_prompt: str) -> AgentExecutor:
    """
    Crea un agente con tools
    :param llm:
    :param tools:
    :param system_prompt:
    :return: AgentExecutor
    """
    logger.debug(f"Entering create agent function")
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),  # richiesto dal tools agent
    ])
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)  # type: ignore
    return agent_executor

def create_chain(llm: BaseChatModel, system_prompt: str) -> RunnableSerializable[dict, Any]:
    logger.debug("Entering the create chain function")
    return ChatPromptTemplate.from_template(system_prompt) | llm | StrOutputParser()

def agent_node(state: AgentState, agent: AgentLike, name: str, field_to_update: str, go_to: str):
    logger.debug("Entering the agent node function")
    result = agent.invoke(state)
    if isinstance(agent, RunnableSerializable):
        return Command(
            update={
                field_to_update: result['field_to_update']
            },
            goto=result['go_to']
        )


def set_environment_variables(project_name: str = "") -> None:
    """
    Funzione di caricamento delle variabili d'ambiente da .env
    :param project_name: il nome del progetto per il logging su langsmith
    :return:
    """
    logger.debug("Setting env variable from .env file")
    if not project_name:
        project_name = f"Test_{date.today()}"

    os.environ["OPENAI_API_KEY"] = str(config("OPENAI_API_KEY"))

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = str(config("LANGCHAIN_API_KEY"))
    os.environ["LANGCHAIN_PROJECT"] = project_name


def upload_json_incidents() -> List[Dict]|None:
    """
    Funzione di caricamento di una lista di json
    :return: La lista di dizionari corrispondenti ai json del file
    """
    logger.debug("Entering the upload json incident function")
    PROJECT_ROOT = Path(__file__).resolve().parent
    path = PROJECT_ROOT.parent / "data" / "incidents.json"
    try:
        logger.debug(f"loading file: {path.name}")
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.error(f"Errore nel caricamento del file json:{path.name}")
        return None

def upload_topics(strip: bool = True, drop_empty: bool = True)-> List[str]|None:
    """
    funzione di caricamento di una lista di stinghe
    Il file deve avere stringhe separate da virgola
    :return: La lista di stringhe
    """
    logger.debug("Entering the upload topics function")
    PROJECT_ROOT = Path(__file__).resolve().parent
    path = PROJECT_ROOT.parent / "data" / "topics.txt"
    items: List[str] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter=",")
        for row in reader:            # row è una lista di celle su quella riga
            for cell in row:
                s = cell.strip() if strip else cell
                if drop_empty and s == "":
                    continue
                items.append(s)
    return items if items else []

def save_topics(topics: Dict[str,float])-> None:
    logger.debug("Entering the save topics function")
    PROJECT_ROOT = Path(__file__).resolve().parent
    path = PROJECT_ROOT.parent / "data" / "topics.txt"
    new_topics = set(topics.keys())
    saved_topics = set(upload_topics())
    all_topics = new_topics | saved_topics
    logger.info("Saving new topics to file")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(all_topics)
    pass

def group_scores(topics: Dict[str, float]) -> tuple[float, float, str, str]:
    logger.debug("Entering the group score function")
    ROOT_CAUSE_TOPICS = {
        "availability", "latency", "auth", "database", "network", "config", "capacity", "diagnostics"
    }
    ENTITY_GRAPH_TOPICS = {
        "dependency", "deployment", "incident_management","restart_candidate","notification_required"
    }
    tnorm = {str(k).lower().strip(): float(v) for k, v in (topics or {}).items()}
    rc = {k: v for k, v in tnorm.items() if k in ROOT_CAUSE_TOPICS}
    eg = {k: v for k, v in tnorm.items() if k in ENTITY_GRAPH_TOPICS}
    rc_score = max(rc.values()) if rc else 0.0
    eg_score = max(eg.values()) if eg else 0.0
    rc_top = max(rc, key=rc.get) if rc else ""
    eg_top = max(eg, key=eg.get) if eg else ""
    return rc_score, eg_score, rc_top, eg_top

def merge_topic_scores(old: Dict[str, float], new: Dict[str, float]) -> Dict[str, float]:
    """Unisce i topic mantenendo per ogni chiave lo score massimo (con clamp [0,1])."""
    logger.debug("Entering the merge topic scores function")
    out = dict(old or {})
    for k, v in (new or {}).items():
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if not (0.0 <= v <= 1.0):
            v = max(0.0, min(1.0, v))
        kk = str(k).lower().strip()
        out[kk] = max(v, out.get(kk, 0.0))
    return out

def choose_worker_tool(topics: Dict[str, float], incident: Dict) -> Tuple[str, float, str]:
    """
    Ritorna (tool_name, confidence, reason).
    """
    logger.debug("Entering che choose woker tool function")
    t = {str(k).lower().strip(): float(v) for k, v in (topics or {}).items()}
    impact = incident.get("impact")  # 1,2,3
    state  = (incident.get("state") or "").lower()

    # 1) Restart se forte candidato e il ticket non è chiuso
    if t.get("restart_candidate", 0.0) >= 0.70 and state not in {"resolved", "closed"}:
        return RESTART_WORKER_NAME, t["restart_candidate"], "restart_candidate strong and ticket open"

    # 2) Notifica se richiesta esplicita o impatto alto + availability molto alto
    if t.get("notification_required", 0.0) >= 0.70:
        return "notify_team_worker", t["notification_required"], "notification_required strong"
    if impact == 1 and t.get("availability", 0.0) >= 0.85:
        return NOTIFY_TEAM_WORKER_NAME, t["availability"], "high impact + availability signal"

    # 3) Diagnostica se richiesto/utile
    if t.get("diagnostics", 0.0) >= 0.60:
        return DIAGNOSTIC_WORKER_NAME, t["diagnostics"], "diagnostics indicated"

    # 4) Fallback: log work note
    conf = max(t.get("incident_management", 0.0), 0.50)
    return LOG_WORK_NOTE_WORKER_NAME, conf, "fallback to work note"

def parse_worker_log(raw_output: str | dict) -> WorkerLog | dict | None:
    """
    Prova a costruire un WorkerLog dal tool output.
    Se non riesce, ritorna un dict con 'raw_output' o None.
    """
    logger.debug("Entering the parse worker log function")
    if isinstance(raw_output, str):
        try:
            data = json.loads(raw_output)
        except json.JSONDecodeError:
            return {"raw_output": raw_output}
    elif isinstance(raw_output, dict):
        data = raw_output
    else:
        return {"raw_output": str(raw_output)}

    tool_output = data.get("tool_output", data)

    if isinstance(tool_output, dict):
        try:
            return WorkerLog.model_validate(tool_output)
        except Exception as e:
            logger.error(f"WorkerLog validation failed: {e}")
            return {"raw_output": tool_output}
    else:
        return {"raw_output": str(tool_output)}


def save_run_to_file(content: str, folder: str = "runs", filename: str | None = None) -> Path:
    """
    Salva il contenuto in una cartella, con nome file parametrico o basato su timestamp.

    Args:
        content: testo da salvare
        folder: cartella di destinazione (default "runs")
        filename: nome del file (se None, usa timestamp giorno-mese-anno-ora-min-sec)

    Returns:
        Path al file salvato
    """
    logger.debug("Entering the save_run_to_file function")
    Path(folder).mkdir(parents=True, exist_ok=True)

    if filename is None:
        timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
        filename = f"run{timestamp}.txt"

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    path = PROJECT_ROOT / folder / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path

if __name__ == "__main__":
    save_run_to_file("abc")
