import json
import time
import uuid
from datetime import datetime

from openai import APIError

from assets.helper.costants import RESTART_WORKER_NAME, DIAGNOSTIC_WORKER_NAME, NOTIFY_TEAM_WORKER_NAME, \
    LOG_WORK_NOTE_WORKER_NAME, ROUTER_SUPERVISOR_NAME, TOOL_INVOCATION_SUPERVISOR_NAME
from assets.helper.logging import add_log_to_state
from assets.utils import create_agent, choose_worker_tool, parse_worker_log, create_chain
from assets.custom_obj import AgentState, AgentRole, Directive, WorkerLog

from assets.prompts import ROUTER_SUPERVISOR_PROMPT, TOOL_INVOCATION_SUPERVISOR_PROMPT
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from loguru import logger
from langchain_community.callbacks import get_openai_callback

from assets.utils import group_scores
from assets.nodes.workers import restart_worker_tool, diagnostics_worker_tool, notify_team_worker_tool, \
    log_work_note_worker_tool
from assets.prompts import TOOL_SUPERVISOR_PROMPT


TOOL_REGISTRY = {
    RESTART_WORKER_NAME: restart_worker_tool,
    DIAGNOSTIC_WORKER_NAME: diagnostics_worker_tool,
    NOTIFY_TEAM_WORKER_NAME: notify_team_worker_tool,
    LOG_WORK_NOTE_WORKER_NAME: log_work_note_worker_tool,
}

def router_supervisor_deterministic(topics):
    ROUTE_MIN = 0.50  # conf. minima per considerare “forte” un segnale
    MARGIN = 0.10
    logger.info(f"Using NO LLM in router supervisor node")
    rc_score, eg_score, rc_top, eg_top = group_scores(topics or {})
    # Decision policy
    llm_count = False
    cb = None
    if rc_score < ROUTE_MIN and eg_score < ROUTE_MIN:
        route = "entity_graph_consultant"  # raccolta contesto prima
        reason = f"weak signals (rc={rc_score:.2f}, eg={eg_score:.2f})"
    else:
        if rc_score > eg_score + MARGIN:
            route = "root_cause_consultant"
            reason = f"root-cause dominance: {rc_top}={rc_score:.2f}"
        elif eg_score > rc_score + MARGIN:
            route = "entity_graph_consultant"
            reason = f"entity-graph dominance: {eg_top}={eg_score:.2f}"
        else:
            # tie-break: preferisci costruire contesto
            route = "entity_graph_consultant"
            reason = f"tie (rc={rc_score:.2f}, eg={eg_score:.2f}) → prefer entity"
    return llm_count, cb, route, reason, rc_score, eg_score

def router_supervisor_node(state: AgentState) -> Command:
    #@TODO rivedere il sistema di soglie rispetto ai topic, introdurre elementi di dinamismo
    #@TODO meccanismo di validazione di nuovi topic -> esportare i dati su file per la gestione dinamica

    LLM = ChatOpenAI(model=state.model, temperature=state.temperature, max_retries=3, streaming=False)

    logger.warning("Entering the router supervisor node")
    start_time = time.perf_counter()
    topics = state.token.topics

    if state.llm_supervisor:
        logger.info(f"Using LLM in router supervisor node")
        topics_json = json.dumps(topics, ensure_ascii=False)
        chain = create_chain(LLM, ROUTER_SUPERVISOR_PROMPT)
        input = {
            "topics_json": topics_json
        }
        try:
            with get_openai_callback() as cb:
                result = chain.invoke(input, config={"callbacks": [cb]})
                logger.info(f"Router supervisor node results: {result}")
                try:
                    result_json = json.loads(result) if isinstance(result, str) else result
                    if not isinstance(result_json, dict):
                        result_json = {}
                except Exception as e:
                    logger.error(f"Failed to parse RC consultant JSON: {e}")
                    result_json = {}
                route = result_json.get("route", "entity_graph_consultant")
                reason = result_json.get("reason", "")
                rc_score = result_json.get("rc_score", 0)
                eg_score = result_json.get("eg_score", 0)
                llm_count = True
        except APIError as e:
            logger.error(f"LLM server error after retries: {e}. Falling back to heuristic.")
            llm_count, cb, route, reason, rc_score, eg_score = router_supervisor_deterministic(topics)
    else:
        llm_count, cb, route, reason, rc_score, eg_score = router_supervisor_deterministic(topics)

    directive_text = f"Routing to route: {route}"
    directive = Directive(
        id=str(uuid.uuid4()),
        action=directive_text,
        confidence=rc_score if route == "root_cause_consultant" else eg_score,
        source_token_id=state.token.id,
        timestamp=datetime.now(),
        metadata={
            "directive_text": directive_text,
            "directive_reason": reason,
            "processing_time": time.time() - start_time
        }
    )
    logger.info(f"Directive created with ID: {directive.id}")
    state.directives = [directive]

    state = add_log_to_state(
        agent_name=ROUTER_SUPERVISOR_NAME,
        agent_role=AgentRole.supervisor.value,
        start_time=start_time,
        llm_count=llm_count,
        llm_callback=cb,
        state=state
    )
    logger.info("-"*50)
    return Command(
        update={
            "nodes_logs": state.nodes_logs,
            "directives": state.directives
        },
        goto=route
    )
def tool_invocation_supervisor_node(state: AgentState) -> Command:

    LLM = ChatOpenAI(model=state.model, temperature=state.temperature, max_retries=3, streaming=False)

    logger.warning("Entering the tool_invocation_supervisor node")
    start_time = time.perf_counter()

    topics = state.token.topics
    incident = state.incident
    inc_dict = incident.model_dump(
        mode="python",
        exclude_none=True,
        by_alias=True,
    )
    with get_openai_callback() as cb:
        if state.llm_supervisor:
            logger.info(f"Using LLM in tool invocation supervisor node")
            available_tools = list(TOOL_REGISTRY.keys())
            chain = create_chain(LLM, TOOL_INVOCATION_SUPERVISOR_PROMPT)
            input = {
                "incident_json": inc_dict,
                "topics": topics,
                "available_tools": available_tools
            }
            try:
                result = chain.invoke(input, config={"callbacks": [cb]})
                logger.info(f"Tool invocation supervisor node results: {result}")
                try:
                    result_json = json.loads(result) if isinstance(result, str) else result
                    if not isinstance(result_json, dict):
                        result_json = {}
                except Exception as e:
                    logger.error(f"Failed to parse RC consultant JSON: {e}")
                    result_json = {}
                tool_name = result_json.get("tool_name", LOG_WORK_NOTE_WORKER_NAME)
                confidence = result_json.get("confidence", 0)
                reason = result_json.get("reason", "No reason available")
                directive_text = result_json.get("directive_text",
                                                 f"[Directive] Execute tool '{tool_name}' for incident {incident.id}. ")
            except APIError as e:
                logger.error(f"LLM server error after retries: {e}. Falling back to heuristic.")
                tool_name, confidence, reason = choose_worker_tool(topics, inc_dict)
                directive_text = f"[Directive] Execute tool '{tool_name}' for incident {incident.id}. "
        else:
            logger.info(f"Using NO-LLM in tool invocation supervisor node")
            tool_name, confidence, reason = choose_worker_tool(topics, inc_dict)
            directive_text = f"[Directive] Execute tool '{tool_name}' for incident {incident.id}. "

        directive = Directive(
            id=str(uuid.uuid4()),
            action=directive_text,
            confidence=confidence,
            source_token_id=getattr(state.token, "id", None),
            timestamp=datetime.now(),
            metadata={
                "selected_tool": tool_name,
                "directive_text": directive_text,
                "directive_reason": reason if reason else "Reason not detected",
                "processing_time": time.perf_counter() - start_time,
            },
        )
        logger.info(f"Directive created with ID: {directive.id} → tool={tool_name}")
        logger.info(f"Reason: {directive.metadata['directive_reason']}")

        tool_obj = TOOL_REGISTRY[tool_name]
        agent = create_agent(
            llm=LLM,
            tools=[tool_obj],
            system_prompt=TOOL_SUPERVISOR_PROMPT,
        )
        try:
            result = agent.invoke({
                # molti agent executor richiedono 'input'
                "input": directive_text,
                # variabili usate dal prompt
                "directive": directive_text,
                "directive_id": directive.id,
                "incident_json": inc_dict,
                "topics": topics,
                "tool_name": tool_name
            })
            logger.debug(f"tool_invocation_supervisor agent result: {result}")
            logger.debug("------------------------------------------")
            logger.debug(f"Creating worker log from supervisor node")
            worker_log = parse_worker_log(result.get("output"))
            if isinstance(worker_log, WorkerLog):
                state.nodes_logs.setdefault("worker", []).append(worker_log)
            else:
                logger.error(f"Worker log non valido, salvato come raw: {worker_log}")
            logger.debug("------------------------------------------")
        except APIError as e:
            logger.error(f"LLM server error after retries: {e}. No worker called.")

    state.directives = [directive]
    state = add_log_to_state(
        agent_name=TOOL_INVOCATION_SUPERVISOR_NAME,
        agent_role=AgentRole.supervisor.value,
        start_time=start_time,
        llm_count=True,
        llm_callback=cb,
        state=state,
    )

    update = {
        "nodes_logs": state.nodes_logs,
        "directives": state.directives,
        "last_executed_tool": tool_name,
        "last_tool_result": result.get("output", result),
    }
    logger.info("-"*50)
    return Command(update=update, goto="__end__")
