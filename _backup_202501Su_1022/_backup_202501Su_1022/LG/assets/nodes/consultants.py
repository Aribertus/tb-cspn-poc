import json
import time
import uuid
from datetime import datetime

from assets.helper.costants import INPUT_CONSULTANT_NAME, ROUTER_SUPERVISOR_NAME, ROOT_CAUSE_CONSULTANT_NAME, \
    TOOL_INVOCATION_SUPERVISOR_NAME, ENTITY_GRAPH_CONSULTANT_NAME
from assets.helper.logging import add_log_to_state
from assets.utils import create_chain, save_topics, merge_topic_scores
from assets.custom_obj import AgentState, Token, AgentRole
from assets.prompts import INPUT_CONSULTANT_PROMPT, ROOT_CAUSE_CONSULTANT_PROMPT, ENTITY_GRAPH_CONSULTANT_PROMPT
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from loguru import logger
from langchain_community.callbacks import get_openai_callback





def input_consultant_node(state: AgentState) -> Command:
    LLM = ChatOpenAI(model=state.model, temperature=state.temperature, max_retries=3, streaming=False)

    logger.warning("Entering the input consultant node")
    start_time = time.perf_counter()
    chain = create_chain(LLM, INPUT_CONSULTANT_PROMPT)
    input = {
        "incident_json": state.incident,
        "existing_topics": state.topics
    }
    with get_openai_callback() as cb:
        result = chain.invoke(input, config={"callbacks": [cb]})

    logger.info(f"Consultant node results: {result}")
    try:
        result_json = json.loads(result) if isinstance(result, str) else result
        if not isinstance(result_json, dict):
            result_json = {}
    except Exception as e:
        logger.error(f"Failed to parse RC consultant JSON: {e}")
        result_json = {}
    token = Token(
        id=str(uuid.uuid4()),
        layer="observation",
        topics=result_json,
        content=(state.incident.get("description") if isinstance(state.incident, dict) else ""),
        timestamp=datetime.now(),
        metadata={
            "agent": "input_consultant",
            "processing_time": time.time() - start_time
        }
    )
    logger.info(f"Token created with ID: {token.id}")
    save_topics(result_json)
    state.token = token
    state = add_log_to_state(
        agent_name=INPUT_CONSULTANT_NAME,
        agent_role=AgentRole.consultant.value,
        start_time=start_time,
        llm_count=True,
        llm_callback=cb,
        state=state
    )
    logger.info("-"*50)
    return Command(
        update={
            "nodes_logs": state.nodes_logs,
            "token": token,
            "topics": set(json.loads(result).keys())
        },
        goto=ROUTER_SUPERVISOR_NAME
    )

def root_cause_consultant_node(state: "AgentState") -> Command:
    LLM = ChatOpenAI(model=state.model, temperature=state.temperature, max_retries=3, streaming=False)
    logger.warning("Entering the root_cause_consultant node")
    start_time = time.perf_counter()

    chain = create_chain(LLM, ROOT_CAUSE_CONSULTANT_PROMPT)
    input = {
        "incident_json": state.incident,
        "existing_topics": state.topics
    }

    with get_openai_callback() as cb:
        result = chain.invoke(input, config={"callbacks": [cb]})

    logger.info(f"Root-cause consultant raw result: {result}")

    try:
        result_json = json.loads(result) if isinstance(result, str) else result
        if not isinstance(result_json, dict):
            result_json = {}
    except Exception as e:
        logger.error(f"Failed to parse RC consultant JSON: {e}")
        result_json = {}

    # Merge topic scores (mantieni gli score)
    merged_topics = merge_topic_scores(state.token.topics, result_json)

    # Token
    token = Token(
        id=str(uuid.uuid4()),
        layer="observation",
        topics=result_json,
        content=(state.incident.get("description") if isinstance(state.incident, dict) else ""),
        timestamp=datetime.now(),
        metadata={
            "agent": ROOT_CAUSE_CONSULTANT_NAME,
            "processing_time": time.perf_counter() - start_time
        }
    )
    logger.info(f"RC Token created with ID: {token.id}")

    save_topics(result_json)
    state.token = token

    # Log LLM usage
    state = add_log_to_state(
        agent_name="root_cause_consultant",
        agent_role=AgentRole.consultant.value,
        start_time=start_time,
        llm_count=True,
        llm_callback=cb,
        state=state
    )
    logger.info("-"*50)
    return Command(
        update={
            "nodes_logs": state.nodes_logs,
            "token": token,
            "topics": set(merged_topics.keys())
        },
        goto=TOOL_INVOCATION_SUPERVISOR_NAME
    )

def entity_graph_consultant_node(state: "AgentState") -> Command:
    LLM = ChatOpenAI(model=state.model, temperature=state.temperature, max_retries=3, streaming=False)
    logger.warning("Entering the entity_graph_consultant node")
    start_time = time.perf_counter()

    chain = create_chain(LLM, ENTITY_GRAPH_CONSULTANT_PROMPT)
    input = {
        "incident_json": state.incident,
        "existing_topics": state.topics
    }

    with get_openai_callback() as cb:
        result = chain.invoke(input, config={"callbacks": [cb]})

    logger.info(f"Entity-graph consultant raw result: {result}")

    try:
        result_json = json.loads(result) if isinstance(result, str) else result
        if not isinstance(result_json, dict):
            result_json = {}
    except Exception as e:
        logger.error(f"Failed to parse EG consultant JSON: {e}")
        result_json = {}

    merged_topics = merge_topic_scores(state.token.topics, result_json)

    token = Token(
        id=str(uuid.uuid4()),
        layer="analysis:entity_graph",
        topics=result_json,
        content=(state.incident.get("description") if isinstance(state.incident, dict) else ""),
        timestamp=datetime.now(),
        metadata={
            "agent": ENTITY_GRAPH_CONSULTANT_NAME,
            "processing_time": time.perf_counter() - start_time
        }
    )
    logger.info(f"EG Token created with ID: {token.id}")

    save_topics(result_json)
    state.token = token

    # Log LLM usage
    state = add_log_to_state(
        agent_name=ENTITY_GRAPH_CONSULTANT_NAME,
        agent_role=AgentRole.consultant.value,
        start_time=start_time,
        llm_count=True,
        llm_callback=cb,
        state=state
    )
    logger.info("-"*50)
    return Command(
        update={
            "nodes_logs": state.nodes_logs,
            "token": token,
            "topics": set(merged_topics.keys())
        },
        goto=TOOL_INVOCATION_SUPERVISOR_NAME
    )
