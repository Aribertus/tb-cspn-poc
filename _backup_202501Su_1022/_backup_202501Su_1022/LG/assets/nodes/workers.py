import random
from typing import Dict, Any

from langchain.tools import tool
from loguru import logger
from datetime import datetime

from assets.helper.costants import DIAGNOSTIC_WORKER_NAME, NOTIFY_TEAM_WORKER_NAME, LOG_WORK_NOTE_WORKER_NAME
from assets.helper.logging import worker_log_factory


@tool("restart_worker")
def restart_worker_tool(directive: str, directive_id: str) -> Dict[str, Any]:
    """
    Restart a failing component/service according to the directive.

    Args:
        directive: The plain-text instruction for the worker (e.g., "service=orders restart-now").
        directive_id: A unique identifier to correlate logs and actions across nodes.
    Returns:
        The json representation of the worker log.
    """
    logger.info("-" * 50)
    logger.warning("Entering the restart_worker_tool")
    worker_log = worker_log_factory(
        node_name=DIAGNOSTIC_WORKER_NAME,
        processing_time=random.randint(50, 200),
        token_usage=0,
        total_cost=0,
        llm_count=0,
        directive_id=directive_id,
        action=directive,
        success="ok",
        timestamp=datetime.now()
    )
    logger.info(f"[restart_worker] id={directive_id} directive={directive}")
    logger.info("-"*50)
    return worker_log.model_dump(mode="json")


@tool("diagnostics_worker")
def diagnostics_worker_tool(directive: str, directive_id: str) -> Dict[str, Any]:
    """
    Run diagnostics as requested by the directive.

    Args:
        directive: The diagnostic instruction (e.g., "collect=cpu,mem duration=120s").
        directive_id: A unique identifier to correlate logs and actions across nodes.
    Returns:
        The json representation of the worker log
    """
    logger.info("-" * 50)
    logger.warning("Entering the diagnostics_worker_tool")
    worker_log = worker_log_factory(
        node_name=DIAGNOSTIC_WORKER_NAME,
        processing_time=random.randint(50, 200),
        token_usage=0,
        total_cost=0,
        llm_count=0,
        directive_id=directive_id,
        action=directive,
        success="ok",
        timestamp=datetime.now()
    )

    logger.info(f"[diagnostics_worker] id={directive_id} directive={directive}")
    logger.info("-"*50)
    return worker_log.model_dump(mode="json")


@tool("notify_team_worker")
def notify_team_worker_tool(directive: str, directive_id: str) -> Dict[str, Any]:
    """
    Notify the on-call team according to the directive.

    Args:
        directive: The message/routing info (e.g., "channel=sev1 text='DB down'").
        directive_id: A unique identifier to correlate logs and actions across nodes.
    Returns:
        The json representation of the worker log
    """
    logger.info("-" * 50)
    logger.warning("Entering the notify_team_worker_tool")
    worker_log = worker_log_factory(
        node_name=NOTIFY_TEAM_WORKER_NAME,
        processing_time=random.randint(50, 200),
        token_usage=0,
        total_cost=0,
        llm_count=0,
        directive_id=directive_id,
        action=directive,
        success="ok",
        timestamp=datetime.now()
    )
    logger.info(f"[notify_team_worker] id={directive_id} directive={directive}")
    logger.info("-"*50)
    return worker_log.model_dump(mode="json")


@tool("log_work_note_worker")
def log_work_note_worker_tool(directive: str, directive_id: str) -> Dict[str, Any]:
    """
    Append a work note in the incident record.

    Args:
        directive: The work note content or template reference.
        directive_id: A unique identifier to correlate logs and actions across nodes.

    Returns:
        The json representation of the worker log
    """
    logger.info("-" * 50)
    logger.warning("Entering the log_work_note_worker_tool")
    worker_log = worker_log_factory(
        node_name=LOG_WORK_NOTE_WORKER_NAME,
        processing_time=random.randint(50, 200),
        token_usage=0,
        total_cost=0,
        llm_count=0,
        directive_id=directive_id,
        action=directive,
        success="ok",
        timestamp=datetime.now()
    )
    logger.info(f"[log_work_note_worker] id={directive_id} directive={directive}")
    logger.info("-" * 50)
    return worker_log.model_dump(mode="json")
