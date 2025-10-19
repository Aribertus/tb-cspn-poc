from __future__ import annotations

import json
import random
import time
from datetime import datetime
from typing import Any, List, Dict
from tabulate import tabulate
from langchain_community.callbacks import OpenAICallbackHandler
from loguru import logger

from assets.utils import save_run_to_file
from assets.custom_obj import (
    AgentRole,
    AgentState,
    BaseLog,
    ConsultantLog,
    SupervisorLog,
    WorkerLog,
    Processed_Logs,
)
from assets.helper.config_helper import AppSettings


def worker_log_factory(
    node_name: str,
    processing_time: int,
    token_usage: int,
    total_cost: float,
    llm_count: int,
    directive_id: str,
    action: str,
    success: str,
    timestamp: datetime
) -> WorkerLog:
    return WorkerLog(
        node_name=node_name,
        processing_time=processing_time,
        token_usage=token_usage,
        total_cost=total_cost,
        llm_count=llm_count,
        directive_id=directive_id,
        action=action,
        success=success,
        timestamp=timestamp
    )

def add_log_to_state(
        agent_name: str,
        agent_role: str,
        state: AgentState|None,
        start_time: float,
        llm_count: bool,
        llm_callback: OpenAICallbackHandler|None,
        **role_specific_info:Any
) -> AgentState | WorkerLog | None:
    logger.debug(f"Adding {agent_name} log to state")
    log = BaseLog(
        node_name=agent_name,
        processing_time=round((time.perf_counter() - start_time) * 1000),
        token_usage=llm_callback.total_tokens if llm_callback else 0,
        total_cost=llm_callback.total_cost if llm_callback else 0,
        llm_count=llm_callback.successful_requests if llm_count else 0,
    )
    match agent_role:
        case AgentRole.consultant.value:
            consultant_log = ConsultantLog(
                **log.model_dump(),
                token_id=state.token.id,
                topic_extracted=state.token.topics.keys(),
                input_length=len(state.token.content)
            )
            state.nodes_logs[AgentRole.consultant.value].append(consultant_log)
            logger.debug("Consultant log added successfully")
            return state
        case AgentRole.supervisor.value:
            supervisor_log = SupervisorLog(
                **log.model_dump(),
                actions=[directive.action for directive in state.directives],
                reasons=[directive.metadata['directive_reason'] for directive in state.directives],
                token_id=state.directives[0].source_token_id,
                directive_generated=len(state.directives),
                timestamp=datetime.now()
            )
            state.nodes_logs[AgentRole.supervisor.value].append(supervisor_log)
            logger.debug("Supervisor log added successfully")
            return state
        case AgentRole.worker.value:
            data = log.model_dump()
            data.pop("processing_time", None)
            worker_log = WorkerLog(
                **data,
                processing_time=random.randint(50, 200),
                directive_id=role_specific_info['directive_id'],
                action=role_specific_info['action'],
                success="ok",
                timestamp=datetime.now()
            )
            #state.nodes_logs[AgentRole.worker.value].append(worker_log)
            logger.info("Worker log successfully created")
            return worker_log
        case _:
            pass

def log_processing(logs: List[Dict[str, Dict[str, List[BaseLog]]]]) -> Processed_Logs:
    logger.debug("Entering the log processing function")
    final_cost = 0
    total_llm_calls = 0
    total_time = 0
    total_success = 0
    total_items = 0
    # Per ogni incident
    for i, log in enumerate(logs):
        log_value = log[f'Inc{i}']
        logger.debug(log_value)
        for role, entries in log_value.items():
            for entry in entries:
                final_cost += entry.total_cost
                total_llm_calls += entry.llm_count
                total_time += entry.processing_time
                if role == "worker":
                    total_success += 1 if entry.success == "ok" else 0

        total_items += 1

    total_time_in_minutes = total_time / 60000
    total_success_rate = (total_success / total_items) * 100
    throughput_per_min = total_items / total_time_in_minutes
    processed_logs = Processed_Logs(
        final_cost=final_cost,
        total_llm_calls=total_llm_calls,
        total_time=total_time,
        total_items=total_items,
        total_success_rate=total_success_rate,
        throughput_per_min=throughput_per_min
    )
    return processed_logs

def print_summary(logs: Processed_Logs, settings: AppSettings) -> None:
    """
    Stampa un riepilogo dei log processati in diversi formati usando loguru.

    Args:
        settings: the settings parameter from yml file
        logs: Oggetto Processed_Logs con i dati aggregati.
        style: Tipo di output:
               - "simple" : output leggibile e allineato
               - "table"  : tabella stringhe formattate
               - "pretty" : tabella con tabulate (richiede libreria esterna)
    """

    if settings.style == "simple":
        output = (
            "=== Processing Summary ===\n"
            f"Final cost:                 {logs.final_cost:.10f}\n"
            f"Total LLM calls:            {logs.total_llm_calls}\n"
            f"Total execution time:       {logs.total_time} ms\n"
            f"Total processed items:      {logs.total_items}\n"
            f"Total success rate:         {logs.total_success_rate:.2f}%\n"
            f"Items processed per minute: {logs.throughput_per_min:.2f}\n"
            "===========================\n"
        )
    elif settings.style == "table":
        output = (
            "\n=== Processing Summary ===\n"
            f"{'Final cost:':25}{logs.final_cost:.10f}\n"
            f"{'Total LLM calls:':25}{logs.total_llm_calls}\n"
            f"{'Execution time (ms):':25}{logs.total_time}\n"
            f"{'Processed items:':25}{logs.total_items}\n"
            f"{'Success rate (%):':25}{logs.total_success_rate:.2f}\n"
            f"{'Throughput (items/min):':25}{logs.throughput_per_min:.2f}\n"
            "===========================\n"
        )
    elif settings.style == "pretty":
        summary = [
            ["Final cost", f"{logs.final_cost:.10f}"],
            ["Total LLM calls", logs.total_llm_calls],
            ["Execution time (ms)", logs.total_time],
            ["Processed items", logs.total_items],
            ["Success rate (%)", f"{logs.total_success_rate:.2f}"],
            ["Throughput (items/min)", f"{logs.throughput_per_min:.2f}"]
        ]
        output = tabulate(summary, headers=["Metric", "Value"], tablefmt="pretty")
    else:
        raise ValueError(f"Unknown style '{settings.style}'. Use 'simple', 'table', or 'pretty'.")

    report = compose_final_report(settings, output)

    path = save_run_to_file(report, settings.folder, settings.filename)
    log_str = "*"*30 + " INC ANALYSIS TERMINATED " + "*"*30
    logger.info(log_str)
    log_str = "\n" + report
    logger.info(log_str.center(60))
    logger.info(f"Run salvato in: {path.resolve()}")

def compose_final_report(settings: AppSettings, output: str) -> str:
    cfg_dict = settings.model_dump()
    ts_iso = datetime.now().isoformat(timespec="seconds")
    cfg_text = json.dumps(cfg_dict, indent=2, ensure_ascii=False)
    cfg_header = "## Settings (JSON)"
    report = (
        "================= INCIDENT ANALYSIS REPORT =================\n"
        f"Timestamp: {ts_iso}\n"
        "============================================================\n\n"
        f"{cfg_header}\n"
        f"{cfg_text}\n"
        "## Summary\n"
        f"{output if output.endswith('\n') else output + '\n'}"
    )
    return report

if __name__=="__main__":
    from langchain_community.callbacks import OpenAICallbackHandler

    print("OK: import riuscito")

