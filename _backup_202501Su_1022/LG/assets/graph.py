import uuid
from langgraph.graph.state import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from assets.custom_obj import AgentState, AgentRole, Incident
from assets.nodes.consultants import (
    input_consultant_node,
    root_cause_consultant_node,
    entity_graph_consultant_node,
)
from assets.nodes.supervisors import (
    router_supervisor_node,
    tool_invocation_supervisor_node,
)
from assets.helper.costants import (
    INPUT_CONSULTANT_NAME,
    ROUTER_SUPERVISOR_NAME,
    ROOT_CAUSE_CONSULTANT_NAME,
    ENTITY_GRAPH_CONSULTANT_NAME,
    TOOL_INVOCATION_SUPERVISOR_NAME,
)

class IncidentsGraph:
    def __init__(self, llm_call: bool, topics: set[str] | None = None):
        self.builder = StateGraph(AgentState)
        memory = MemorySaver()

        # Nodi
        self.builder.add_node(INPUT_CONSULTANT_NAME, input_consultant_node)
        self.builder.add_node(ROUTER_SUPERVISOR_NAME, router_supervisor_node)
        self.builder.add_node(ROOT_CAUSE_CONSULTANT_NAME, root_cause_consultant_node)
        self.builder.add_node(ENTITY_GRAPH_CONSULTANT_NAME, entity_graph_consultant_node)
        self.builder.add_node(TOOL_INVOCATION_SUPERVISOR_NAME, tool_invocation_supervisor_node)
        self.builder.set_entry_point(INPUT_CONSULTANT_NAME)
        self.graph = (
            self.builder
            .compile(checkpointer=memory)
            .with_config({
                "configurable": {"thread_id": str(uuid.uuid4())}
            })
        )
        self.state = AgentState(
            topics=topics or set(),
            llm_supervisor=llm_call,
            incident=None,
            token=None,
            directives=[],
            nodes_logs={
                AgentRole.consultant.value: [],
                AgentRole.supervisor.value: [],
                AgentRole.worker.value: []
            }
        )

    def run(self, incident: Incident) -> AgentState:
        state_dict = self.state.model_dump()

        invoke_input = {
        **state_dict,
        "incident": incident
        }

        new_state_dict = self.graph.invoke(invoke_input)

        self.state = AgentState(**new_state_dict)

        return self.state

if __name__ == "__main__":
    pass
