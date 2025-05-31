# tb_cspn_poc/tb_cspn_snakes.py
from snakes.nets import PetriNet, Place, Transition, Variable, Expression, Substitution

# Wrapper to make token hashable
class TokenWrapper:
    def __init__(self, data):
        self.data = data

    def __hash__(self):
        return hash(str(self.data))

    def __eq__(self, other):
        return isinstance(other, TokenWrapper) and self.data == other.data

    def __repr__(self):
        return repr(self.data)

# Create the Petri Net
net = PetriNet('TB-CSPN')

# Define places
net.add_place(Place('incoming_news'))
net.add_place(Place('consulted'))
net.add_place(Place('evaluated'))
net.add_place(Place('action_taken'))

# Define transitions
net.add_transition(Transition('consultant_transition'))
net.add_transition(Transition('supervisor_transition'))
net.add_transition(Transition('worker_transition'))

# Define arcs
net.add_input('incoming_news', 'consultant_transition', Variable('x'))
net.add_output('consulted', 'consultant_transition', Expression('x'))

net.add_input('consulted', 'supervisor_transition', Variable('x'))
net.add_output('evaluated', 'supervisor_transition', Expression('x'))

net.add_input('evaluated', 'worker_transition', Variable('x'))
net.add_output('action_taken', 'worker_transition', Expression('x'))

# Fire helper
def fire_if_enabled(transition_name, token):
    transition = net.transition(transition_name)
    sub = Substitution(x=token)
    if transition.enabled(sub):
        transition.fire(sub)
        print(f"Transition '{transition_name}' fired.")
    else:
        print(f"Transition '{transition_name}' not enabled.")

# Simulate processing inputs
inputs = [
    {"text": "Tech sector sees surge amid AI breakthroughs", "AI_stocks": 0.9},
    {"text": "Uncertainty rises after unexpected Fed decision", "AI_stocks": 0.3, "market_volatility": 0.9},
    {"text": "Retail stocks underperform despite holiday sales", "retail": 0.8, "AI_stocks": 0.2}
]

for raw in inputs:
    print(f"\n[Input] Processing: {raw['text']}")
    wrapped = TokenWrapper(raw)
    net.place('incoming_news').add(wrapped)

    # Consultant
    fire_if_enabled('consultant_transition', wrapped)

    # Supervisor
    consulted_tokens = list(net.place('consulted').tokens)
    for token in consulted_tokens:
        if token.data.get("AI_stocks", 0) > 0.8:
            fire_if_enabled('supervisor_transition', token)
        else:
            print("Supervisor threshold not met.")

    # Worker
    evaluated_tokens = list(net.place('evaluated').tokens)
    for token in evaluated_tokens:
        fire_if_enabled('worker_transition', token)

# Final output
print("\nFinal tokens in action_taken:", net.place('action_taken').tokens)
