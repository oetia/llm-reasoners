from .base_reasoner import BaseReasoner
from .osworld_adapter import OSWorldAdapter

class OSWorldReasoner(BaseReasoner):
    def __init__(self):
        super().__init__()
        self.osworld_adapter = OSWorldAdapter()

    def reason(self, task_description):
        steps = self.break_down_task(task_description)
        results = []
        for step in steps:
            result = self.osworld_adapter.perform_action(step)
            results.append(result)
        return results
