from ..base import AgentModule


class BaseWorldModel(AgentModule):
    def __init__(self, identity):
        self.identity = identity

    def __call__(self, state, memory, intent, **kwargs):
        raise NotImplementedError


class PromptedWorldModel(BaseWorldModel):
    def __init__(self, identity, llm, prompt_template, parser):
        super().__init__(identity)
        self.llm = llm
        self.prompt_template = prompt_template
        self.parser = parser

    def __call__(self, state, memory, intent, verbose=False, **kwargs):
        user_prompt = self.prompt_template.format(
            state=state, memory=memory, intent=intent, **kwargs
        )

        if verbose:
            print("===========================PromptedWorldModel===========================")
            print("SYSTEM PROMPT in PromptedWorldModel:")
            print(str(self.identity))
            print("-" * 100)
            print(f"USER PROMPT in PromptedWorldModel:")
            print(user_prompt)
        llm_outputs = self.llm(
            system_prompt=str(self.identity), user_prompt=user_prompt, parser=self.parser, **kwargs
        )

        return llm_outputs[0]
