from ..base import AgentModule


class BaseCritic(AgentModule):
    def __init__(self, identity, *args, **kwargs):
        self.identity = identity


class PromptedCritic(BaseCritic):
    def __init__(self, identity, llm, prompt_template, parser):
        super().__init__(identity)
        self.llm = llm
        self.prompt_template = prompt_template
        self.parser = parser

    def __call__(self, state, memory, llm_kwargs=None, **kwargs):
        if llm_kwargs is None:
            llm_kwargs = {}
        user_prompt = self.prompt_template.format(
            state=state, memory=memory, **kwargs)
        llm_outputs = self.llm(
            system_prompt=str(self.identity), user_prompt=user_prompt, parser=self.parser, **llm_kwargs
        )

        return llm_outputs