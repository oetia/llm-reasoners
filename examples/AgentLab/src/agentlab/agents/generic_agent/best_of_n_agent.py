import difflib
import logging
import time
from copy import copy
from pathlib import Path
from warnings import warn
from dataclasses import asdict, dataclass


import bgym
from browsergym.experiments.agent import AgentInfo
from browsergym.experiments.loop import ExpArgs, ExpResult, yield_all_exp_results
from bs4 import BeautifulSoup
from langchain.schema import AIMessage, BaseMessage
from langchain_community.adapters.openai import convert_message_to_dict
from agentlab.llm.chat_api import BaseModelArgs, make_system_message, make_user_message
from agentlab.llm.llm_utils import Discussion, ParseError, SystemMessage, retry
from agentlab.agents import dynamic_prompting as dp

from openai.types.chat.chat_completion import ChatCompletion


from agentlab.agents.agent_args import AgentArgs
from agentlab.agents.dynamic_prompting import ActionFlags
from agentlab.experiments.study import Study
from agentlab.llm.chat_api import make_assistant_message
from agentlab.llm.llm_utils import Discussion, messages_to_dict, parse_html_tags_raise
from agentlab.llm.tracking import cost_tracker_decorator


from .generic_agent import GenericAgent, GenericAgentArgs
from .generic_agent_prompt import GenericPromptFlags, MainPrompt

from agentlab.llm.chat_api import ChatModel

class BestOfNAgentArgs(GenericAgentArgs):

    def make_agent(self):
        return BestOfNAgent(
            chat_model_args=self.chat_model_args, flags=self.flags, max_retry=self.max_retry
        )
    

class BestOfNAgent(GenericAgent):

    def __init__(
        self,
        chat_model_args: BaseModelArgs,
        flags: GenericPromptFlags,
        max_retry: int = 4,
        # n_sample: int = 1
    ):
        super().__init__(chat_model_args, flags, max_retry)

        # self.chat_llm = chat_model_args.make_model()
        # self.chat_model_args = chat_model_args
        # self.max_retry = max_retry
        # self.n_sample = chat_model_args.n
        # print('aksdhjfkldsjfklajsdklf;jsdklfjl;sdjk;lsdajfk;lsdjfkljasdkl;fjsda;fj;asdjfkl;sdjf;lsdjfkl;jsd;lfjsdkl;jf;asd')
        # print(self.chat_llm.n)

        # self.flags = flags
        # self.action_set = self.flags.action.action_set.make_action_set()
        # self._obs_preprocessor = dp.make_obs_preprocessor(flags.obs)

        # self._check_flag_constancy()
        # self.reset(seed=None)

    # def obs_preprocessor(self, obs: dict) -> dict:
    #     return self._obs_preprocessor(obs)



    @cost_tracker_decorator
    def get_action(self, obs, step: int = None, exp_dir: str = None):
        self.obs_history.append(obs)

        main_prompt = MainPrompt(
            action_set=self.action_set,
            obs_history=self.obs_history,
            actions=self.actions,
            memories=self.memories,
            thoughts=self.thoughts,
            previous_plan=self.plan,
            step=self.plan_step,
            flags=self.flags,
        )

        max_prompt_tokens, max_trunc_itr = self._get_maxes()

        system_prompt = SystemMessage(dp.SystemPrompt().prompt)

        human_prompt = dp.fit_tokens(  # TODO: @zj (1) process as planner input (2) change to our prompt w/ planner results
            shrinkable=main_prompt,
            max_prompt_tokens=max_prompt_tokens,
            model_name=self.chat_model_args.model_name,
            max_iterations=max_trunc_itr,
            additional_prompts=system_prompt,
        )
        try:
            # TODO, we would need to further shrink the prompt if the retry
            # cause it to be too long

            chat_messages = Discussion([system_prompt, human_prompt])
            # print(chat_messages)
            ans_dict = retry(
                self.chat_llm,
                chat_messages,     # <--------- figure out whats happening with chat_messages, because this gets modified within retry() and if this modification is necessary, thats kind of annoying
                n_retry=self.max_retry,
                parser=best_of_n_parser_log_prob,
            )
            print('best of n agent get_action(): \n', ans_dict)
            ans_dict["busted_retry"] = 0
            # inferring the number of retries, TODO: make this less hacky
            ans_dict["n_retry"] = (len(chat_messages) - 3) / 2
        except ParseError as e:
            ans_dict = dict(
                action=None,
                n_retry=self.max_retry + 1,
                busted_retry=1,
            )

        stats = self.chat_llm.get_stats()
        stats["n_retry"] = ans_dict["n_retry"]
        stats["busted_retry"] = ans_dict["busted_retry"]

        self.plan = ans_dict.get("plan", self.plan)
        self.plan_step = ans_dict.get("step", self.plan_step)
        self.actions.append(ans_dict["action"])
        self.memories.append(ans_dict.get("memory", None))
        self.thoughts.append(ans_dict.get("think", None))

        agent_info = AgentInfo(
            think=ans_dict.get("think", None),
            chat_messages=chat_messages,
            stats=stats,
            extra_info={"chat_model_args": asdict(self.chat_model_args)},
        )
        return ans_dict["action"], agent_info

    def reset(self, seed=None):
        self.seed = seed
        self.plan = "No plan yet"
        self.plan_step = -1
        self.memories = []
        self.thoughts = []
        self.actions = []
        self.obs_history = []

    def _check_flag_constancy(self):
        flags = self.flags
        if flags.obs.use_som:
            if not flags.obs.use_screenshot:
                warn(
                    """
Warning: use_som=True requires use_screenshot=True. Disabling use_som."""
                )
                flags.obs.use_som = False
        if flags.obs.use_screenshot:
            if not self.chat_model_args.vision_support:
                warn(
                    """
Warning: use_screenshot is set to True, but the chat model \
does not support vision. Disabling use_screenshot."""
                )
                flags.obs.use_screenshot = False
        return flags

    def _get_maxes(self):
        maxes = (
            self.flags.max_prompt_tokens,
            self.chat_model_args.max_total_tokens,
            self.chat_model_args.max_input_tokens,
        )
        maxes = [m for m in maxes if m is not None]
        max_prompt_tokens = min(maxes) if maxes else None
        max_trunc_itr = (
            self.flags.max_trunc_itr
            if self.flags.max_trunc_itr
            else 20  # dangerous to change the default value here?
        )
        return max_prompt_tokens, max_trunc_itr
    
def best_of_n_parser_log_prob(res: ChatCompletion):
    """
    Input: ChatCompletion object
    Output: messages dict, and ans_dict that has keys 'think' and 'action' and string values

    This function takes in the output of the chatmodel. It sorts the n completions by logprob, and constructs
    the ans_dict for the most probable answer.
    messages dict is in format of {'role': 'assistant', 'content': 'content of completion'}
    """
    # print(res.choices[0].message.content)
    # for choice in res.choices:
    #     print(choice.message.content)
    #     print(choice.logprobs.content)
    #     ranked_responses.append(choice.message.content, choice.logprobs.content)
    # creates list [(completion, avg logprob), (completion, avg logprob), ...]
    ranked_responses = [(choice.message.content, sum(token.logprob for token in choice.logprobs.content) / len(choice.logprobs.content)) for choice in res.choices]
    # print(ranked_responses)
    best = sorted(ranked_responses, key=lambda x: x[1], reverse=True)[0][0]
    message = {'role': 'assistant', 'content': best}
    return message, parse_html_tags_raise(best, ['think', 'action'])



