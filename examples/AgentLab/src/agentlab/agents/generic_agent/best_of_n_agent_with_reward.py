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
from agentlab.llm.llm_utils import Discussion, ParseError, SystemMessage, retry, json_parser
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
from .reward_model_prompt import RewardModelPrompt
from .best_of_n_agent import BestOfNAgentArgs, BestOfNAgent

from agentlab.llm.chat_api import ChatModel

@dataclass
class BestOfNWithRewardAgentArgs(GenericAgentArgs):

    def __init__(self, proposer_model_args, reward_model_args, flags, max_retry=4):
        self.chat_model_args = proposer_model_args
        self.reward_model_args = reward_model_args
        self.flags = flags
        self.max_retry = max_retry
        self.__post_init__()

    def __post_init__(self):
        try:  # some attributes might be temporarily args.CrossProd for hyperparameter generation
            self.agent_name = f"GenericAgent-{self.chat_model_args.model_name}".replace("/", "_")
        except AttributeError:
            pass

    def make_agent(self):
        return BestOfNWithRewardAgent(
            proposer_model_args=self.chat_model_args, reward_model_args=self.reward_model_args, flags=self.flags, max_retry=self.max_retry, 
        )
    
    
class BestOfNWithRewardAgent(GenericAgent):
    def __init__(
            self, 
            proposer_model_args: BaseModelArgs, 
            reward_model_args: BaseModelArgs,
            flags, 
            max_retry = 4
    ):
        super().__init__(proposer_model_args, flags, max_retry)
        self.reward_model_args = reward_model_args 
        self.reward_model_llm = reward_model_args.make_model()

    @cost_tracker_decorator
    def get_action(self, obs, step: int = None, exp_dir: str = None):
        self.obs_history.append(obs)

        # building prompt for proposer
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

        proposer_human_prompt = dp.fit_tokens(  # TODO: @zj (1) process as planner input (2) change to our prompt w/ planner results
            shrinkable=main_prompt,
            max_prompt_tokens=max_prompt_tokens,
            model_name=self.chat_model_args.model_name,
            max_iterations=max_trunc_itr,
            additional_prompts=system_prompt,
        )

        # building prompt for RewardModel
        reward_model_prompt = RewardModelPrompt(
            action_set=self.action_set,
            obs_history=self.obs_history,
            actions=self.actions,
            memories=self.memories,
            thoughts=self.thoughts,
            previous_plan=self.plan,
            step=self.plan_step,
            flags=self.flags,
        )

        # max_prompt_tokens, max_trunc_itr = self._get_maxes()

        reward_model_system_prompt = SystemMessage(dp.RewardModelSystemPrompt().prompt)


        reward_human_prompt = dp.fit_tokens(  # TODO: @zj (1) process as planner input (2) change to our prompt w/ planner results
            shrinkable=reward_model_prompt,
            max_prompt_tokens=max_prompt_tokens,
            model_name=self.reward_model_args.model_name,
            max_iterations=max_trunc_itr,
            additional_prompts=system_prompt,
        )
        try:
            # TODO, we would need to further shrink the prompt if the retry
            # cause it to be too long

            proposer_chat_messages = Discussion([system_prompt, proposer_human_prompt])
            reward_chat_messages = Discussion([reward_model_system_prompt, reward_human_prompt])
            ans_dict = retry(
                self.chat_llm,
                proposer_chat_messages,
                n_retry=self.max_retry,
                parser=best_of_n_with_reward_parser,
                log=True,
                reward_model_chat=self.reward_model_llm,
                reward_model_message=reward_chat_messages
            )
            # print('best of n agent get_action(): \n', ans_dict)
            ans_dict["busted_retry"] = 0
            # inferring the number of retries, TODO: make this less hacky
            ans_dict["n_retry"] = (len(proposer_chat_messages) - 3) / 2
        except ParseError as e:
            ans_dict = dict(
                action=None,
                n_retry=self.max_retry + 1,
                busted_retry=1,
            )

        stats = self.chat_llm.get_stats()
        # stats['reward_model_input_tokens'] = reward_model_stats['input_tokens']
        # stats['reward_model_output_tokens'] = reward_model_stats['output_tokens']
        # stats['reward_model_cost'] = reward_model_stats['cost']

        stats["n_retry"] = ans_dict["n_retry"]
        stats["busted_retry"] = ans_dict["busted_retry"]

        self.plan = ans_dict.get("plan", self.plan)
        self.plan_step = ans_dict.get("step", self.plan_step)
        self.actions.append(ans_dict["action"])
        self.memories.append(ans_dict.get("memory", None))
        self.thoughts.append(ans_dict.get("think", None))

        agent_info = AgentInfo(
            think=ans_dict.get("think", None),
            chat_messages=proposer_chat_messages,
            stats=stats,
            extra_info={"chat_model_args": asdict(self.chat_model_args)},
            reward_model_messages = reward_chat_messages
        )
        return ans_dict["action"], agent_info

    def rank_responses(message):
        # send message to chat, get response
        pass


def best_of_n_with_reward_parser(prompt, reward_model: ChatModel, actions: list):
    """
    Input: 
    res: ChatCompletion object
    reward_model: ChatModel object
    Output: messages dict, and ans_dict that has keys 'think' and 'action' and string values

    This function takes in the output of the chatmodel. It sorts the n completions by logprob, and constructs
    the ans_dict for the most probable answer.
    messages dict is in format of {'role': 'assistant', 'content': 'content of completion'}
    """

    # answers = [choice.message.content for choice in res.choices]
    print(prompt)
    answer = reward_model(prompt)['content']
    # print('THIS IS THE ANSWERRRRRRRRRRRR', answer)

    parsed, valid, message = json_parser(answer)
    print(parsed, valid, message)

    # this should look like '#1' or '#2', 1-based indexing
    best_proposal_number = sorted(parsed.items(), key= lambda kv: kv[1], reverse=True)[0][0]
    best_proposal_number = int(best_proposal_number.replace('#', '')) - 1
    print('best action: ', actions[best_proposal_number])

    return actions[best_proposal_number], parse_html_tags_raise(actions[best_proposal_number], ['think', 'action'])

