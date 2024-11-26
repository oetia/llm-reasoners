from pprint import pprint
import json
from datetime import datetime
from functools import partial

from llm import LLM


from agent_model.modules import (
    LLMReasonerPlanner,
    PolicyPlanner,
    PromptedActor,
    PromptedCritic,
    PromptedEncoder,
    PromptedPolicy,
    PromptedWorldModel,
)
from agent_model.variables import (
    AgentInstructionEnvironmentIdentity,
    BrowserActionSpace,
    BrowserGymObservationSpace,
    StepKeyValueMemory,
)
from agent_prompts import (
    actor_prompt_template,
    critic_prompt_template,
    encoder_prompt_template,

    policy_prompt_template,
    world_model_prompt_template,
)
# from logger import AgentLogger
from utils import ParseError, parse_html_tags_raise


import logging

logger = logging.getLogger("testing")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def parser(text, keys, optional_keys=()):
    try:
        ans_dict = parse_html_tags_raise(text, keys, optional_keys)
    except ParseError as e:
        return None, False, str(e)
    return ans_dict, True, ''


class Agent():
    def __init__(self,
                 llm: LLM,
                 use_world_model_planning: bool,
                 use_intent_only_memory: bool):
        self.llm = llm
        self.action_space = BrowserActionSpace(
            action_subsets=['chat', 'bid'],
            use_nav=True,
            strict=False,
            multiaction=False,
        )
        self.observation_space = BrowserGymObservationSpace()

        agent_name = 'Web Browsing Agent'
        agent_description = 'An information and automation assistant who responds to \
user instructions by browsing the internet. The assistant strives to answer each question \
accurately, thoroughly, efficiently, and politely, and to be forthright when it is \
impossible to answer the question or carry out the instruction. The assistant will \
end the task once it sends a message to the user.'
        self.identity = AgentInstructionEnvironmentIdentity(
            agent_name=agent_name,
            agent_description=agent_description,
            observation_space=self.observation_space,
            action_space=self.action_space,
        )

        # the goal is currently going to be none. gets populated when self.observation_space gets updated in step() / get_action()
        # with open("./system_prompt.txt", "w") as f:
        #     f.write(str(self.identity))

        self.encoder = PromptedEncoder(
            self.identity, self.llm, prompt_template=encoder_prompt_template, parser=partial(
                parser, keys=['state'])
        )
        self.use_intent_only_memory = use_intent_only_memory
        if self.use_intent_only_memory:
            self.memory = StepKeyValueMemory(['intent'])
        else:
            self.memory = StepKeyValueMemory(['state', 'intent'])

        self.use_world_model_planning = use_world_model_planning
        if self.use_world_model_planning:

            # generate proposals
            self.policy = PromptedPolicy(
                self.identity, self.llm, prompt_template=policy_prompt_template, parser=partial(
                    parser, keys=['intent'], optional_keys=['think'])
            )

            # predict the next state that results from those proposals
            self.world_model = PromptedWorldModel(
                self.identity,
                self.llm,
                prompt_template=world_model_prompt_template,
                parser=partial(parser, keys=['next_state'])
            )

            # evaluate the new state
            self.critic = PromptedCritic(
                self.identity, self.llm, prompt_template=critic_prompt_template,
                parser=partial(
                    parser, keys=['status', 'on_the_right_track'], optional_keys=['think']
                )
            )

            # self.planner = PolicyPlanner(self.policy)
            self.planner = LLMReasonerPlanner(
                self.policy, self.world_model, self.critic)

        else:
            self.policy = PromptedPolicy(
                self.identity, self.llm, prompt_template=policy_prompt_template, parser=partial(
                    parser, keys=['intent'], optional_keys=['think'])
            )

            self.planner = PolicyPlanner(self.policy)

        self.actor = PromptedActor(
            self.identity, self.llm, prompt_template=actor_prompt_template, parser=partial(
                parser, keys=['action'])
        )

        self.reset()

    def reset(self):
        self.identity.reset()
        self.memory.reset()

    def get_action(self, raw_obs: dict):
        observation, info = self.observation_space.parse_observation(raw_obs)
        if info.get('return_action') is not None:
            step = {
                'observation': observation,
                'state': None,
                'intent': None,
                'action': info['return_action'],
            }
            return info['return_action'], step

        # doesn't come with goal upon initialization. has to get it from obs object.
        self.identity.update(user_instruction=observation['goal'])

        obs_txt = observation['clean_axtree_txt']
        obs_screenshot = observation['screenshot_som_base64']
        logger.info(f'*Observation*: {obs_txt}')

        # state = self.encoder(obs_txt, self.memory)["state"]
        state = self.encoder(obs_txt, obs_screenshot, self.memory)["state"]
        logger.info(f'*State*: {state}')

        planner_algorithm_output = None  # search algorihtm trace for visualization
        if self.use_world_model_planning:
            planner_result_dict = self.planner(state, self.memory)
            intent = planner_result_dict['intent']
            planner_algorithm_output = planner_result_dict['planner_algorithm_output']
        else:
            intent = self.planner(state, self.memory)['intent']
        logger.info(f'*Intent*: {intent}')

        action = self.actor(obs_txt, obs_screenshot, state,
                            self.memory, intent)['action']
        logger.info(f'*Action*: {action}')

        step = {
            'observation': observation,
            'state': state,
            'intent': intent,
            'action': action,
        }
        self.memory.update(**step)
        self.memory.step()

        return action, step, planner_algorithm_output
