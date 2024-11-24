# import time

import pprint
from env import *
from utils import get_serializable_obs

from typing import NamedTuple

from agent import Agent
from llm import LLM

env = get_env("miniwob.login-user")
obs, env_info = reset_env(env)
# obs = get_serializable_obs(env.unwrapped, obs)
# goal_string = obs["goal_object"][0]["text"]
# print(goal_string)


llm = LLM(model="gpt-4o-mini",
          api_key="[key]")
agent = Agent(llm, use_world_model_planning=True, use_intent_only_memory=True)
# agent = Agent(llm, use_world_model_planning=False, use_intent_only_memory=True)
# action, info = agent.get_action(obs)
# print(action)

history = []
action = ''
step_count = 0
reward = 0

terminated = False
truncated = False

# multiple conditions of success
while not action.startswith('send_msg_to_user') and step_count < 30 and not terminated and not truncated:
    obs = get_serializable_obs(env.unwrapped, obs)
    action, step_info, planner_algorithm_output = agent.get_action(obs)
    history.append((obs, action, step_info, planner_algorithm_output))

    obs, reward, terminated, truncated, step_info = step_env(env, action)

    step_count += 1

# different ways in which the task can be completed
is_complete = (action.startswith('send_msg_to_user')
               and action not in ["send_msg_to_user('Error encountered when browsing.')",
                                  "send_msg_to_user('Too many errors encountered. Task failed.')"])

task_success = reward > 0


print(is_complete, task_success)
