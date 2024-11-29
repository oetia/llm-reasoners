import logging
from browsergym.core.action.highlevel import HighLevelActionSet
from browsergym.core.env import BrowserEnv
from browsergym.experiments import EnvArgs

from utils import obs_preprocessor


# HIGH LEVEL ACTION SET DEF

def get_browser_action_set():
    return HighLevelActionSet(
        subsets=["chat", "tab", "nav", "bid", "infeas"],
        strict=False,  # less strict on the parsing of the actions
        multiaction=False,
        demo_mode="off",  # add visual effects
    )


# MANAGING BROWSERENV

# need to add in support for open-ended tasks
def get_env(task_name, action_set: HighLevelActionSet = get_browser_action_set(), seed: int = 16, headless: bool = False):
    env_args = EnvArgs(
        task_name=task_name,
        task_seed=seed,
        max_steps=100,
        headless=headless,
        record_video=True,
        # viewport={"width": 500, "height": 500},  # can be played with if needed
    )

    env = env_args.make_env(
        action_mapping=action_set.to_python_code,
        exp_dir="./results",
    )
    return env


def reset_env(env: BrowserEnv, seed: int = 16, logger: logging.Logger = None):
    obs, env_info = env.reset(seed=seed)
    # obs = obs_preprocessor(obs)
    # log_obs(logger, obs, "INITIAL STATE:")
    return obs, env_info


def step_env(env: BrowserEnv, action: str, logger: logging.Logger = None):
    obs, reward, terminated, truncated, step_info = env.step(action)
    # obs = obs_preprocessor(obs)
    # log_chosen_action(logger, action)
    # log_reward(logger, reward)
    # log_obs(logger, obs, "NEW STATE:")
    # _send_chat_info(env.unwrapped.chat, action, step_info)
    return obs, reward, terminated, truncated, step_info

# for backtracking in the real environment
# when you expand a node, you need to take the node's current action history, make sure that env is aligned with the current state, then you can expand.


def reset_and_replay_actions(env: BrowserEnv, action_history: list[str]) -> BrowserEnv:
    obs, env_info = env.reset(seed=16)
    for action in action_history:
        obs, reward, terminated, truncated, step_info = env.step(action)
        # _send_chat_info(env.unwrapped.chat, action, step_info)
    return env
