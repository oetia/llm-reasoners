import datetime
import json
import logging
import os
import gymnasium as gym
from OSWorld.desktop_env.desktop_env import DesktopEnv
from typing import Dict, NamedTuple, Optional, Callable, Any

from reasoners import Environment

"""
world_model equivalent under llm-reasoners

currently this is basically just a wrapper around a gym environment 
implemented by browsergym. i’m not too familiar with osworld, but you may 
have to deviate a lot from how the step function is implemented here.

backtracking. not sure what osworld has available, but when making a step, 
you need to make sure that the env is aligned with the state object that 
mcts is focused on. currently for browsergym there’s a rather naive 
implementation of just calling env.reset(), then replaying the action_history 
stored in the currently focused state node. don’t know if this would be 
viable for osworld, but there’s probably a lot you can optimize here. 
"""
import time

ActionGym = Any

logger = logging.getLogger("desktopenv.gym_env")


class StateGym(NamedTuple):
    step_idx: int
    # action history used to reconstruct the env state for backtracking
    action_history: list[ActionGym]
    # gym observation objects
    last_obs: dict
    # outputs from env.step()
    current_obs: dict
    reward: float
    terminated: bool


class EnvironmentGym(Environment):
    """
    WorldModel, but for gym environments. Instead of being based off of a
    textual example, takes in a gym environment. An LLM will not be used
    for generating new states. The gym environment's step function takes care of that.

    Attributes
    -----------
    env : DesktopEnv
        OSWorld's environment object
    example : dict
        the benchmark example we are testing
    env_seed : int
        the seed for the gym environment (Default 42)
    obs_preprocessor : Optional[Callable[[dict], dict]]
        optional function to process the observation returned from
        resetting/stepping the environment before it is stored into the state tuple
    env_current_obs : dict
        the current observation of the environment which is used to check if
        a passed in state is aligned with the environment's current state
    task_dir : str
        directory where configs, logs stored
    """

    def __init__(
        self,
        env: DesktopEnv,
        example: Dict[str, Any],
        env_seed: int = 42,
        obs_preprocessor: Optional[Callable[[dict], dict]] = None,
        task_dir: str = None,
    ):
        self.env = env
        self.example = example
        self.env_seed = env_seed
        self.obs_preprocessor = obs_preprocessor
        self.env_current_obs: dict = None
        self.task_dir = task_dir

    def init_state(self) -> StateGym:
        obs = self.env.reset(task_config=self.example, seed=self.env_seed)
        if self.obs_preprocessor is not None:
            obs = self.obs_preprocessor(obs)
        self.env_current_obs = obs
        self.env.controller.start_recording()


        return StateGym(
            step_idx=0,
            last_obs={},
            current_obs=obs,
            action_history=[],
            reward=0,
            terminated=False,
        )

    def step(self, state: StateGym, action: ActionGym) -> tuple[StateGym, dict]:
        """
        Takes in a state and action and steps the environment. Should be noted
        that the environment may not be aligned with the state passed in. If
        the environment's current state (self.env_current_obs) is not the
        same as the state passed in, backtracking is needed. The basic
        implementation of this is rather naive, as it just resets the
        environment and replays the actions in the state's action_history list.
        Depending on the environment, there may be far more efficient ways to do so.

        Parameters
        ----------
        state : StateGym
            the state to step from
        action : ActionGym
            the action to take from the state

        Returns
        -------
        next_state : StateGym
            the next state after taking the action
        aux : dict
            used to pass the environment's reward to the search algorithm,
            which then passes it to the SearchConfig's reward function
        """

        if self.env_current_obs != state.current_obs:
            self.env.reset(task_config=self.example, seed=self.env_seed)
            for action in state.action_history:
                self.env.step(action)

        start = time.time()
        obs, reward, terminated, step_info = self.env.step(action)
        logger.info("Reward: %.2f", reward)
        logger.info("Done: %s", terminated)
        # Save screenshot and trajectory information
        action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

        with open(
            os.path.join(self.task_dir, f"step_{action_timestamp}.png"),
            "wb",
        ) as _f:
            _f.write(obs["screenshot"])
        with open(os.path.join(self.task_dir, "traj.jsonl"), "a") as f:
            f.write(
                json.dumps(
                    {
                        "step_num": 0,
                        "action_timestamp": action_timestamp,
                        "action": action,
                        "reward": reward,
                        "done": terminated,
                        "info": step_info,
                        "screenshot_file": f"step_{action_timestamp}.png",
                    }
                )
            )
            f.write("\n")

        if self.obs_preprocessor is not None:
            obs = self.obs_preprocessor(obs)
        self.env_current_obs = obs

        end = time.time()
        print(f"env step time: {end - start}")

        with open(f"{self.task_dir}/time.txt", "a+") as f:
            f.write(f"env step time: {end - start}\n")

        next_state = StateGym(
            step_idx=state.step_idx + 1,
            last_obs=state.current_obs,
            current_obs=obs,
            action_history=state.action_history + [action],
            reward=reward,
            terminated=terminated,
        )

        return next_state, {"env_reward": reward}

    def is_terminal(self, state: StateGym) -> bool:
        """
        Checks if environment reached terminal state
        """
        return state.terminated
