import gymnasium as gym
from osworld_env import OSWorldEnv  # Assuming OSWorldEnv is the OSWorld equivalent of DesktopEnv
from typing import Dict, NamedTuple, Optional, Callable, Any
from reasoners import Environment
import time

ActionGym = Any

class StateGym(NamedTuple):
    step_idx: int
    action_history: list[ActionGym]
    last_obs: dict
    current_obs: dict
    reward: float
    terminated: bool

class EnvironmentGym(Environment):
    def __init__(
        self,
        env: OSWorldEnv,
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

        return StateGym(
            step_idx=0,
            last_obs={},
            current_obs=obs,
            action_history=[],
            reward=0,
            terminated=False,
        )

    def step(self, state: StateGym, action: ActionGym) -> tuple[StateGym, dict]:
        if self.env_current_obs != state.current_obs:
            self.env.restore_checkpoint(state.action_history)  # Efficient backtracking

        start = time.time()
        obs, reward, terminated, step_info = self.env.step(action)
        if self.obs_preprocessor is not None:
            obs = self.obs_preprocessor(obs)
        self.env_current_obs = obs
        end = time.time()

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
        return state.terminated
