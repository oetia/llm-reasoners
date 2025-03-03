import gymnasium as gym
from desktop_env import DesktopEnv
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
    truncated: bool  # Added missing truncated field

class EnvironmentGym(Environment):
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
        """
        Initializes the environment and sets up the first state.
        Ensures observations are correctly preprocessed.
        """
        obs = self.env.reset(task_config=self.example, seed=self.env_seed)
        if self.obs_preprocessor:
            obs = self.obs_preprocessor(obs)
        self.env_current_obs = obs

        return StateGym(
            step_idx=0,
            last_obs={},
            current_obs=obs,
            action_history=[],
            reward=0,
            terminated=False,
            truncated=False,
        )

    def step(self, state: StateGym, action: ActionGym) -> tuple[StateGym, dict]:
        """
        Steps the environment forward, ensuring it is synchronized with the given state.
        Uses optimized backtracking where possible to restore previous states efficiently.
        """

        # Check if environment state needs realignment
        if self.env_current_obs != state.current_obs:
            print("⚠️  Desynchronization detected. Attempting to realign environment state...")
            
            # Efficient Backtracking - Use OSWorld's state restoration if available
            if hasattr(self.env, "restore_state"):
                self.env.restore_state(state.current_obs)
            else:
                # Fallback: Reset and replay action history
                self.env.reset(task_config=self.example, seed=self.env_seed)
                for prev_action in state.action_history:
                    self.env.step(prev_action)

        start = time.time()
        obs, reward, terminated, truncated, step_info = self.env.step(action)

        if self.obs_preprocessor:
            obs = self.obs_preprocessor(obs)

        self.env_current_obs = obs  # Sync environment observation

        end = time.time()
        print(f"⏳ Step Execution Time: {end - start:.4f} seconds")

        # Log performance metrics
        if self.task_dir:
            with open(f"{self.task_dir}/time.txt", "a+") as f:
                f.write(f"Step Execution Time: {end - start:.4f} seconds\n")

        next_state = StateGym(
            step_idx=state.step_idx + 1,
            last_obs=state.current_obs,
            current_obs=obs,
            action_history=state.action_history + [action],
            reward=reward,
            terminated=terminated,
            truncated=truncated,
        )

        return next_state, {"env_reward": reward}

    def is_terminal(self, state: StateGym) -> bool:
        """
        Determines if the environment has reached a terminal state.
        """
        return state.terminated
