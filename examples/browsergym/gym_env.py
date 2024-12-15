
from abc import abstractmethod, ABC
import gymnasium as gym
from typing import NamedTuple, Callable, Any

from support import get_env, build_propose_prompt, build_evaluation_prompt, obs_preprocessor, check_validity_of_action_proposal
from browsergym.core.action.highlevel import HighLevelActionSet
from browsergym.experiments import EnvArgs
from browsergym.core.action.parsers import highlevel_action_parser

from reasoners import SearchConfig, LanguageModel
from reasoners import SearchConfig, WorldModel, LanguageModel, Reasoner
from reasoners.algorithm import MCTS, BeamSearch
from reasoners.lm import OpenAIModel
from reasoners.visualization import visualize
from reasoners.visualization.tree_snapshot import NodeData, EdgeData
from reasoners.algorithm.mcts import MCTSNode

import os
import re
import json
import pickle

Action = Any


class StateGym(NamedTuple):
    step_idx: int
    last_obs: dict  # instead of strings these will be obs objects
    current_obs: dict
    # still need action history to be able to reconstruct the state for backtracking in mcts
    action_history: list[Action]
    reward: float
    terminated: bool
    truncated: bool


# class Dynamics(ABC):
#     @abstractmethod
#     def step(self, action: Action) -> tuple[StateGym, dict]:
#         raise NotImplementedError


class Environment(WorldModel):
    def __init__(self, env: gym.Env, env_seed: int = 42, max_steps=20, obs_preprocessor: Callable = None):
        self.env = env
        self.env_seed = env_seed
        self.obs_preprocessor = obs_preprocessor
        self.max_steps = max_steps
        self.env_current_obs = None

    def init_state(self):
        obs, env_info = self.env.reset(
            seed=self.env_seed)
        if self.obs_preprocessor is not None:
            obs = self.obs_preprocessor(obs)
        self.env_current_obs = obs

        print(obs["goal_object"])

        return StateGym(step_idx=0, last_obs={}, current_obs=obs, action_history=[], reward=0, terminated=False, truncated=False)

    def step(self, state: StateGym, action: Action):
        if self.env_current_obs != state.current_obs:
            self.env.reset(seed=self.env_seed)
            for action in state.action_history:
                self.env.step(action)
        try:
            obs, reward, terminated, truncated, step_info = self.env.step(
                action)
            if self.obs_preprocessor is not None:
                obs = self.obs_preprocessor(obs)
            self.env_current_obs = obs

            next_state = StateGym(step_idx=state.step_idx + 1,
                                  last_obs=state.current_obs, current_obs=obs,
                                  action_history=state.action_history +
                                  [action],
                                  reward=reward, terminated=terminated, truncated=truncated)

            return next_state, {"env_reward": reward}
        except NameError as e:  # invalid action passed in
            return state, {"env_reward": 0}

    def is_terminal(self, state: StateGym) -> bool:
        return state.terminated or state.truncated or state.step_idx >= self.max_steps


class SearchConfigBrowsergym(SearchConfig):
    def __init__(self,
                 action_set: HighLevelActionSet,
                 llm: LanguageModel,
                 n_proposals: int = 5, proposal_temperature: float = 0.7,
                 evaluation_temperature: float = 0.25,
                 use_axtree: bool = True, use_html: bool = False, use_screenshot: bool = False) -> None:
        super().__init__()
        self.action_set = action_set
        self.llm = llm
        self.n_proposals = n_proposals
        self.proposal_temperature = proposal_temperature
        self.evlaution_temperature = evaluation_temperature
        self.use_axtree = use_axtree
        self.use_html = use_html
        self.use_screenshot = use_screenshot

    def get_actions(self, state: StateGym) -> list[Action]:

        system_msgs, user_msgs, full_prompt_text = build_propose_prompt(
            state.current_obs,
            self.action_set, state.action_history,
            self.use_axtree, self.use_html, self.use_screenshot
        )

        response = self.llm.generate(
            full_prompt_text, num_return_sequences=self.n_proposals, temperature=self.proposal_temperature)
        action_proposals = response.text

        clustered_actions = []
        action_codes = set()
        for action_proposal in action_proposals:

            if check_validity_of_action_proposal(action_proposal):
                action_code = self.action_set.to_python_code(action_proposal)
                if action_code not in action_codes:
                    action_codes.add(action_code)
                    clustered_actions.append(action_proposal)

        return clustered_actions

    def fast_reward(self, state: StateGym, action: Action) -> tuple[float, dict]:

        system_msgs, user_msgs, full_prompt_txt = build_evaluation_prompt(
            state.current_obs, action, self.action_set, state.action_history,
            self.use_axtree, self.use_html, self.use_screenshot
        )

        response = self.llm.generate(
            full_prompt_txt, num_return_sequences=self.n_proposals, temperature=self.proposal_temperature)

        evaluation = response.text[0]

        json_string = re.search(r"\{.*\}", evaluation, re.DOTALL).group()
        json_object = json.loads(json_string)
        evaluation = json_object["score"] / 10

        return evaluation, {"self_eval": evaluation}

    def reward(self, state: StateGym, action: Action, **kwargs) -> tuple[float, dict]:
        return kwargs["self_eval"] + 100 * kwargs["env_reward"], kwargs


def run_task(task_name: str):
    browser_action_set = HighLevelActionSet(
        # subsets=["chat", "tab", "nav", "bid", "infeas"],
        subsets=["webarena"],
        strict=False,
        multiaction=True,
        demo_mode="off",  # 'default' is on
    )

    env_args = EnvArgs(
        task_name=task_name,
        task_seed=42,
        max_steps=100,
        headless=True,
        record_video=True,
    )

    # check to see if directory exists
    if not os.path.exists(f"./results/tree-search/{task_name}"):
        os.makedirs(f"./results/tree-search/{task_name}")

    env = env_args.make_env(
        action_mapping=browser_action_set.to_python_code,
        exp_dir=f"./results/tree-search/{task_name}",
    )

    llm = OpenAIModel(model="gpt-4o-mini")

    world_model = Environment(env=env, obs_preprocessor=obs_preprocessor)
    search_config = SearchConfigBrowsergym(
        action_set=browser_action_set, n_proposals=10, llm=llm, use_axtree=True, use_html=False, use_screenshot=False)
    algorithm = MCTS(n_iters=10,
                     depth_limit=10,
                     w_exp=10**.5,
                     #  w_exp=2**.5,
                     uct_with_fast_reward=True,
                     disable_tqdm=False,
                     output_trace_in_each_iter=True)

    reasoner = Reasoner(world_model, search_config, algorithm)

    result_rap = reasoner("")

    with open(f"./results/tree-search/{task_name}/result.pkl", "wb") as f:
        pickle.dump(result_rap, f)

    def browsergym_node_data_factory(n: MCTSNode) -> NodeData:
        return NodeData({"axtree": n.state.current_obs["axtree_txt"]} if n.state is not None else {"expanded": "not expanded"})

    def browsergym_edge_data_factory(n: MCTSNode) -> EdgeData:
        function_calls = highlevel_action_parser.search_string(
            n.action
        )
        function_calls = sum(function_calls.as_list(), [])

        python_code = ""
        for function_name, function_args in function_calls:
            python_code += (
                function_name +
                "(" + ", ".join([repr(arg) for arg in function_args]) + ")\n"
            )

        return EdgeData({"Q": n.Q,
                        "self_eval": n.fast_reward_details["self_eval"],
                         "action": python_code})

    # visualize(result_rap,
    #           node_data_factory=browsergym_node_data_factory,
    #           edge_data_factory=browsergym_edge_data_factory)

    with open(f"./results/tree-search/{task_name}/success.txt", "w") as f:
        if result_rap.terminal_state.reward == 1.0:
            f.write("TASK COMPLETED SUCCESSFULLY")
        else:
            f.write("TASK FAILED")

    env.close()


if __name__ == "__main__":

    tasks = [
        # "webarena.27", # failure
        # "webarena.28", # failure
        # "webarena.29",  # failure
        # "webarena.30",  # failure
        # "webarena.31",  # failure

        # change bio to ...
        # "webarena.399", # success
        # "webarena.400", # success
        # "webarena.401",  # success
        # "webarena.402",  #  success
        # "webarena.403", # success

        # "webarena.405", # failure
        # "webarena.406",  # failure
        # "webarena.410",  # failure

        # ultimately same tasks
        # "webarena.596",  # failure
        # "webarena.597",  # failure
        # "webarena.599",  # success

        # "webarena.619",  # failure

        # "webarena.642",  # failure
        # "webarena.66",  # failure
        # "webarena.67",  # failure
        # "webarena.68",  # failuk
        # "webarena.69",  # success
        # "webarena.718",  # failure
        # "webarena.731"  # success
    ]

    for task in tasks:
        print(task)
        try:
            run_task(task)
        except Exception as e:
            print(e)
            pass

    # run_task("miniwob.login-user")

    # run_task("miniwob.buy-ticket")
    # run_task("miniwob.form-sequence")
    # run_task("webarena.596")
    # run_task("webarena.27")
