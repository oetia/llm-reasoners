import argparse
import os
import pickle
import time

from reasoners import Reasoner
from reasoners.algorithm import MCTS
from reasoners.lm import OpenAIModel
from browsergym.core.action.highlevel import HighLevelActionSet
from browsergym.experiments import EnvArgs

from gym_env import EnvironmentGym
from search_config import SearchConfigBrowsergym
from utils.misc import obs_preprocessor
from utils.parse import parse_common_arguments


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run a task with specified parameters."
    )
    parse_common_arguments(parser)

    # MCTS parameters
    parser.add_argument(
        "--n_iters", type=int, default=2, help="Number of iterations for MCTS."
    )
    parser.add_argument(
        "--depth_limit", type=int, default=10, help="Depth limit for MCTS."
    )
    parser.add_argument(
        "--w_exp",
        type=float,
        default=10**0.5,
        help="Exploration weight of the UCT score for MCTS.",
    )

    return parser.parse_args()


def run_task(args):
    browser_action_set = HighLevelActionSet(
        subsets=[args.action_set],
        strict=False,
        multiaction=True,
        demo_mode="off",
    )

    env_args = EnvArgs(
        task_name=args.task_name,
        task_seed=args.task_seed,
        max_steps=args.max_steps,
        headless=True,
        record_video=True,
    )

    exp_dir = os.path.join(args.exp_dir, args.task_name)
    os.makedirs(exp_dir, exist_ok=True)

    env = env_args.make_env(
        action_mapping=browser_action_set.to_python_code,
        exp_dir=exp_dir,
    )

    llm = OpenAIModel(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        backend=args.backend,
    )

    world_model = EnvironmentGym(env=env, obs_preprocessor=obs_preprocessor)
    search_config = SearchConfigBrowsergym(
        action_set=browser_action_set,
        n_proposals=10,
        llm=llm,
        use_axtree=True,
        use_html=False,
        use_screenshot=False,
    )
    algorithm = MCTS(
        n_iters=args.n_iters,
        depth_limit=args.depth_limit,
        w_exp=args.w_exp,
        uct_with_fast_reward=True,
        disable_tqdm=False,
        output_trace_in_each_iter=True,
    )

    reasoner = Reasoner(world_model, search_config, algorithm)

    plan_result = reasoner()

    with open(f"{exp_dir}/result.pkl", "wb") as f:
        pickle.dump(plan_result, f)

    env.close()

    return plan_result.terminal_state and plan_result.terminal_state.reward == 1.0


if __name__ == "__main__":
    args = parse_arguments()

    start_time = time.time()
    success = run_task(args)

    if success:
        print("Task completed successfully.")
    else:
        print(
            "Task didn't reach the goal. Please check the detailed result w/ visualization (python visualize.py --task_name <task_name>).",
        )

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")


# import argparse
# import os
# import pickle
# import time
# import json
# from tqdm import tqdm

# from reasoners import Reasoner
# from reasoners.algorithm import MCTS
# from reasoners.lm import OpenAIModel
# from browsergym.core.action.highlevel import HighLevelActionSet
# from browsergym.experiments import EnvArgs

# from gym_env import EnvironmentGym
# from search_config import SearchConfigBrowsergym
# from utils.misc import obs_preprocessor
# from utils.parse import parse_common_arguments


# def parse_arguments():
#     parser = argparse.ArgumentParser(
#         description="Run a task with specified parameters."
#     )
#     parse_common_arguments(parser)

#     # MCTS parameters
#     parser.add_argument(
#         "--n_iters", type=int, default=2, help="Number of iterations for MCTS."
#     )
#     parser.add_argument(
#         "--depth_limit", type=int, default=10, help="Depth limit for MCTS."
#     )
#     parser.add_argument(
#         "--w_exp",
#         type=float,
#         default=10**0.5,
#         help="Exploration weight of the UCT score for MCTS.",
#     )

#     return parser.parse_args()


# def run_task(args):
#     browser_action_set = HighLevelActionSet(
#         subsets=[args.action_set],
#         strict=False,
#         multiaction=True,
#         demo_mode="off",
#     )

#     env_args = EnvArgs(
#         task_name=args.task_name,
#         task_seed=args.task_seed,
#         max_steps=args.max_steps,
#         headless=True,
#         record_video=True,
#     )

#     exp_dir = os.path.join(args.exp_dir, args.task_name)
#     os.makedirs(exp_dir, exist_ok=True)

#     env = env_args.make_env(
#         action_mapping=browser_action_set.to_python_code,
#         exp_dir=exp_dir,
#     )

#     llm = OpenAIModel(
#         model=args.model,
#         temperature=args.temperature,
#         max_tokens=args.max_tokens,
#         backend=args.backend,
#     )

#     world_model = EnvironmentGym(env=env, obs_preprocessor=obs_preprocessor)
#     search_config = SearchConfigBrowsergym(
#         action_set=browser_action_set,
#         n_proposals=10,
#         llm=llm,
#         use_axtree=True,
#         use_html=False,
#         use_screenshot=False,
#     )
#     algorithm = MCTS(
#         n_iters=args.n_iters,
#         depth_limit=args.depth_limit,
#         w_exp=args.w_exp,
#         uct_with_fast_reward=True,
#         disable_tqdm=False,
#         output_trace_in_each_iter=True,
#     )

#     reasoner = Reasoner(world_model, search_config, algorithm)

#     plan_result = reasoner()

#     # Save the result and action history
#     with open(f"{exp_dir}/result.pkl", "wb") as f:
#         pickle.dump(plan_result, f)

#     # Save action history to a JSON file
#     action_history = plan_result.get_action_history()  # Assuming this method exists
#     with open(f"{exp_dir}/action_history.json", "w") as f:
#         json.dump(action_history, f)

#     env.close()

#     return plan_result.terminal_state and plan_result.terminal_state.reward == 1.0


# if __name__ == "__main__":
#     args = parse_arguments()

#     start_time = time.time()
#     success = run_task(args)

#     if success:
#         print("Task completed successfully.")
#     else:
#         print(
#             "Task didn't reach the goal. Please check the detailed result w/ visualization (python visualize.py --task_name <task_name>).",
#         )

#     end_time = time.time()
#     print(f"Time taken: {end_time - start_time} seconds")
