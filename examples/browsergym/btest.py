# baseline without tree search

from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import pickle
import time
import openai
from support_good import get_env, reset_env, step_env, get_clustered_action_proposals, create_logger, log_obs, log_reward, get_browser_action_set, get_parsed_evaluations_of_action_proposals

completion_dict = {

}


def run_task_baseline(task_name: str):
    logger = create_logger(task_name)

    seed = 16
    action_set = get_browser_action_set()
    action_history = []
    obs_history = []

    env = get_env(task_name, action_set, seed,
                  exp_dir=f"./results/baseline/{task_name}")

    openai_client = openai.OpenAI(
        api_key="[key]"
    )

    obs, env_info = reset_env(env, seed, logger)
    reward, terminated, truncated = None, False, False
    obs_history.append(obs)

    print(obs["goal_object"])

    steps = 0
    while not terminated and not truncated and steps < 10:  # 10 steps max

        action_proposals = get_clustered_action_proposals(
            obs, action_set, action_history, openai_client, logger=logger, n=1)

        action_proposal = action_proposals[0]
        action_history.append(action_proposal)
        obs, reward, terminated, truncated, step_info = step_env(
            env, action_proposal, logger)
        obs_history.append(obs)

        # print(reward, terminated, truncated)
        steps += 1

    # print(terminated, truncated, reward)

    if reward == 1.0:
        print("TASK COMPLETED SUCCESSFULLY")
    else:
        print("TASK FAILED")

    completion_dict[task_name] = {
        "reward": reward,
        "obs_history": obs_history,
        "action_history": action_history
    }

    time.sleep(1)

    env.close()

    success = reward == 1.0
    print(f"TASK {task_name} SUCCESS: {success}")
    return success


tasks = [
    # "webarena.27", "webarena.28", "webarena.29", # failure
    # "webarena.30", "webarena.31", # failure

    # change bio to ...
    # "webarena.399", "webarena.400", "webarena.401", # success
    # "webarena.402", "webarena.403" # failure. strage considering it got it right before
    # "webarena.405", # "webarena.406",  # failure
    # "webarena.410", # failure

    # "webarena.596",  # failure
    # "webarena.597", # failure

    # "webarena.599", # failure
    "webarena.619",  # failure

    # "webarena.642", # failure
    # "webarena.66",  # failure
    # "webarena.67", # failure
    # "webarena.68",  # failuk
    # "webarena.68",  # failure
    # "webarena.69", # failure
    # "webarena.718", # failure
    # "webarena.731" # failure
]

# successes = 0
# for task in tasks:
#     success = run_task_baseline(task)
#     if success:
#         successes += 1


if __name__ == "__main__":
    with Pool(processes=4) as pool:
        results = pool.map(run_task_baseline, tasks)

# print(f"SUCCESS RATE: {successes}/{len(tasks)}")


# Example list of tasks (functions to execute)
# tasks = ['task1', 'task2', 'task3']

# Using ThreadPoolExecutor for I/O-bound tasks


# def parallel_with_threads(tasks):
#     with ThreadPoolExecutor(max_workers=4) as executor:
#         futures = [executor.submit(run_task_baseline, task) for task in tasks]
#         for future in as_completed(futures):
#             print(future.result())


# # Execute one of the following:
# parallel_with_threads(tasks)      # For I/O-bound tasks

with open("./results/baseline/completion_dict.pkl", "wb") as f:
    pickle.dump(completion_dict, f)
