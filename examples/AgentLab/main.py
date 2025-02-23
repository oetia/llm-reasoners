"""
Note: This script is a convenience script to launch experiments instead of using
the command line.

Copy this script and modify at will, but don't push your changes to the
repository.
"""

import logging

from agentlab.agents.generic_agent import (
    AGENT_LLAMA3_70B,
    AGENT_LLAMA31_70B,
    RANDOM_SEARCH_AGENT,
    AGENT_4o,
    AGENT_4o_MINI,
)
from agentlab.agents.plan_agent import PLAN_AGENT_4o_MINI, BASELINE_AGENT_4o_MINI
from agentlab.agents.generic_agent.agent_configs import GenericAgentArgs, BestOfNAgentArgs, BestOfNWithRewardAgentArgs
from agentlab.experiments.study import Study
from agentlab.llm.create_configs import create_best_of_n_agent, create_reward_model
from agentlab.agents.plan_agent.configs import BASELINE_AGENT_FLAGS 


ignore_dependencies = True
avg_step_timeout = 200

logging.getLogger().setLevel(logging.DEBUG)
save_logs = False
if save_logs:
    logging.basicConfig(filename="agentlab.log", level=logging.DEBUG)

# choose your agent or provide a new agent

# BEST_OF_N_AGENT_ARGS = BestOfNAgentArgs(
#     chat_model_args=create_best_of_n_agent(4),
#     flags=BASELINE_AGENT_FLAGS,
# )
# agent_args = [PLAN_AGENT_4o_MINI]

# agent_args = [BASELINE_AGENT_4o_MINI] # this makes <class 'agentlab.agents.generic_agent.generic_agent.GenericAgent'>

# agent_args = [BEST_OF_N_AGENT_ARGS]

# ## select the benchmark to run on
# benchmark = "miniwob_tiny_test"
# benchmark = "miniwob"
# benchmark = "workarena_l1"
# benchmark = "workarena_l2"
# benchmark = "workarena_l3"
benchmark = "webarena"

# Set reproducibility_mode = True for reproducibility
# this will "ask" agents to be deterministic. Also, it will prevent you from launching if you have
# local changes. For your custom agents you need to implement set_reproducibility_mode
reproducibility_mode = False

# Set relaunch = True to relaunch an existing study, this will continue incomplete
# experiments and relaunch errored experiments
relaunch = False

## Number of parallel jobs
n_jobs = 5  # Make sure to use 1 job when debugging in VSCode
# n_jobs = -1  # to use all available cores

tiny_test_task_names = [
    # "assistantbench.validation.1",
    # "assistantbench.validation.2", # task doesn't require visiting TripAdvisor or Yelp (hangs bc the top google hit it full of ads, content gets blocked)
    # "assistantbench.validation.3", # task doesn't require visiting TripAdvisor or Yelp (zillow/redfin hangs really bad)
    # "assistantbench.validation.4",
    # "assistantbench.validation.5", # task doesn't require visiting TripAdvisor or Yelp
    # "assistantbench.validation.6", # task doesn't require visiting TripAdvisor or Yelp
    # "assistantbench.validation.7", # task doesn't require visiting TripAdvisor or Yelp
    # "assistantbench.validation.8", # task doesn't require visiting TripAdvisor or Yelp
    # "assistantbench.validation.9", # task doesn't require visiting TripAdvisor or Yelp
    # "assistantbench.validation.10", # task doesn't require visiting TripAdvisor or Yelp
    # # # "assistantbench.validation.11", # task doesn't require visiting TripAdvisor or Yelp (yahoo fiannce, hangs bc updates to websites are too frequent?)
    # "assistantbench.validation.12", # task doesn't require visiting TripAdvisor or Yelp
    # # # "assistantbench.validation.13", # task doesn't require visiting TripAdvisor or Yelp (zillow hangs really bad)
    # "assistantbench.validation.14", # task doesn't require visiting TripAdvisor or Yelp
    # # # # "assistantbench.validation.15",
    # "assistantbench.validation.16", # task doesn't require visiting TripAdvisor or Yelp
    # "assistantbench.validation.17", # task doesn't require visiting TripAdvisor or Yelp
    # "assistantbench.validation.18", # task doesn't require visiting TripAdvisor or Yelp
    # # "assistantbench.validation.19", # task doesn't require visiting TripAdvisor or Yelp
    # # "assistantbench.validation.20", # task doesn't require visiting TripAdvisor or Yelp
    # # "assistantbench.validation.21", # task doesn't require visiting TripAdvisor or Yelp
    # # "assistantbench.validation.22", # task doesn't require visiting TripAdvisor or Yelp
    # # # "assistantbench.validation.23",
    # # "assistantbench.validation.24", # task doesn't require visiting TripAdvisor or Yelp
    # # # "assistantbench.validation.25",
    # # "assistantbench.validation.26", # task doesn't require visiting TripAdvisor or Yelp
    # # "assistantbench.validation.27", # task doesn't require visiting TripAdvisor or Yelp
    # # "assistantbench.validation.28", # task doesn't require visiting TripAdvisor or Yelp
    # # "assistantbench.validation.29", # task doesn't require visiting TripAdvisor or Yelp
    # # "assistantbench.validation.30", # task doesn't require visiting TripAdvisor or Yelp
    # # "assistantbench.validation.31", # task doesn't require visiting TripAdvisor or Yelp
    # # "assistantbench.validation.32", # task doesn't require visiting TripAdvisor or Yelp
    # # reddit
    "webarena.27", # number of comments that have downvotes > upvotes in the latest post in a subreddit
    "webarena.66", # among the top K posts in a subreddit, tell me xyz
    "webarena.399", # change my bio
    "webarena.404", # upvote the newest post in a subbreddit
    "webarena.409", # reply to this post
    "webarena.580", # create a forum
    "webarena.595", # open a thread of a trending post and sub
    "webarena.600", # post my question
    "webarena.605", # find a subreddit focused on topic and post my question
    "webarena.610", # post a review of recent reading
    "webarena.615", # repost image from one subreddit to another
    "webarena.620", # ask for advice about relationships
    "webarena.627", # create post about topic in relevant subreddit
    "webarena.630", # ask for product recommendation with certain budget
    "webarena.635", # same as above
    "webarena.640", # post a notice for virtual meetup for interest group
    "webarena.647", # post in a technology forum on how to use LLMs for that field
    "webarena.650", # reply to a specific post
    "webarena.718", # thumbs down the top 5 post in a subreddit
    "webarena.720", # like all submission by a user in a subreddit 
    "webarena.726", # dislike all submission by a user in a subreddit 
    "webarena.731", # edit an existing post by me

]

if __name__ == "__main__":  # necessary for dask backend
    # import ray
    # ray.init()

    # BEST_OF_2_AGENT_ARGS = BestOfNAgentArgs(
    #     chat_model_args=create_best_of_n_agent(2),
    #     flags=BASELINE_AGENT_FLAGS,
    # )

    BEST_OF_2_AGENT_WITH_REWARD_ARGS = BestOfNWithRewardAgentArgs(
        proposer_model_args=create_best_of_n_agent(2, log_probs=False),
        reward_model_args=create_reward_model(),
        flags=BASELINE_AGENT_FLAGS
    )

    BEST_OF_4_AGENT_WITH_REWARD_ARGS = BestOfNWithRewardAgentArgs(
        proposer_model_args=create_best_of_n_agent(4, log_probs=False),
        reward_model_args=create_reward_model(),
        flags=BASELINE_AGENT_FLAGS
    )

    BEST_OF_8_AGENT_WITH_REWARD_ARGS = BestOfNWithRewardAgentArgs(
        proposer_model_args=create_best_of_n_agent(8, log_probs=False),
        reward_model_args=create_reward_model(),
        flags=BASELINE_AGENT_FLAGS
    )

    BEST_OF_16_AGENT_WITH_REWARD_ARGS = BestOfNWithRewardAgentArgs(
        proposer_model_args=create_best_of_n_agent(16, log_probs=False),
        reward_model_args=create_reward_model(),
        flags=BASELINE_AGENT_FLAGS
    )
    BEST_OF_32_AGENT_WITH_REWARD_ARGS = BestOfNWithRewardAgentArgs(
        proposer_model_args=create_best_of_n_agent(32, log_probs=False),
        reward_model_args=create_reward_model(),
        flags=BASELINE_AGENT_FLAGS
    )

    BEST_OF_64_AGENT_WITH_REWARD_ARGS = BestOfNWithRewardAgentArgs(
        proposer_model_args=create_best_of_n_agent(64, log_probs=False),
        reward_model_args=create_reward_model(),
        flags=BASELINE_AGENT_FLAGS
    )

    # BEST_OF_128_AGENT_ARGS = BestOfNAgentArgs(
    #     chat_model_args=create_best_of_n_agent(128),
    #     flags=BASELINE_AGENT_FLAGS,
    # )
        # BEST_OF_16_AGENT_ARGS = BestOfNAgentArgs(
    #     chat_model_args=create_best_of_n_agent(2),
    #     flags=BASELINE_AGENT_FLAGS,
    # )

    # BEST_OF_32_AGENT_ARGS = BestOfNAgentArgs(
    #     chat_model_args=create_best_of_n_agent(2),
    #     flags=BASELINE_AGENT_FLAGS,
    # )

    # BEST_OF_32_AGENT_ARGS = BestOfNAgentArgs(
    #     chat_model_args=create_best_of_n_agent(2),
    #     flags=BASELINE_AGENT_FLAGS,
    # )

    agent_args_list = [
        BEST_OF_2_AGENT_WITH_REWARD_ARGS,
        BEST_OF_4_AGENT_WITH_REWARD_ARGS,
        BEST_OF_8_AGENT_WITH_REWARD_ARGS,
        BEST_OF_16_AGENT_WITH_REWARD_ARGS,
        BEST_OF_32_AGENT_WITH_REWARD_ARGS,
        BEST_OF_64_AGENT_WITH_REWARD_ARGS]#, BEST_OF_4_AGENT_ARGS]
    # agent_args_list = [BEST_OF_2_AGENT_ARGS]#, BEST_OF_4_AGENT_ARGS]



    for agent_args in agent_args_list:
        agent_args = [agent_args]
        if reproducibility_mode:
            [a.set_reproducibility_mode() for a in agent_args]

        if relaunch:
            #  relaunch an existing study
            study = Study.load_most_recent(contains=None)
            study.find_incomplete(include_errors=True)

        else:
            study = Study(
                agent_args,
                benchmark,
                logging_level_stdout=logging.WARNING,
                tiny_test_task_names=tiny_test_task_names,
                ignore_dependencies=ignore_dependencies,
                avg_step_timeout=avg_step_timeout,
            )

            study.run(
                n_jobs=n_jobs,
                parallel_backend="ray",
                strict_reproducibility=reproducibility_mode,
                n_relaunch=1,
            )

            if reproducibility_mode:
                study.append_to_journal(strict_reproducibility=True)
