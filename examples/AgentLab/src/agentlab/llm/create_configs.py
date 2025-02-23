from agentlab.llm.chat_api import (
    AzureModelArgs,
    OpenAIModelArgs,
    OpenRouterModelArgs,
    SelfHostedModelArgs,
)


# this file is used for the best-of-n experiment of assistantbench
# used to create OpenAIModelArgs with different n paramter and logprob parameter
# logprobs on by default, as logprobs are used to determine "best" answer

def create_best_of_n_agent(sample_n, log_probs = True):
    if sample_n <= 1:
        raise ValueError('sample_n parameter has to be greater than 1')
    best_of_n_args = OpenAIModelArgs(
        model_name="gpt-4o-mini-2024-07-18",
        max_total_tokens=128_000,
        max_input_tokens=128_000,
        max_new_tokens=16_384,
        vision_support=True,
        n = sample_n,
        logprobs = log_probs
    )
    return best_of_n_args


def create_reward_model():
    reward_model_args = OpenAIModelArgs(
        model_name="gpt-4o-mini-2024-07-18",
        max_total_tokens=128_000,
        max_input_tokens=128_000,
        max_new_tokens=16_384,
        vision_support=True,
        n=1,
        logprobs=False
    )
    return reward_model_args