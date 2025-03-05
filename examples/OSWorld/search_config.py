import re
import json
import random
from reasoners import SearchConfig, LanguageModel

from gym_env import ActionGym, StateGym
from OSWorld.mm_agents.prompts import UITARS_USR_PROMPT_NOTHOUGHT, UITARS_USR_PROMPT_THOUGHT


class SearchConfigOSWorld(SearchConfig):
    """
    SearchConfig for the OSWorld environment.

    Attributes
    ----------
    action_set : ACTION_SPACE
        list of json representing the action set for the OSWorld environment
    llm : LanguageModel
        the language model used for generating proposals and evaluations
    n_proposals : int
        the number of proposals to generate
    proposal_temperature : float
        the temperature to use for generating proposals
    evaluation_temperature : float
        the temperature to use for generating evaluations
    use_axtree : bool
        whether to use the axtree in the prompts
    use_html : bool
        whether to use the page's html in the prompts
    use_screenshot : bool
        whether to use the screenshot (base64 encoded) in the prompts
    """

    def __init__(
        self,
        agent,
        instruction,
        n_proposals: int = 5,
        proposal_temperature: float = 0.7,
        evaluation_temperature: float = 0.25,
        use_axtree: bool = True,
        use_html: bool = False,
        use_screenshot: bool = False,
    ) -> None:
        super().__init__()
        self.agent = agent
        self.instruction = instruction
        self.n_proposals = n_proposals
        self.proposal_temperature = proposal_temperature
        self.evlaution_temperature = evaluation_temperature
        self.use_axtree = use_axtree
        self.use_html = use_html
        self.use_screenshot = use_screenshot

    # TODO : Use a different proposal prompt. if the main additional
    # input is just going to be a screenshot, then most of the complexity in
    # utils/prompts.py can be cut down. there should already be a screenshot
    # append functionality in the browsergym prompt, so that should be reusable.
    # though the action space definitely needs to be redefined.
    def get_actions(self, state: StateGym) -> list[ActionGym]:
        """
        Generate a list of action proposals for the provided state.
        Proposals are generated by re-running the prompt at high temperature.
        Due to this, there can be duplicate proposals, so clustering is
        performed by looking at the final code that the proposal would run,
        and checking for uniqueness.

        Smaller LLMs (i.e. 4o-mini) often have issues with generating calls
        to functions that do not exist. Each function call in the proposal
        is checked to see if it is valid (check_validity_of_action_proposal).

        Parameters
        ----------
        state : StateGym
            the state to generate proposals for

        Returns
        -------
        actions : list[ActionGym]
            a list of unique action proposals
        """
        response, actions = self.agent.predict(self.instruction, state.current_obs)
        return actions

    def get_response(self, state: StateGym) -> list[ActionGym]:
        """
        Gets a response based on the list of possible actions and states

        Parameters
        ----------
        state : StateGym
            the state to generate proposals for

        Returns
        -------
        clustered_actions : list[ActionGym]
            a list of unique action proposals
        """
        response, actions = self.agent.predict(self.instruction, state.current_obs)
        return response

    # this is called when mcts generates a new set of nodes, and needs to
    # decide which to visit next. since there are no visitation statistics
    # accrued at this point, it relies on an llm to generate an evaluation.
    # the prompt for this can also be found under OSWorld/mm_agents/prompts.py.
    def fast_reward(self, state: StateGym, action: ActionGym) -> tuple[float, dict]:
        """
        Generate an evaluation of a state action pair before using the action
        to step the environment. This process is entirely dependent on the
        LLM providing an accurate evaluation. The LLM provides a score from
        0 to 10, which is then divided by 10 to keep the reward between
        0 and 1 (important for UCT calculation in MCTS).

        Parameters
        ----------
        state : StateGym
            the state to evaluate
        action : ActionGym
            the action to evaluate

        Returns
        -------
        evaluation : float
            the evaluation of the state action pair
        aux : dict
            used to pass the self-evaluation to the search algorithm, which
            then passes it to the SearchConfig's reward (not fast_reward) function
        """
        # use self evaluation to replace random number
        prompt = UITARS_USR_PROMPT_THOUGHT.format(action_space=self.agent.action_set,
                                                     language=self.agent.thoughts,
                                                     instruction=self.instruction
                                                    )

        # predict based off prompt and current state, then get response value
        llm_response = self.get_response(prompt, state.current_obs)
        print("fast_reward() Response Value: ", llm_response)
        epsilon = 100

        try:
            # fine grain score between 0 and 100, then normalize
            score = float(llm_response)
            score = max(0, min(score, epsilon))  
            normalized_score = score / epsilon   
        except ValueError:
            print("Response returned my self.get_response is not a scalar: ", ValueError)
            # Default to neutral score if parsing fails
            normalized_score = 0.5  

        return normalized_score, {"self_eval": normalized_score}

        # system_msgs, user_msgs, full_prompt_txt = build_evaluation_prompt(
        #     state.current_obs,
        #     action,
        #     self.action_set,
        #     state.action_history,
        #     self.use_axtree,
        #     self.use_html,
        #     self.use_screenshot,
        # )

        # response = self.llm.generate(
        #     full_prompt_txt,
        #     num_return_sequences=self.n_proposals,
        #     temperature=self.proposal_temperature,
        # )

        # evaluation = response.text[0]

        # json_string = re.search(r"\{.*\}", evaluation, re.DOTALL).group()
        # json_object = json.loads(json_string)
        # evaluation = json_object["score"] / 10

        # return evaluation, {"self_eval": evaluation}

    def reward(
        self, state: StateGym, action: ActionGym, **kwargs
    ) -> tuple[float, dict]:
        """
        Generate a reward for a state action pair after stepping the environment
        with an action. The kwargs passed in are the combined aux dictionaries
        from the SearchConfig's fast_reward and EnvironmentGym's step functions.
        The env_reward for the browsergym environment is sparse, so a massive
        weight is provided to the environment's reward.

        Parameters
        ----------
        state : StateGym
            the state to evaluate
        action : ActionGym
            the action to evaluate
        kwargs : kwargs
            combined aux dictionaries, varying in length

        Returns
        -------
        tuple : (float, dict)
            weighted reward along with the aux dictionaries
        """
        
        return kwargs["self_eval"] + 100 * kwargs["env_reward"], kwargs
