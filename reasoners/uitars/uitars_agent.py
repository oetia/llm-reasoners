from llm_reasoners import BaseReasoner
from .utils import preprocess_input, postprocess_output

class UITARSAgent(BaseReasoner):
    """
    UITARS Agent for task decomposition and reasoning.
    Inherits from BaseReasoner in llm-reasoners.
    """

    def __init__(self, model, config=None):
        """
        Initialize the UITARS agent.

        Args:
            model: The LLM model to use for reasoning.
            config (dict, optional): Configuration for the agent. Defaults to None.
        """
        super().__init__(model, config)
        self.config = config or {}
        # Initialize any UITARS-specific components here
        self.initialized = self._initialize_uitars()

    def _initialize_uitars(self):
        """
        Initialize UITARS-specific components.
        """
        # Add any initialization logic here
        return True

    def run(self, input_text):
        """
        Run the UITARS agent on the input text.

        Args:
            input_text (str): The input text to process.

        Returns:
            list: A list of reasoning steps or outputs.
        """
        # Preprocess the input (e.g., tokenization, formatting)
        processed_input = preprocess_input(input_text, self.config)

        # Perform UITARS reasoning
        reasoning_steps = self._uitars_reasoning(processed_input)

        # Postprocess the output (e.g., formatting, filtering)
        final_output = postprocess_output(reasoning_steps, self.config)

        return final_output

    def _uitars_reasoning(self, input_text):
        """
        Core UITARS reasoning logic.

        Args:
            input_text (str): The preprocessed input text.

        Returns:
            list: A list of reasoning steps.
        """
        # Example: Use the LLM to generate reasoning steps
        prompt = f"Perform task decomposition for: {input_text}"
        reasoning_steps = self.model.generate(prompt)

        # Add any UITARS-specific logic here
        return reasoning_steps
