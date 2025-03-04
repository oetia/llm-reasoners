def preprocess_input(input_text, config):
    """
    Preprocess the input text for UITARS.

    Args:
        input_text (str): The raw input text.
        config (dict): Configuration for preprocessing.

    Returns:
        str: The preprocessed input text.
    """
    # Example: Remove extra spaces, normalize text, etc.
    processed_text = input_text.strip()
    return processed_text

def postprocess_output(reasoning_steps, config):
    """
    Postprocess the output from UITARS.

    Args:
        reasoning_steps (list): The raw reasoning steps.
        config (dict): Configuration for postprocessing.

    Returns:
        list: The postprocessed output.
    """
    # Example: Filter out invalid steps, format the output, etc.
    final_output = [step for step in reasoning_steps if step]  # Remove empty steps
    return final_output
