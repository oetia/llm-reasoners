
import openai
from typing import Callable, Tuple, Optional


def IDENTITY(x):
    return x, True, None


class LLM:
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.client = openai.Client(api_key=api_key)

    def __call__(self, system_prompt: str, user_prompt: str, parser: Callable[[str], Tuple[str, bool, Optional[str]]] = IDENTITY, **kwargs):

        print("KWARGS BEING PASSED TO OPENAI API")
        print(kwargs)

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            # model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **kwargs
        )

        answer_dicts = []
        for choice in response.choices:
            content = choice.message.content
            parsed_content = parser(content)
            answer_dict = parsed_content[0]
            answer_dicts.append(answer_dict)

        return answer_dicts
