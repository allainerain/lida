import json
import logging
from lida.utils import clean_code_snippet
from llmx import TextGenerator, TextGenerationConfig, TextGenerationResponse
from lida.datamodel import Goal, Prompt

SYSTEM_PROMPT = """
You are a helpful and highly skilled data analyst who is trained to provide helpful, prompting questions to guide the user to gain insights from a data visualization given their goal. 

The questions you ask must be the following
- Incite insightful ideas and be meaningful.
- Be related to the goal and the visualization given.
- Clarify domain knowledge of the data ONLY if necessary. If you do this, add an example answer to guide the user.
- Include a rationale (justification for what we will learn from answering the question).
"""

FORMAT_INSTRUCTIONS = """
THE OUTPUT MUST BE A CODE SNIPPET OF A VALID LIST OF JSON OBJECTS. IT MUST USE THE FOLLOWING FORMAT:

```[
    { "index": 0,  "question": "What is the distribution of X", "rationale": "This tells about "} ..
    ]
```
THE OUTPUT SHOULD ONLY USE THE JSON FORMAT ABOVE.
"""

logger = logging.getLogger("lida")

class Prompter(object):
    """Generate questions given some goal"""

    def __init__(self) -> None:
        pass

    def generate(
            self, goal: Goal, 
            textgen_config: TextGenerationConfig, text_gen: TextGenerator, n=5):
        """Generate questions to prompt the user to interpret the chart given some code and goal"""

        user_prompt = f"""
        The visualization is: {goal.visualization}. The question the visualization wants to answer is: {goal.question}. The rationale for choosing the visualization is: {goal.rationale}.
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "assistant", "content": f"{user_prompt}\n\n{FORMAT_INSTRUCTIONS}\n\nThe generated {n} questions are:\n"}
        ]

        result: list[Prompt] = text_gen.generate(messages=messages, config=textgen_config)

        try:
            result = clean_code_snippet(result.text[0]["content"])
            result = json.loads(result)
            
            # cast each item in the list to a Prompt object
            if isinstance(result, dict):
                result = [result]
            result = [Prompt(**x) for x in result]
        except Exception as e:
            logger.info(f"Error decoding JSON: {result.text[0]['content']}")
            print(f"Error decoding JSON: {result.text[0]['content']}")
            raise ValueError(
                "The model did not return a valid JSON object while attempting to generate goals. Consider using a larger model or a model with a higher max token length.")

        return result
