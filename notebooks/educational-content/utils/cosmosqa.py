from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import LabelEncoder


class TransformCosmosQA:
    """
    Transform CosmosQA data into a format that can be used for prompt-based
    learning. This class takes as input the CosmosQA data which has the
    following structure:
    [
        {
            "id": int,
            "context": str,
            "question": str,
            "choices": {
                "A": str,
                "B": str,
                "C": str,
                "D": str
            },
            "answer": str
        },
        ...
    ]

    The output of this class is a tuple of two numpy arrays (X, Y) where X is
    the input prompt and Y is the label. The input prompt is a string that
    contains the context, question, and choices. The context is the same for all
    the few-shot examples and the question-answer pair. The choices are the same
    for the question-answer pair. The context, question, and choices are
    separated by newlines. The answer is not included in the input prompt.
    """

    base_prompt = """
The following is a multiple-choice question about reading comprehension.
You should answer the question based on the given context and you can use
commonsense reasoning when necessary. Please reason step-by-step and select
the correct answer. You only need to output the option.\n\n
"""

    few_shot_examples_ids = [1, 3, 5, 7, 9]

    def __init__(self, data: List[Dict[str, str]]) -> None:
        self.data = data
        self.label_encoder = LabelEncoder()

    def _get_fewshot_examples(self) -> List[Dict[str, str]]:
        """
        Retrieve the few-shot examples from the data.
        """
        fewshot_examples = []
        for idx in self.few_shot_examples_ids:
            fewshot_examples.append(self.data[idx])
            assert self.data[idx]["id"] == idx
        return fewshot_examples

    @staticmethod
    def _format_example(
        example: Dict[str, str], prompt: str, with_answer: bool = False
    ) -> str:
        """
        Format a question-answer pair. The prompt contains the context,
        question, and choices. When this function is called with
        with_answer=True, the answer is included in the prompt.
        """
        prompt += (
            "Context: "
            + example["context"]
            + "\n"
            + "Question: "
            + example["question"]
            + "\nChoices:\n"
        )
        for k, v in example["choices"].items():
            prompt += k + ". " + str(v) + "\n"
        prompt += "Answer:"
        if with_answer:
            prompt += " " + example["answer"] + "\n"
        return prompt

    def _format_prompt(
        self, example: Dict[str, str], fewshot_examples: List[Dict[str, str]]
    ) -> str:
        """
        Format the prompt for a question-answer pair. The prompt contains the
        context, question, and choices, as well as the few-shot examples
        at the beginning. The few-shot examples are formatted using the
        _format_example function. The answer is not included in the prompt.
        """
        prompt = self.base_prompt
        for fewshot_example in fewshot_examples:
            prompt = self._format_example(
                fewshot_example, prompt, with_answer=True
            )
        prompt = self._format_example(example, prompt, with_answer=False)
        return prompt

    def _fit_label_encoder(self) -> None:
        """
        Fit a label encoder to the possible answers.
        """
        possible_answers = []
        for question_answer in self.data:
            choices = question_answer["choices"].keys()
            possible_answers += list(choices)
        possible_answers = sorted(list(set(possible_answers)))
        self.label_encoder.fit(possible_answers)

    def transform_data(self) -> Tuple[NDArray, NDArray]:
        """
        Transform the data into a format that can be used for prompt-based
        learning. The transformation is done according to the following steps:
        1. Fit a label encoder to the possible answers.
        2. Retrieve the few-shot examples to put at the beginning of the prompt.
        3. Format the prompt for each question-answer pair.
        4. Transform the answers into labels using the label encoder.
        5. Return the input prompts and the labels as numpy arrays.
        """
        self._fit_label_encoder()
        X, Y = [], []
        fewshot_examples = self._get_fewshot_examples()
        for question_answer in self.data:
            prompt = self._format_prompt(question_answer, fewshot_examples)
            X.append(prompt)
            Y.append(self.label_encoder.transform([question_answer["answer"]]))
        return np.array(X), np.array(Y).flatten()
