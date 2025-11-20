"""
Risk Control for LLM as a Judge
===============================

This example demonstrates how to use risk control methods for Large Language Models (LLMs) acting as judges.
We simulate a scenario where an LLM evaluates answers, and we want to control the risk of hallucination detection.
"""

# %%
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split

from mapie.risk_control import BinaryClassificationController

pd.set_option("display.width", 1000)

##############################################################################
# First, we load HaluEval Question-Answering Data, an open-source dataset for evaluating hallucination in LLMs.
# Then, we preprocess the data to create a suitable format for our analysis.
url = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json"
df = pd.read_json(url, lines=True)
print("Original dataset:\n", df.head())

# Melt the dataframe to combine right_answer and hallucinated_answer into a single column
df = df.melt(
    id_vars=["knowledge", "question"],
    value_vars=["right_answer", "hallucinated_answer"],
    var_name="answer_type",
    value_name="answer",
    ignore_index=False,  # Keep the original index to allow sorting back to pairs
)

# Sort by index to keep the pairs together (right_answer and hallucinated_answer for the same question)
df = df.sort_index()

# Create the 'hallucinated' flag based on the original column name and drop the helper column 'answer_type'
df["hallucinated"] = df["answer_type"] == "hallucinated_answer"
df = df.drop(columns=["answer_type"])
df = df.reset_index(drop=True)

# Create judge input prompts
df["judge_input"] = df.apply(
    lambda row: f"""
    You are a judge evaluating whether an answer to a question is faithful to the provided knowledge snippet.

    Knowledge: {row["knowledge"]}
    Question: {row["question"]}
    Answer: {row["answer"]}

    Does the answer contain information that is NOT supported by the knowledge?

    Provide a score between 0.0 and 1.0 indicating the probability that the answer is a hallucination.
""",
    axis=1,
)

print(df.head())


# %%
##############################################################################
# For demonstration purposes, we simulate the LLM judge's behavior using a simple table-based predictor.
# In practice, you would replace this with actual LLM API calls to get judge scores or read from a file
# of judge scores obtained from a complex LangChain pipeline for instance.


class TableBasePredictor:
    def __init__(self, df):
        df["judge_score"] = df["hallucinated"].apply(self.generate_biased_score)
        self.df = df[["judge_input", "judge_score"]]
        self.df = self.df.set_index("judge_input")

    def predict_proba(self, X):
        score_positive = self.df.loc[X]["judge_score"].values
        score_negative = 1 - score_positive
        return np.vstack([score_negative, score_positive]).T

    @staticmethod
    def generate_biased_score(is_hallucinated):
        """Generate a biased score based on whether the answer is hallucinated."""
        if is_hallucinated:
            return np.random.beta(a=3, b=1)
        else:
            return np.random.beta(a=1, b=3)


llm_judge = TableBasePredictor(df)

# %%
##############################################################################
# Next, we split the data into calibration and test sets. We then initialize a
# :class:`~mapie.risk_control.BinaryClassificationController` using the LLM judge's
# probability estimation function, a risk metric (here, "precision"), a target risk level,
# and a confidence level. We use the calibration data to compute statistically guaranteed thresholds.

X = df["judge_input"].to_numpy()
y = df["hallucinated"].astype(int)

X_calib, X_test, y_calib, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42
)

bcc = BinaryClassificationController(
    predict_function=llm_judge.predict_proba,
    risk="precision",
    target_level=0.9,
    confidence_level=0.9,
    best_predict_param_choice="recall",
)
bcc.calibrate(X_calib, y_calib)

print(f"The best threshold is: {bcc.best_predict_param}")

y_pred_controlled = bcc.predict(X_test)
precision = precision_score(y_test, y_pred_controlled)

print(f"Precision on the test set is: {precision}")
# %%
