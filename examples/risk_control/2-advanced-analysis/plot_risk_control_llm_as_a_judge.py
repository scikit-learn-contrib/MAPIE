"""
Risk Control for LLM as a Judge
===============================

This example demonstrates how to use risk control methods for Large Language Models (LLMs) acting as judges.
We simulate a scenario where an LLM evaluates answers, and we want to control the risk of hallucination detection.
"""

# sphinx_gallery_thumbnail_number = 2

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split

from mapie.risk_control import BinaryClassificationController

RANDOM_STATE = 0
np.random.seed(RANDOM_STATE)

pd.set_option("display.max_colwidth", None)

##############################################################################
# First, we load HaluEval Question-Answering Data, an open-source dataset for evaluating hallucination in LLMs.
# Then, we preprocess the data to create a suitable format for our analysis.
url = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json"
df = pd.read_json(url, lines=True)

print("# Sample of the original dataset:\n")
for col in df.columns:
    print(f'"{col}":    {df[col].iloc[0]}\n')


# Melt the dataframe to combine right_answer and hallucinated_answer into a single column
df = df.melt(
    id_vars=["knowledge", "question"],
    value_vars=["right_answer", "hallucinated_answer"],
    var_name="answer_type",
    value_name="answer",
    ignore_index=False,  # Keep the original index to allow sorting back to pairs
)

# Sort by index to keep the pairs together (right_answer and hallucinated_answer for
# the same question)
df = df.sort_index()

# Create the 'hallucinated' flag based on the original column name and drop the helper
# column 'answer_type'
df["hallucinated"] = df["answer_type"] == "hallucinated_answer"
df = df.drop(columns=["answer_type"])
df = df.reset_index(drop=True)

# Create judge input prompts
df["judge_input"] = df.apply(
    lambda row: f"""
    You are a judge evaluating whether an answer to a question is faithful to the 
    provided knowledge snippet.

    Knowledge: {row["knowledge"]}
    Question: {row["question"]}
    Answer: {row["answer"]}

    Does the answer contain information that is NOT supported by the knowledge?

    Provide a score between 0.0 and 1.0 indicating the probability that the answer is a
    hallucination.
""",
    axis=1,
)


# Create synthetic judge scores
def generate_biased_score(is_hallucinated):
    """Generate a biased score based on whether the answer is hallucinated."""
    if is_hallucinated:
        return np.random.beta(a=3, b=1)
    else:
        return np.random.beta(a=1, b=3)

# for reproductibility of results across infrastruct
np.random.seed(12)
df["judge_score"] = df["hallucinated"].apply(generate_biased_score)

df = df[["judge_input", "judge_score", "hallucinated"]]
df = df.set_index("judge_input")


print("# Sample of the processed dataset:\n")
print(f'"{df.index.name}":    {df.index[0]}')
for col in df.columns:
    print(f'"{col}":    {df[col].iloc[0]}\n')


##############################################################################
# For demonstration purposes, we simulate the LLM judge's behavior using a simple table-based predictor.
# In practice, you would replace this with actual LLM API calls to get judge scores or read from a file
# of judge scores obtained from a complex LangChain pipeline for instance.


class TableBasePredictor:
    def __init__(self, df):
        self.df = df

    def predict_proba(self, X):
        score_positive = self.df.loc[X]["judge_score"].values
        score_negative = 1 - score_positive
        return np.vstack([score_negative, score_positive]).T


llm_judge = TableBasePredictor(df)

plt.figure()
plt.hist(
    df[df["hallucinated"]]["judge_score"],
    bins=30,
    alpha=0.8,
    label="Hallucinated answer",
    density=True,
)
plt.hist(
    df[~df["hallucinated"]]["judge_score"],
    bins=30,
    alpha=0.8,
    label="Correct answer",
    density=True,
)
plt.xlabel("Judge Score (Probability of Hallucination)")
plt.ylabel("Density")
plt.title("Distribution of Judge Scores")
plt.legend()
plt.show()

##############################################################################
# Next, we split the data into calibration and test sets. We then initialize a
# :class:`~mapie.risk_control.BinaryClassificationController` using the LLM judge's
# probability estimation function, a risk metric (here, "precision"), a target risk level,
# and a confidence level. We use the calibration data to compute statistically guaranteed thresholds.

X = df.index.to_numpy()  # index is the judge_input
y = df["hallucinated"].astype(int)

X_calib, X_test, y_calib, y_test = train_test_split(
    X, y, test_size=0.95, random_state=RANDOM_STATE
)
target_precision = 0.8
confidence_level = 0.9

bcc = BinaryClassificationController(
    predict_function=llm_judge.predict_proba,
    risk="precision",
    target_level=target_precision,
    confidence_level=confidence_level,
    best_predict_param_choice="recall",
)
bcc.calibrate(X_calib, y_calib)

print(f"The best threshold is: {bcc.best_predict_param}")

y_calib_pred_controlled = bcc.predict(X_calib)
precision_calib = precision_score(y_calib, y_calib_pred_controlled)

y_test_pred_controlled = bcc.predict(X_test)
precision_test = precision_score(y_test, y_test_pred_controlled)


##############################################################################
# Finally, let us visualize the precision achieved on the calibration set for
# the tested thresholds, highlighting the valid thresholds and the best one
# (which maximizes recall).

proba_positive_class = llm_judge.predict_proba(X_calib)[:, 1]

tested_thresholds = bcc._predict_params
precisions = np.full(len(tested_thresholds), np.inf)
for i, threshold in enumerate(tested_thresholds):
    y_pred = (proba_positive_class >= threshold).astype(int)
    precisions[i] = precision_score(y_calib, y_pred)

naive_threshold_index = np.argmin(
    np.where(precisions >= target_precision, precisions - target_precision, np.inf)
)
naive_threshold = tested_thresholds[naive_threshold_index]

valid_thresholds_indices = np.array(
    [t in bcc.valid_predict_params for t in tested_thresholds]
)
best_threshold_index = np.where(tested_thresholds == bcc.best_predict_param)[0][0]

plt.figure()
plt.scatter(
    tested_thresholds[valid_thresholds_indices],
    precisions[valid_thresholds_indices],
    c="tab:green",
    label="Valid thresholds",
)
plt.scatter(
    tested_thresholds[~valid_thresholds_indices],
    precisions[~valid_thresholds_indices],
    c="tab:red",
    label="Invalid thresholds",
)
plt.scatter(
    tested_thresholds[best_threshold_index],
    precisions[best_threshold_index],
    c="tab:green",
    label="Best threshold",
    marker="*",
    edgecolors="k",
    s=300,
)
plt.scatter(
    tested_thresholds[naive_threshold_index],
    precisions[naive_threshold_index],
    c="tab:red",
    label="Naive threshold",
    marker="*",
    edgecolors="k",
    s=300,
)
plt.axhline(target_precision, color="tab:gray", linestyle="--")
plt.text(
    0.7,
    target_precision + 0.02,
    "Target precision",
    color="tab:gray",
    fontstyle="italic",
)
plt.xlabel("Threshold")
plt.ylabel("Precision")
plt.legend()
plt.show()

proba_positive_class_test = llm_judge.predict_proba(X_test)[:, 1]
y_pred_naive = (proba_positive_class_test >= naive_threshold).astype(int)

print(
    "With the naive threshold, the precision is:\n"
    f"- {precisions[naive_threshold_index]:.3f} on the calibration set\n"
    f"- {precision_score(y_test, y_pred_naive):.3f} on the test set."
)

print(
    "\n\nWith risk control, the precision is:\n"
    f"- {precision_calib:.3f} on the calibration set \n"
    f"- {precision_test:.3f} on the test set."
)

##############################################################################
# While the naive threshold achieves the target precision on the calibration set,
# it fails to do so on the test set. This highlights the importance of using
# risk control methods to ensure that performance guarantees hold on unseen data.
