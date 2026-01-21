"""
Risk Control for LLM as a Judge
===============================

This example demonstrates how to use risk control methods for Large Language Models (LLMs) acting as judges.
We simulate a scenario where an LLM evaluates answers, and we want to control the risk of hallucination detection.
Moreover, we want the judge to abstain from making a decision when uncertain.
That is, we want the the precision of deciding that an answer is hallucinated or not to be heigh, while controlling the rate of abstention.

In binary classification, this corresponds of finding two thresholds to apply on the model predicted probability scores, such that

- if the score is below the lower threshold, the answer is classified as "not hallucinated",
- if the score is above the upper threshold, the answer is classified as "hallucinated",
- if the score is between the two thresholds, the judge abstains from making a decision.

Hence, we want to control the precision of the non-abstained predictions, while minimizing the abstention rate.
The procedure falls into the scope of mutli-parameter (here two parameters: lower and upper thresholds)
and multi-risk (here three risks: precision on "hallucinated" class, precision on "not hallucinated" class, and abstention rate)
risk control for binary classification.
"""

# sphinx_gallery_thumbnail_number = 2

import matplotlib.patches as patches
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from numpy.typing import NDArray
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


# for reproductibility of results across infrastructure
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
# We now split the data into calibration and test sets.

X = df.index.to_numpy()  # index is the judge_input
y = df["hallucinated"].astype(int)

X_calib, X_test, y_calib, y_test = train_test_split(
    X, y, test_size=0.95, random_state=RANDOM_STATE
)

#############################################################################
# Next, we define a multi-parameter prediction function. For an answer to be classified
# as not hallucinated, we want the predicted score of the positive class to below a
# lower threshold `lambda_1`. For an answer to be classified as hallucinated, we want
# the predicted score of the positive class to be above an upper threshold `lambda_2`.
# For answers with intermediate scores, the judge abstains from making a decision.


def abstain_to_answer(X, lambda_1, lambda_2) -> NDArray[np.int_]:
    """Predict function with abstention based on two thresholds.
    - if the score is below `lambda_1`, predict 0 (not hallucinated)
    - if the score is above `lambda_2`, predict 1 (hallucinated)
    - if the score is between `lambda_1` and `lambda_2`, abstain (np.nan)
    """
    y_score = llm_judge.predict_proba(X)[:, 1]
    y_pred = np.full_like(y_score, np.nan)
    y_pred = np.where(y_score <= lambda_1, 0, y_pred)
    y_pred = np.where(y_score >= lambda_2, 1, y_pred)
    return y_pred


#############################################################################
# Given the abstention task and the definition of `abstain_to_answer`, we have
# the constraint that `lambda_1` <= `lambda_2`. Therefore, can avoid exploring
# the area of the bi-variate parameter set such that `lambda_1` > `lambda_2`.
# We can consider a set of a grid values that respects the former constraint.

to_explore = []
for i in range(9):
    lambda_1 = i / 10
    for j in range(i + 1, 10):
        lambda_2 = j / 10
        if lambda_2 > 0.99:
            break
        to_explore.append((lambda_1, lambda_2))
to_explore = np.array(to_explore)

##############################################################################
# Finally, we initialize a :class:`~mapie.risk_control.BinaryClassificationController`
# using the `abstain_to_answer` prediction function and three specific risks that are
# instances of :class:`BinaryClassificationRisk`:
#
# - `precision_negative` : the precision of the class 0, "negative class"
# - `precision_positive` : the precision of the class 1, "positive class"
# - `abstention_rate` : the rate of abstention of the judge, "proportion of np.nan predictions"
#
# We set target levels for each risk and use the calibration data to compute
# statistically guaranteed thresholds. Among the valid thresholds, we select the one that minimizes
# the abstention rate thus minimizing of human manual review.

target_precision_negative = 0.7
target_precision_positive = 0.7
target_abstention_rate = 0.2
confidence_level = 0.8

bcc = BinaryClassificationController(
    predict_function=abstain_to_answer,
    risk=["precision_negative", "precision_positive", "abstention_rate"],
    target_level=[
        target_precision_negative,
        target_precision_positive,
        target_abstention_rate,
    ],
    confidence_level=confidence_level,
    best_predict_param_choice="abstention_rate",
    list_predict_params=to_explore,
)
bcc.calibrate(X_calib, y_calib)

print(
    f"{len(bcc.valid_predict_params)} two-dimensional parameters found that guarantee with a confidence of {confidence_level}:\n"
    f"- precision of at least {target_precision_negative} for predicting not hallucinated,\n"
    f"- precision of at least {target_precision_positive} for predicting hallucinated, and\n"
    f"- an abstention rate of at most {target_abstention_rate}.\n\n"
    f"Among these, the best parameter that minimizes the abstention rate is {bcc.best_predict_param}.\n"
    "That is, using thresholds:\n"
    f"- lambda_1 = {bcc.best_predict_param[0]},\n"
    f"- lambda_2 = {bcc.best_predict_param[1]}"
)

##############################################################################
# Finally, we visualize the p-values associated with each explored parameter pair.
# The valid parameter zone is highlighted, as well as the best parameter pair.

grid_size = 10
matrix = np.full((grid_size, grid_size), np.nan)

# Build p-values matrix
for i, (l1, l2) in enumerate(to_explore):
    row = int(l1 * grid_size)
    col = int(l2 * grid_size)
    matrix[row, col] = bcc.p_values[i, 0]

# Build valid thresholds mask
valid_matrix = np.zeros((grid_size, grid_size), dtype=int)
for l1, l2 in bcc.valid_predict_params:
    row = int(l1 * grid_size)
    col = int(l2 * grid_size)
    valid_matrix[row, col] = 1

# Plot p-value matrix
fig, ax = plt.subplots(figsize=(7.5, 7.5))

colors = ["#cde2f9", "#96bfd7", "#3765a9"]
cmap = LinearSegmentedColormap.from_list("custom_blue", colors, gamma=0.5)
masked_matrix = np.ma.masked_invalid(matrix)
im = ax.imshow(masked_matrix, cmap=cmap, interpolation="nearest")

for i in range(grid_size):
    for j in range(grid_size):
        if np.isnan(matrix[i, j]):
            rect = patches.Rectangle(
                (j - 0.5, i - 0.5),
                1,
                1,
                hatch="///",
                facecolor="none",
                edgecolor="grey",
                linewidth=0,
            )
            ax.add_patch(rect)

# Add valid parameters area shape
for i in range(grid_size):
    for j in range(grid_size):
        if valid_matrix[i, j] == 1:
            neighbors = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
            if i - 1 < 0 or valid_matrix[i - 1, j] == 0:
                ax.plot([j - 0.5, j + 0.5], [i - 0.5, i - 0.5], color="#2ecc71", lw=3)
            if i + 1 >= grid_size or valid_matrix[i + 1, j] == 0:
                ax.plot([j - 0.5, j + 0.5], [i + 0.5, i + 0.5], color="#2ecc71", lw=3)
            if j - 1 < 0 or valid_matrix[i, j - 1] == 0:
                ax.plot([j - 0.5, j - 0.5], [i - 0.5, i + 0.5], color="#2ecc71", lw=3)
            if j + 1 >= grid_size or valid_matrix[i, j + 1] == 0:
                ax.plot([j + 0.5, j + 0.5], [i - 0.5, i + 0.5], color="#2ecc71", lw=3)

# Add best predict param as a star
best_l1, best_l2 = bcc.best_predict_param
ax.scatter(
    best_l2 * grid_size,
    best_l1 * grid_size,
    c="#2ecc71",
    marker="*",
    edgecolors="k",
    s=300,
    label="Best threshold pair",
)

# Set up axes, theme and legend
ax.set_xlabel(r"$\lambda_2$", fontsize=16)
ax.set_ylabel(r"$\lambda_1$", fontsize=16)
ax.set_title(
    "P-values per parameter pair\nwith valid parameter zone highlighted", fontsize=16
)
ax.set_xticks(range(grid_size))
ax.set_xticklabels(np.round(np.arange(grid_size) / grid_size, 2), fontsize=14)
ax.set_yticks(range(grid_size))
ax.set_yticklabels(np.round(np.arange(grid_size) / grid_size, 2), fontsize=14)

cbar = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.2, fraction=0.035)
cbar.set_label("P-value", fontsize=12)
legend_elements = [
    Patch(facecolor="none", edgecolor="grey", hatch="///", label="Non-explored zone"),
    Patch(facecolor="none", edgecolor="#2ecc71", label="Valid parameter zone", lw=2),
    Line2D(
        [0],
        [0],
        marker="*",
        color="w",
        label="Best threshold pair",
        markerfacecolor="#2ecc71",
        markeredgecolor="k",
        markersize=15,
    ),
]
ax.legend(
    handles=legend_elements,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.25),
    ncol=2,
    fontsize=12,
    frameon=False,
)
plt.tight_layout()
plt.show()
ax.set_ylabel(r"lambda_1")
ax.set_title("Valid parameters")
fig.tight_layout()
plt.show()
