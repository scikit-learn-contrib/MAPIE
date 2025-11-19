# %%
import numpy as np
import pandas as pd

pd.set_option("display.width", 1000)


# Load HaluEval Question-Answering Data
url = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json"
df = pd.read_json(url, lines=True)

print(df.head())
# %%

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

# Create the 'hallucinated' flag based on the original column name
df["hallucinated"] = df["answer_type"] == "hallucinated_answer"

# Drop the helper column 'answer_type'
df = df.drop(columns=["answer_type"])

df = df.reset_index(drop=True)


# Generate biased scores using a beta distribution
def generate_biased_score(is_hallucinated):
    if is_hallucinated:
        return np.random.beta(a=5, b=1)
    else:
        return np.random.beta(a=1, b=5)


df["judge_score"] = df["hallucinated"].apply(generate_biased_score)

print(df.head())
# %%
