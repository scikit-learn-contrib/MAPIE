from sklearn.utils import deprecated

from mapie.conformity_scores.sets.utils import (
    get_true_label_position as get_true_label_position_new_path,
)


@deprecated(
    "WARNING: Deprecated path to import get_true_label_position. "
    "Please prefer the new path: "
    "[from mapie.conformity_scores.sets.utils import get_true_label_position]."
)
def get_true_label_position(*args, **kwargs):
    return get_true_label_position_new_path(*args, **kwargs)
