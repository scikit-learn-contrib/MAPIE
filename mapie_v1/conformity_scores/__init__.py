from mapie.conformity_scores import (
    AbsoluteConformityScore,
    GammaConformityScore,
    ResidualNormalisedScore,
    LACConformityScore,
    TopKConformityScore,
    APSConformityScore,
    RAPSConformityScore,
    BaseRegressionScore,
    BaseClassificationScore,
)

CONFORMITY_SCORES_STRING_MAP = {
    BaseRegressionScore: {
        "absolute": AbsoluteConformityScore,
        "gamma": GammaConformityScore,
        "residual_normalized": ResidualNormalisedScore,
    },
    BaseClassificationScore: {
        "lac": LACConformityScore,
        "top_k": TopKConformityScore,
        "aps": APSConformityScore,
        "raps": RAPSConformityScore,
    },
}
