"""Generate the API reference from MAPIE's public Python objects."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from importlib import import_module
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Mapping,
    MutableMapping,
    Sequence,
)

if TYPE_CHECKING:
    from mkdocs.structure.files import Files


@dataclass(frozen=True)
class ApiSymbol:
    """A public Python object rendered by mkdocstrings."""

    name: str
    path: str
    value: Any


@dataclass(frozen=True)
class ApiSection:
    """A group of public objects on an API page."""

    title: str
    symbols: tuple[ApiSymbol, ...]


@dataclass(frozen=True)
class ApiPage:
    """Metadata and public objects for one generated API page."""

    slug: str
    title: str
    description: str
    sections: tuple[ApiSection, ...]
    related: str = ""


INTERNAL_SYMBOLS = {
    "mapie.conditional_conformal_prediction": {
        "binary_search",
        "finish_dual_setup",
        "setup_cvx_problem",
    },
    "mapie.utils": {
        "NotFittedError",
        "check_is_fitted",
        "check_proba_normalized",
        "check_sklearn_user_model_is_fitted",
        "check_valid_ltt_params_index",
    },
}

METRIC_SECTION_TITLES = {
    "calibration": "Calibration",
    "classification": "Classification",
    "conditional": "Conditional Coverage",
    "regression": "Regression",
    "uncertainty": "Uncertainty",
}

METRIC_SECTION_ORDER = (
    "regression",
    "classification",
    "conditional",
    "uncertainty",
    "calibration",
)

OPTIONAL_METRIC_MODULES = ("conditional",)


def public_symbols(module_name: str) -> tuple[ApiSymbol, ...]:
    """Return the documented public objects of a module.

    ``__all__`` is authoritative when present. Otherwise, public classes and
    functions defined in the module are discovered automatically.
    """
    module = import_module(module_name)
    exported_names = getattr(module, "__all__", None)
    lazy_imports = getattr(module, "_LAZY_IMPORTS", {})

    if exported_names is None:
        excluded = INTERNAL_SYMBOLS.get(module_name, set())
        exported_names = [
            name
            for name, value in vars(module).items()
            if not name.startswith("_")
            and name not in excluded
            and (inspect.isclass(value) or inspect.isfunction(value))
            and getattr(value, "__module__", None) == module_name
        ]

    return tuple(
        ApiSymbol(
            name,
            f"{lazy_imports.get(name, module_name)}.{name}",
            getattr(module, name),
        )
        for name in exported_names
    )


def _partition_symbols(
    symbols: Sequence[ApiSymbol],
    groups: Sequence[tuple[str, Callable[[ApiSymbol], bool]]],
) -> tuple[ApiSection, ...]:
    """Partition symbols into named sections, retaining an automatic fallback."""
    remaining = list(symbols)
    sections = []
    for title, predicate in groups:
        selected = [symbol for symbol in remaining if predicate(symbol)]
        if selected:
            sections.append(ApiSection(title, tuple(selected)))
            remaining = [symbol for symbol in remaining if symbol not in selected]
    if remaining:
        sections.append(ApiSection("Other", tuple(remaining)))
    return tuple(sections)


def _origin_contains(*parts: str) -> Callable[[ApiSymbol], bool]:
    """Return a predicate matching fragments of an object's source module."""

    def predicate(symbol: ApiSymbol) -> bool:
        origin = getattr(symbol.value, "__module__", "")
        return any(part in origin for part in parts)

    return predicate


def _origin_endswith(*suffixes: str) -> Callable[[ApiSymbol], bool]:
    """Return a predicate matching the end of an object's source module."""

    def predicate(symbol: ApiSymbol) -> bool:
        origin = getattr(symbol.value, "__module__", "")
        return any(origin.endswith(suffix) for suffix in suffixes)

    return predicate


def _metric_sections() -> tuple[ApiSection, ...]:
    """Build metric sections from submodules exported by ``mapie.metrics``."""
    metrics = import_module("mapie.metrics")
    sections = []
    module_names = list(metrics.__all__)
    module_names.extend(
        name for name in OPTIONAL_METRIC_MODULES if name not in module_names
    )
    ordered_names = [name for name in METRIC_SECTION_ORDER if name in module_names]
    ordered_names.extend(name for name in module_names if name not in ordered_names)
    for module_basename in ordered_names:
        module_name = f"mapie.metrics.{module_basename}"
        title = METRIC_SECTION_TITLES.get(
            module_basename, module_basename.replace("_", " ").title()
        )
        sections.append(ApiSection(title, public_symbols(module_name)))
    return tuple(sections)


def api_pages() -> tuple[ApiPage, ...]:
    """Return the complete generated API reference specification."""
    conformity_symbols = public_symbols("mapie.conformity_scores")
    risk_symbols = public_symbols("mapie.risk_control")
    exchangeability_symbols = public_symbols("mapie.exchangeability_testing")

    return (
        ApiPage(
            "regression",
            "Regression",
            "Conformal prediction methods for regression tasks.",
            (ApiSection("Conformalizers", public_symbols("mapie.regression")),),
        ),
        ApiPage(
            "classification",
            "Classification",
            "Conformal prediction methods for classification tasks.",
            (ApiSection("Conformalizers", public_symbols("mapie.classification")),),
        ),
        ApiPage(
            "conditional-conformal-prediction",
            "Conditional Conformal Prediction",
            "Conformal prediction methods with conditional validity guarantees.",
            (
                ApiSection(
                    "Conformalizers",
                    public_symbols("mapie.conditional_conformal_prediction"),
                ),
            ),
            """
These estimators require the optional `conditional` dependency:

```bash
pip install "mapie[conditional]"
```

See [Theory](../content/conformal-prediction/conditional-guarantees.md) and the
runnable
[regression](../generated/regression/2-advanced-analysis/plot_conditional_conformal_regression_groups.md)
and
[classification](../generated/classification/2-advanced-analysis/plot_conditional_conformal_classification_groups.md)
examples.
""",
        ),
        ApiPage(
            "conformity-scores",
            "Conformity Scores",
            "Conformity score classes for regression and classification.",
            _partition_symbols(
                conformity_symbols,
                (
                    (
                        "Regression",
                        _origin_contains(
                            "conformity_scores.regression",
                            "conformity_scores.bounds",
                        ),
                    ),
                    (
                        "Classification",
                        _origin_contains(
                            "conformity_scores.classification",
                            "conformity_scores.sets",
                        ),
                    ),
                ),
            ),
        ),
        ApiPage(
            "metrics",
            "Metrics",
            (
                "Evaluation metrics for conformal prediction, calibration, "
                "and uncertainty estimation."
            ),
            _metric_sections(),
        ),
        ApiPage(
            "risk-control",
            "Risk Control",
            "Risk-controlling prediction methods.",
            _partition_symbols(
                risk_symbols,
                (
                    (
                        "Controllers",
                        _origin_endswith(
                            "binary_classification",
                            "adaptive_conformal_risk_control",
                            "multi_label_classification",
                            "semantic_segmentation",
                        ),
                    ),
                    ("Risks", _origin_contains(".risks")),
                    ("FWER Procedures", _origin_contains("fwer_control")),
                ),
            ),
        ),
        ApiPage(
            "calibration",
            "Calibration",
            "Calibration methods for probabilistic predictions.",
            (ApiSection("Calibrators", public_symbols("mapie.calibration")),),
        ),
        ApiPage(
            "exchangeability-testing",
            "Exchangeability Testing",
            "Tests and monitoring tools for assessing exchangeability.",
            _partition_symbols(
                exchangeability_symbols,
                (
                    (
                        "High-level Interfaces",
                        _origin_endswith("exchangeability", "risk_monitoring"),
                    ),
                    (
                        "Individual Tests",
                        _origin_endswith("martingales", "permutations"),
                    ),
                ),
            ),
            """
See [Exchangeability Tests](../content/exchangeability-testing/theory.md) for conceptual
background and guidance on selecting a test.
""",
        ),
        ApiPage(
            "utils",
            "Utilities",
            "Utility functions and resampling classes.",
            (
                ApiSection("Data Splitting", public_symbols("mapie.utils")),
                ApiSection("Resampling", public_symbols("mapie.subsample")),
            ),
        ),
    )


def _summary(symbol: ApiSymbol) -> str:
    """Return a table-safe one-line summary for a public object."""
    docstring = inspect.getdoc(symbol.value) or "API reference."
    first_paragraph = docstring.split("\n\n", maxsplit=1)[0]
    return " ".join(first_paragraph.split()).replace("|", r"\|")


def render_api_page(page: ApiPage) -> str:
    """Render one API detail page."""
    chunks = [f"# {page.title}", "", page.description]
    if page.related:
        chunks.extend(("", page.related.strip()))

    for section in page.sections:
        chunks.extend(("", f"## {section.title}", ""))
        for index, symbol in enumerate(section.symbols):
            if index:
                chunks.extend(("", "---", ""))
            chunks.extend(
                (
                    f"::: {symbol.path}",
                    "    options:",
                    "      heading_level: 3",
                )
            )
    return "\n".join(chunks) + "\n"


def render_overview(pages: Iterable[ApiPage]) -> str:
    """Render the API overview with links to every generated object."""
    chunks = [
        "# API Reference",
        "",
        "Complete API documentation for MAPIE.",
    ]
    for page in pages:
        chunks.extend(("", "---", "", f"## [{page.title}]({page.slug}.md)"))
        for section in page.sections:
            chunks.extend(
                (
                    "",
                    f"### {section.title}",
                    "",
                    '<div class="api-overview-table" markdown>',
                    "",
                    "| Item | Description |",
                    "|---|---|",
                )
            )
            for symbol in section.symbols:
                link = f"{page.slug}.md#{symbol.path}"
                chunks.append(f"| [`{symbol.name}`]({link}) | {_summary(symbol)} |")
            chunks.extend(("", "</div>"))
    return "\n".join(chunks) + "\n"


def api_nav(pages: Iterable[ApiPage]) -> list[Mapping[str, str]]:
    """Return the generated API navigation."""
    return [{"Overview": "api/index.md"}] + [
        {page.title: f"api/{page.slug}.md"} for page in pages
    ]


def _replace_api_nav(
    nav: Sequence[Any], generated_nav: list[Mapping[str, str]]
) -> None:
    """Replace the API Reference section in a raw MkDocs nav."""
    for item in nav:
        if isinstance(item, MutableMapping) and "API Reference" in item:
            item["API Reference"] = generated_nav
            return
    raise ValueError("mkdocs.yml must contain an 'API Reference' nav section.")


def on_config(config):
    """Generate the API navigation before MkDocs builds its page tree."""
    pages = api_pages()
    _replace_api_nav(config["nav"], api_nav(pages))
    return config


def on_files(files: Files, *, config) -> Files:
    """Add the generated API pages as virtual MkDocs files."""
    from mkdocs.structure.files import File

    pages = api_pages()
    generated = {"api/index.md": render_overview(pages)}
    generated.update({f"api/{page.slug}.md": render_api_page(page) for page in pages})
    for src_uri, content in generated.items():
        files.append(File.generated(config, src_uri, content=content))
    return files
