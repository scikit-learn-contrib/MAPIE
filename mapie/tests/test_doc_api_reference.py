from doc.hooks.api_reference import (
    api_nav,
    api_pages,
    on_config,
    public_symbols,
    render_api_page,
    render_overview,
)


def test_public_symbols_use_exports_and_exclude_internal_helpers() -> None:
    regression_names = {symbol.name for symbol in public_symbols("mapie.regression")}
    conditional_names = {
        symbol.name
        for symbol in public_symbols("mapie.conditional_conformal_prediction")
    }

    assert regression_names == {
        "ConformalizedQuantileRegressor",
        "CrossConformalRegressor",
        "CrossConformalizedQuantileRegressor",
        "JackknifeAfterBootstrapRegressor",
        "SplitConformalRegressor",
        "TimeSeriesRegressor",
    }
    assert conditional_names == {
        "ConditionalSplitConformalClassifier",
        "ConditionalSplitConformalRegressor",
    }


def test_api_pages_cover_every_discovered_symbol_once() -> None:
    pages = api_pages()
    paths = [
        symbol.path
        for page in pages
        for section in page.sections
        for symbol in section.symbols
    ]

    assert all(section.symbols for page in pages for section in page.sections)
    assert len(paths) == len(set(paths))
    assert "mapie.metrics.uncertainty.auroc" in paths
    assert "mapie.metrics.conditional.coverage_gap" in paths
    assert "mapie.exchangeability_testing.RiskMonitoring" in paths
    assert (
        "mapie.risk_control.adaptive_conformal_risk_control."
        "ConditionalExpectedRiskController"
    ) in paths


def test_generated_markdown_references_every_public_symbol() -> None:
    pages = api_pages()
    overview = render_overview(pages)

    assert '<div class="api-overview-table" markdown>' in overview

    for page in pages:
        detail = render_api_page(page)
        assert f"# {page.title}" in detail
        assert f"## [{page.title}]({page.slug}.md)" in overview
        for section in page.sections:
            for symbol in section.symbols:
                assert f"::: {symbol.path}" in detail
                assert f"{page.slug}.md#{symbol.path}" in overview


def test_api_navigation_is_generated_from_pages() -> None:
    pages = api_pages()
    expected_nav = api_nav(pages)
    config = {
        "nav": [
            {"Home": "index.md"},
            {"API Reference": []},
            {"Contributing": "contributing-docs.md"},
        ]
    }

    assert on_config(config) == config
    assert config["nav"][1] == {"API Reference": expected_nav}
