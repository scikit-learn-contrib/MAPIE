"""MkDocs hook to fix mkdocs-gallery nav generation.

mkdocs-gallery generates nav entries like {"Gallery Title": []} for gallery
index pages, which render as broken links to the homepage. This hook removes
those empty section entries after the gallery plugin populates the nav.
"""
import logging

log = logging.getLogger("mkdocs.hooks.gallery_nav_fix")


def on_pre_build(config):
    """Fix gallery nav entries after the gallery plugin has populated them."""
    if "nav" not in config or config["nav"] is None:
        return

    _fix_empty_gallery_sections(config["nav"])


def _fix_empty_gallery_sections(nav):
    """Walk nav and remove empty section entries."""
    for i, item in enumerate(nav):
        if isinstance(item, dict):
            for key, value in item.items():
                if isinstance(value, list) and value:
                    cleaned = [v for v in value if not _is_empty_section(v)]
                    if len(cleaned) != len(value):
                        log.info("Removed empty gallery section(s) from '%s'", key)
                    nav[i] = {key: cleaned}
                    _fix_empty_gallery_sections(cleaned)


def _is_empty_section(item):
    """Check if a nav item is an empty section like {"Title": []}."""
    if isinstance(item, dict) and len(item) == 1:
        value = list(item.values())[0]
        return isinstance(value, list) and len(value) == 0
    return False
