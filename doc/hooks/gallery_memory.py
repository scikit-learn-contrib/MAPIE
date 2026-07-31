"""Keep gallery builds within the memory available on Read the Docs."""

import gc
import os
from functools import wraps

from mkdocs_gallery import gen_single


# Several examples use ``n_jobs=-1``.  Limit the worker pool before joblib is
# first used so that a documentation build cannot scale its memory use with the
# number of CPUs exposed by the build host.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")


def _release_example_globals(result):
    """Release objects retained by mkdocs-gallery after an example finishes."""
    run_vars = result.script.run_vars
    if run_vars is not None:
        if run_vars.example_globals is not None:
            run_vars.example_globals.clear()
            run_vars.example_globals = None
        run_vars.fake_main = None
    gc.collect()
    return result


if not getattr(gen_single.generate_file_md, "_mapie_releases_globals", False):
    _original_generate_file_md = gen_single.generate_file_md

    @wraps(_original_generate_file_md)
    def _generate_file_md_without_retained_globals(*args, **kwargs):
        result = _original_generate_file_md(*args, **kwargs)
        return _release_example_globals(result)

    _generate_file_md_without_retained_globals._mapie_releases_globals = True
    gen_single.generate_file_md = _generate_file_md_without_retained_globals
