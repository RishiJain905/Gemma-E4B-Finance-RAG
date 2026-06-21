import pytest


def pytest_addoption(parser):
    parser.addoption("--live", action="store_true", default=False,
                     help="run @pytest.mark.live tests (network / live model on :8087)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--live"):
        return
    markexpr = config.getoption("-m", default="") or ""
    if "live" in markexpr and "not live" not in markexpr:
        return  # user explicitly selected live via -m
    skip_live = pytest.mark.skip(reason="needs --live (or -m live)")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
