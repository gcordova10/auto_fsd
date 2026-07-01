"""Tests for the country clip selector (pure DataFrame logic, no HF, no SDK).

The `nvidia_physical_ai` package `__init__` eagerly imports the `physical_ai_av`
SDK (via `camera.py`), which isn't installed in CI — so we load the selector
module **directly from its file** to test the pure function without triggering
that import chain.
"""

import importlib.util
import pathlib

import pandas as pd
import pytest

_MODULE = (pathlib.Path(__file__).resolve().parents[1]
           / "data_parsing" / "nvidia_physical_ai" / "select_clips_by_country.py")
_spec = importlib.util.spec_from_file_location("select_clips_by_country", _MODULE)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
select_clip_uuids_by_country = _mod.select_clip_uuids_by_country


def _df(countries, uuids):
    return pd.DataFrame({"country": countries}, index=uuids)


def test_selects_matching_country_case_insensitive():
    df = _df(["Spain", "Germany", "spain", "USA"], ["a", "b", "c", "d"])
    assert select_clip_uuids_by_country(df, "Spain") == ["a", "c"]   # case-insensitive
    assert select_clip_uuids_by_country(df, "germany") == ["b"]


def test_substring_match():
    df = _df(["United States", "United Kingdom", "Spain"], ["a", "b", "c"])
    assert select_clip_uuids_by_country(df, "United") == ["a", "b"]


def test_no_match_returns_empty():
    df = _df(["Spain", "Germany"], ["a", "b"])
    assert select_clip_uuids_by_country(df, "France") == []


def test_uuids_returned_as_strings_in_order():
    df = _df(["Spain", "Spain", "Germany"], ["z1", "z2", "z3"])
    out = select_clip_uuids_by_country(df, "Spain")
    assert out == ["z1", "z2"] and all(isinstance(u, str) for u in out)


def test_missing_country_column_raises():
    df = pd.DataFrame({"city": ["Madrid"]}, index=["a"])
    with pytest.raises(KeyError, match="country"):
        select_clip_uuids_by_country(df, "Spain")
