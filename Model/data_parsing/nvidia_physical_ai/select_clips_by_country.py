#!/usr/bin/env python3
"""Select NVIDIA PhysicalAI-AV clip UUIDs by country (geographic training subset).

The full dataset is ~100 TB. `download_dataset.py` already downloads a subset by
`--clips N` (random) or `--clip-uuids`. This adds the missing selection step:
pick the clips for a given country from `metadata/data_collection.parquet`, so you
can train/eval on a realistic geographic slice on modest hardware. It does NOT
download anything itself — it reuses `download_dataset.py`:

    uuids=$(python select_clips_by_country.py --country Spain --limit 20)
    python download_dataset.py --out ./nvidia_av_data --clip-uuids $uuids

The country matching is a case-insensitive substring on the dataset's own
`country` column, so the exact query depends on how NVIDIA labels it (e.g.
"Spain"); pass whatever the column uses.
"""

from __future__ import annotations

import argparse

import pandas as pd

_REPO_ID = "nvidia/PhysicalAI-Autonomous-Vehicles"
_METADATA_FILE = "metadata/data_collection.parquet"


def select_clip_uuids_by_country(data_collection: pd.DataFrame, country: str) -> list[str]:
    """Clip UUIDs whose ``country`` matches ``country`` (case-insensitive substring).

    Args:
        data_collection: the dataset's ``metadata/data_collection.parquet`` as a
            DataFrame indexed by clip UUID, with a ``country`` column.
        country: substring to match (case-insensitive), e.g. ``"Spain"``.

    Returns:
        List of clip UUIDs (strings) for the matching country, in file order.

    Raises:
        KeyError: if ``data_collection`` has no ``country`` column.
    """
    if "country" not in data_collection.columns:
        raise KeyError("data_collection must have a 'country' column; "
                       f"got {list(data_collection.columns)}")
    mask = data_collection["country"].astype(str).str.contains(
        country, case=False, na=False)
    return data_collection.index[mask].astype(str).tolist()


def _load_data_collection(cache_dir: str) -> pd.DataFrame:
    """Download (if needed) and read the dataset's country metadata parquet."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=_REPO_ID, filename=_METADATA_FILE,
                           repo_type="dataset", local_dir=cache_dir)
    return pd.read_parquet(path)


def parse_args():
    p = argparse.ArgumentParser(
        description="List NVIDIA PhysicalAI-AV clip UUIDs for a country "
                    "(feed into download_dataset.py --clip-uuids)")
    p.add_argument("--country", required=True,
                   help="Country substring to match (case-insensitive), e.g. Spain")
    p.add_argument("--cache-dir", default=".hf_cache",
                   help="Where to cache the downloaded metadata parquet")
    p.add_argument("--limit", type=int, default=None,
                   help="Keep at most N clips (default: all matches)")
    return p.parse_args()


def main():
    args = parse_args()
    df = _load_data_collection(args.cache_dir)
    uuids = select_clip_uuids_by_country(df, args.country)
    if args.limit is not None:
        uuids = uuids[:args.limit]
    print(" ".join(uuids))   # space-separated → download_dataset.py --clip-uuids


if __name__ == "__main__":
    main()
