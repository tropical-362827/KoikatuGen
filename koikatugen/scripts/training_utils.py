import pandas as pd


def filter_by_ranking_ratio(
    df: pd.DataFrame, ranking_path: str, top_k: int
) -> pd.DataFrame:
    ranking = pd.read_parquet(ranking_path).copy()
    ranking["id"] = pd.to_numeric(ranking["id"], errors="coerce")
    ranking["download_1"] = pd.to_numeric(ranking["download_1"], errors="coerce")
    ranking["good"] = pd.to_numeric(ranking["good"], errors="coerce")

    ranking = ranking[(ranking["id"].notna()) & (ranking["download_1"] > 0)].copy()
    ranking["ratio"] = ranking["good"].fillna(0) / ranking["download_1"]
    ranking = ranking.sort_values(
        ["ratio", "good", "download_1"], ascending=False
    ).head(top_k)

    top_ids = set(ranking["id"].astype(int))
    base_ids = df.index.to_series().astype(str).str.extract(r"^(\d+)_")[0].astype(int)
    filtered = df.loc[base_ids.isin(top_ids)].copy()

    print(
        f"Ranking cutoff kept {len(top_ids)} ids and {len(filtered)} / {len(df)} rows"
    )

    return filtered
