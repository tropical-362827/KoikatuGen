import argparse
import datetime
import os
import pickle

import pandas as pd
from ctgan import CTGAN

from koikatugen.dataset.schema import categorical_keys
from koikatugen.dataset.transforms import clip_continuous_columns
from koikatugen.scripts.template_utils import save_training_metadata
from koikatugen.scripts.training_utils import filter_by_ranking_ratio


def train_ctgan(df: pd.DataFrame, args: argparse.Namespace, outdir: str):
    model = CTGAN(
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=True,
        enable_gpu=not args.cpu_only,
    )
    model.fit(df, discrete_columns=categorical_keys)

    checkpoint_path = os.path.join(outdir, "model.pkl")
    with open(checkpoint_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Checkpoint saved to {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description="KoikatuGen CTGAN trainer")
    parser.add_argument("--parquet", default="./data/preprocessed/kks_charas.parquet")
    parser.add_argument(
        "--ranking-parquet",
        default="./data/preprocessed/kks_stat_20230103.parquet",
    )
    parser.add_argument(
        "--ratio-top-k",
        type=int,
        default=10000,
        help="Keep rows whose base ids are in the top-K good/download_1 ranking",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet)
    if args.ratio_top_k > 0:
        df = filter_by_ranking_ratio(df, args.ranking_parquet, args.ratio_top_k)

    df = clip_continuous_columns(df)

    t = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    outdir = os.path.join("outputs", "ctgan", t)
    os.makedirs(outdir, exist_ok=True)
    save_training_metadata(outdir, args.parquet)

    train_ctgan(df, args, outdir)


if __name__ == "__main__":
    main()
