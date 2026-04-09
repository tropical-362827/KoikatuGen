import argparse
import os
import pickle

import pandas as pd
from kkloader.KoikatuCharaData import KoikatuCharaData

from koikatugen.dataset.transforms import clip_continuous_columns, dataframe_to_kkchara


def generate_ctgan(args: argparse.Namespace, outdir: str):
    with open(args.checkpoint, "rb") as f:
        model = pickle.load(f)

    sampled = model.sample(args.n)
    sampled = clip_continuous_columns(pd.DataFrame(sampled))

    for i, row in sampled.iterrows():
        kc = dataframe_to_kkchara(
            row, KoikatuCharaData.load("./data/templates/default.png")
        )
        kc["Parameter"]["lastname"] = f"{i:03d}"
        kc["Parameter"]["firstname"] = ""
        kc.save(os.path.join(outdir, f"{i:03d}.png"))

    print(f"Generated {args.n} characters to {outdir}")


def main():
    parser = argparse.ArgumentParser(description="KoikatuGen CTGAN generator")
    parser.add_argument("--checkpoint", required=True, help="Path to CTGAN pickle file")
    parser.add_argument(
        "--n", type=int, default=10, help="Number of characters to generate"
    )
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    if args.outdir is None:
        run_name = os.path.basename(os.path.dirname(args.checkpoint))
        args.outdir = os.path.join("outputs", "ctgan", run_name, "generated")
    os.makedirs(args.outdir, exist_ok=True)

    generate_ctgan(args, args.outdir)


if __name__ == "__main__":
    main()
