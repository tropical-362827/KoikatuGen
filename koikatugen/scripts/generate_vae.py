import argparse
import os

import pandas as pd
import torch
from kkloader.KoikatuCharaData import KoikatuCharaData

from koikatugen.dataset.transforms import category_to_onehot, dataframe_to_kkchara
from koikatugen.models.vae import VAE


def generate_vae(df: pd.DataFrame, args: argparse.Namespace, outdir: str):
    model = VAE(df.shape[1], args.latent_dim)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    )
    model.eval()

    y = model.sample(args.n).numpy()

    kc_df = pd.DataFrame(y, columns=df.columns)
    for i, row in kc_df.iterrows():
        kc = dataframe_to_kkchara(
            row, KoikatuCharaData.load("./data/templates/default.png")
        )
        kc["Parameter"]["lastname"] = f"{i:03d}"
        kc["Parameter"]["firstname"] = ""
        kc.save(os.path.join(outdir, f"{i:03d}.png"))

    print(f"Generated {args.n} characters to {outdir}")


def main():
    parser = argparse.ArgumentParser(description="KoikatuGen VAE generator")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--parquet", default="./data/preprocessed/kk_charas.parquet")
    parser.add_argument("--latent-dim", type=int, default=800)
    parser.add_argument(
        "--n", type=int, default=10, help="Number of characters to generate"
    )
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet)
    df = category_to_onehot(df)

    if args.outdir is None:
        run_name = os.path.basename(os.path.dirname(args.checkpoint))
        args.outdir = os.path.join("outputs", "vae", run_name, "generated")
    os.makedirs(args.outdir, exist_ok=True)

    generate_vae(df, args, args.outdir)


if __name__ == "__main__":
    main()
