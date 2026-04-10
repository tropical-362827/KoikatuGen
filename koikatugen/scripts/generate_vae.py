import argparse
import os

import pandas as pd
import torch
from kkloader.KoikatuCharaData import KoikatuCharaData

from koikatugen.dataset.transforms import dataframe_to_kkchara
from koikatugen.models.vae import VAE
from koikatugen.scripts.template_utils import (
    load_training_metadata,
    resolve_template_path,
)


def generate_vae(
    feature_columns: list[str], latent_dim: int, args: argparse.Namespace, outdir: str
):
    model = VAE(len(feature_columns), latent_dim)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    )
    model.eval()

    y = model.sample(args.n).numpy()

    kc_df = pd.DataFrame(y, columns=feature_columns)
    template_path = resolve_template_path(args.checkpoint)
    for i, row in kc_df.iterrows():
        kc = dataframe_to_kkchara(row, KoikatuCharaData.load(template_path))
        kc["Parameter"]["lastname"] = f"{i:03d}"
        kc["Parameter"]["firstname"] = ""
        kc.save(os.path.join(outdir, f"{i:03d}.png"))

    print(f"Generated {args.n} characters to {outdir} using {template_path}")


def main():
    parser = argparse.ArgumentParser(description="KoikatuGen VAE generator")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument(
        "--n", type=int, default=10, help="Number of characters to generate"
    )
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()
    metadata = load_training_metadata(args.checkpoint)
    if metadata is None:
        raise ValueError(
            "Could not load training metadata from the checkpoint directory."
        )
    feature_columns = metadata["feature_columns"]
    latent_dim = metadata["latent_dim"]

    if args.outdir is None:
        run_name = os.path.basename(os.path.dirname(args.checkpoint))
        args.outdir = os.path.join("outputs", "vae", run_name, "generated")
    os.makedirs(args.outdir, exist_ok=True)

    generate_vae(feature_columns, latent_dim, args, args.outdir)


if __name__ == "__main__":
    main()
