import argparse
import datetime
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from koikatugen.dataset.transforms import category_to_onehot
from koikatugen.models.vae import VAE
from koikatugen.scripts.template_utils import save_training_metadata
from koikatugen.scripts.training_utils import filter_by_ranking_ratio


def train_vae(df: pd.DataFrame, args: argparse.Namespace, outdir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = torch.tensor(df.values.astype("float32")).to(device)
    loader = DataLoader(TensorDataset(data), batch_size=args.batch_size, shuffle=True)

    model = VAE(data.shape[1], args.latent_dim).to(device)
    optimizer = torch.optim.Adamax(model.parameters())

    pbar = tqdm(range(1, args.epochs + 1), desc="train")
    for epoch in pbar:
        model.train()
        total_loss = 0.0
        for (x,) in loader:
            optimizer.zero_grad()
            x_recon, mean, logvar = model(x)
            loss = model.loss(x, x_recon, mean, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        pbar.set_postfix(loss=f"{avg_loss:.6f}")

        if epoch % 5 == 0 or epoch == args.epochs:
            torch.save(
                model.state_dict(), os.path.join(outdir, f"epoch_{epoch:03d}.pth")
            )

    print(f"Checkpoints saved to {outdir}")


def main():
    parser = argparse.ArgumentParser(description="KoikatuGen VAE trainer")
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
    parser.add_argument("--latent-dim", type=int, default=800)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=200)
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet)
    if args.ratio_top_k > 0:
        df = filter_by_ranking_ratio(df, args.ranking_parquet, args.ratio_top_k)
    df = category_to_onehot(df)
    df = df.clip(0, 1)

    t = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    outdir = os.path.join("outputs", "vae", t)
    os.makedirs(outdir, exist_ok=True)
    save_training_metadata(
        outdir,
        args.parquet,
        extra_metadata={
            "latent_dim": args.latent_dim,
            "feature_columns": df.columns.tolist(),
        },
    )

    train_vae(df, args, outdir)


if __name__ == "__main__":
    main()
