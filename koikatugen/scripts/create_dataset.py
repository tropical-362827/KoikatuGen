import argparse

from koikatugen.dataset.loader import make_dataset


def main():
    parser = argparse.ArgumentParser(description="KoikatuGen dataset creator")
    parser.add_argument("--chara-dir", default="./data/raw/kk_chara")
    parser.add_argument(
        "--out-parquet", default="./data/preprocessed/kk_charas.parquet"
    )
    args = parser.parse_args()

    make_dataset(args.chara_dir, args.out_parquet)


if __name__ == "__main__":
    main()
