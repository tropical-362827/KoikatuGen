import json
import os

import numpy as np
import pandas as pd
from kkloader.KoikatuCharaData import KoikatuCharaData
from tqdm import tqdm

from koikatugen.dataset.schema import categorical_keys, scalar_keys, vector_keys


def _is_int(v):
    try:
        int(v)
        return True
    except Exception:
        return False


def kkchara_to_vector(kc):
    c = {}
    for k in vector_keys + scalar_keys + categorical_keys:
        keys = list(map(lambda x: int(x) if _is_int(x) else x, k.split("_")))
        if len(keys) == 2:
            c[k] = kc["Custom"][keys[0]][keys[1]]
        elif len(keys) == 3:
            c[k] = kc["Custom"][keys[0]][keys[1]][keys[2]]
        elif len(keys) == 4:
            c[k] = kc["Custom"][keys[0]][keys[1]][keys[2]][keys[3]]
    return c


def get_dataframe(kcv, ids=None):
    cols: dict[str, list] = {}

    for s in scalar_keys:
        cols[s] = kcv[s] if isinstance(kcv[s], list) else [kcv[s]]

    for v in vector_keys:
        mat = np.array(kcv[v])
        if mat.ndim == 1:
            mat = mat[np.newaxis, :]
        for i in range(mat.shape[1]):
            cols[f"{v}_{i}"] = mat[:, i].tolist()

    for c in categorical_keys:
        cols[c] = kcv[c] if isinstance(kcv[c], list) else [kcv[c]]

    return pd.DataFrame(cols, index=ids)


def make_dataset(chara_dir: str, out_parquet: str):
    paths = sorted(p for p in os.listdir(chara_dir) if p.endswith(".png"))
    a = {k: [] for k in vector_keys + categorical_keys + scalar_keys}
    ids = []
    skipped: list[dict] = []

    for filename in (pbar := tqdm(paths)):
        filepath = os.path.join(chara_dir, filename)
        pbar.set_postfix_str(filename)
        try:
            kc = KoikatuCharaData.load(filepath)
        except AssertionError:
            skipped.append({"file": filename, "reason": "invalid character data"})
            continue
        except ValueError:
            skipped.append({"file": filename, "reason": "extra blockdata"})
            continue
        except TypeError:
            skipped.append({"file": filename, "reason": "unparseable blockdata"})
            continue

        if kc["Parameter"]["sex"] != 1:
            skipped.append({"file": filename, "reason": "male character"})
            continue

        ids.append(os.path.splitext(filename)[0])
        c = kkchara_to_vector(kc)
        for k in vector_keys + categorical_keys + scalar_keys:
            a[k].append(c[k])

    df = get_dataframe(a, ids)
    df.to_parquet(out_parquet)

    skip_json = os.path.splitext(out_parquet)[0] + "_skipped.json"
    with open(skip_json, "w") as f:
        json.dump(skipped, f, indent=2, ensure_ascii=False)

    reason_counts: dict[str, int] = {}
    for s in skipped:
        reason_counts[s["reason"]] = reason_counts.get(s["reason"], 0) + 1

    print(f"\nSaved {len(ids)} characters to {out_parquet}")
    print(f"Skipped {len(skipped)} files -> {skip_json}")
    for reason, count in reason_counts.items():
        print(f"  {reason}: {count}")
