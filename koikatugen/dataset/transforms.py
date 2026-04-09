import numpy as np
import pandas as pd

from koikatugen.dataset.schema import (
    categories,
    categorical_keys,
    scalar_keys,
    vector_keys,
)


def _is_int(v):
    try:
        int(v)
        return True
    except Exception:
        return False


def _softmax(x):
    z = np.exp(x)
    return z / np.sum(z)


def category_to_onehot(df: pd.DataFrame) -> pd.DataFrame:
    for k, c in zip(categorical_keys, categories):
        df[k] = pd.Categorical(df[k], categories=c)
    return pd.get_dummies(df, columns=categorical_keys)


def dataframe_to_kkchara(df, kc_origin):
    def _apply(key, value):
        keys = list(map(lambda x: int(x) if _is_int(x) else x, key.split("_")))
        if len(keys) == 2:
            kc_origin["Custom"][keys[0]][keys[1]] = value
        elif len(keys) == 3:
            kc_origin["Custom"][keys[0]][keys[1]][keys[2]] = value
        elif len(keys) == 4:
            kc_origin["Custom"][keys[0]][keys[1]][keys[2]][keys[3]] = value

    for s in scalar_keys:
        _apply(s, df[s].tolist())

    for v in vector_keys:
        element_keys = df.index[df.index.str.startswith(v)]
        _apply(v, df[element_keys].values.tolist())

    for c in categorical_keys:
        element_keys = df.index[df.index.str.startswith(c)]
        probs = _softmax(df[element_keys].values)
        choice_key = np.random.choice(element_keys, p=probs)
        _apply(c, int(choice_key.split("_")[-1]))

    return kc_origin
