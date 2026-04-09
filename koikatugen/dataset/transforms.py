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


def _coerce_category_value(value, allowed_values):
    try:
        numeric_value = int(value)
    except (TypeError, ValueError):
        if value in allowed_values:
            return value
        return allowed_values[0]

    if numeric_value in allowed_values:
        return numeric_value

    return min(allowed_values, key=lambda candidate: abs(candidate - numeric_value))


def clip_continuous_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for s in scalar_keys:
        if s in df.columns:
            df[s] = df[s].clip(0, 1)

    for v in vector_keys:
        element_keys = [c for c in df.columns if c.startswith(f"{v}_")]
        if element_keys:
            df[element_keys] = df[element_keys].clip(0, 1)

    return df


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
        if c in df.index:
            _apply(
                c, _coerce_category_value(df[c], categories[categorical_keys.index(c)])
            )
            continue

        element_keys = df.index[df.index.str.startswith(c)]
        probs = _softmax(df[element_keys].values)
        choice_key = np.random.choice(element_keys, p=probs)
        _apply(c, int(choice_key.split("_")[-1]))

    return kc_origin
