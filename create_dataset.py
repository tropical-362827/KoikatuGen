#/usr/bin/env python
# -*- coding:utf-8 -*-

from KoikatuCharaLoader import KoikatuCharaData
from KoikatuWebAPI import KoikatuWebAPI
import pandas as pd
import numpy as np
import os
import sys
from glob import glob

vector_keys = [
    "face_shapeValueFace",
    "face_eyebrowColor",
    "face_hlUpColor",
    "face_hlDownColor",
    "face_whiteBaseColor",
    "face_whiteSubColor",
    "face_eyelineColor",
    "face_moleColor",
    "face_moleLayout",
    "face_lipLineColor",
    "body_shapeValueBody",
    "body_skinMainColor",
    "body_skinSubColor",
    "body_sunburnColor",
    "body_nipColor",
    "body_underhairColor",
    "body_nailColor",
    "face_pupil_0_baseColor",
    "face_pupil_0_subColor",
    "face_pupil_1_baseColor",
    "face_pupil_1_subColor",
    "face_baseMakeup_eyeshadowColor",
    "face_baseMakeup_cheekColor",
    "face_baseMakeup_lipColor",
    "face_baseMakeup_paintColor_0",
    "face_baseMakeup_paintColor_1",
    "face_baseMakeup_paintLayout_0",
    "face_baseMakeup_paintLayout_1",
    "body_paintColor_0",
    "body_paintColor_1",
    "body_paintLayout_0",
    "body_paintLayout_1",
    "hair_parts_0_baseColor",
    "hair_parts_0_startColor",
    "hair_parts_0_endColor",
    "hair_parts_0_outlineColor",
    "hair_parts_1_baseColor",
    "hair_parts_1_startColor",
    "hair_parts_1_endColor",
    "hair_parts_1_outlineColor",
    "hair_parts_2_baseColor",
    "hair_parts_2_startColor",
    "hair_parts_2_endColor",
    "hair_parts_2_outlineColor",
    "hair_parts_3_baseColor",
    "hair_parts_3_startColor",
    "hair_parts_3_endColor",
    "hair_parts_3_outlineColor"
]

scalar_keys = [
    "face_detailPower",
    "face_cheekGlossPower",
    "face_pupilWidth",
    "face_pupilHeight",
    "face_pupilX",
    "face_pupilY",
    "face_eyelineUpWeight",
    "face_lipGlossPower",
    "body_bustSoftness",
    "body_bustWeight",
    "body_detailPower",
    "body_skinGlossPower",
    "body_nipGlossPower",
    "body_areolaSize",
    "body_nailGlossPower",
    "face_pupil_0_gradBlend",
    "face_pupil_0_gradOffsetY",
    "face_pupil_0_gradScale",
    "face_pupil_1_gradBlend",
    "face_pupil_1_gradOffsetY",
    "face_pupil_1_gradScale",
    "hair_parts_0_length",
    "hair_parts_1_length",
    "hair_parts_2_length",
    "hair_parts_3_length"
]

categorical_keys = [
    "face_detailId",
    "face_eyebrowId",
    "face_noseId",
    "face_hlUpId",
    "face_hlDownId",
    "face_whiteId",
    "face_eyelineUpId",
    "face_eyelineDownId",
    "face_moleId",
    "face_lipLineId",
    "face_foregroundEyes",
    "face_foregroundEyebrow",
    "body_skinId",
    "body_detailId",
    "body_sunburnId",
    "body_nipId",
    "body_underhairId",
    "hair_glossId",
    "face_pupil_0_id",
    "face_pupil_0_gradMaskId",
    "face_pupil_1_id",
    "face_pupil_1_gradMaskId",
    "face_baseMakeup_eyeshadowId",
    "face_baseMakeup_cheekId",
    "face_baseMakeup_lipId",
    "face_baseMakeup_paintId_0",
    "face_baseMakeup_paintId_1",
    "body_paintId_0",
    "body_paintId_1",
    "body_paintLayoutId_0",
    "body_paintLayoutId_1",
    "hair_parts_0_id",
    "hair_parts_1_id",
    "hair_parts_2_id",
    "hair_parts_3_id"
]

categories = [
    [0, 1, 2, 3, 4, 5],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 200, 201],
    [0, 1, 2, 3, 4, 200, 201],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 200, 201, 202, 203, 206, 207, 208, 209, 210, 211],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 200, 201, 202, 203, 205, 206, 207, 208],
    [0, 1, 2, 3, 4],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 200, 201, 202, 203, 204, 205, 206, 207, 208],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    [0, 1, 2, 3],
    [0, 1, 2, 3, 4, 5, 6],
    [0, 1, 2],
    [0, 1, 2],
    [0],
    [0, 1, 2, 3],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [0, 1, 2, 3, 4, 5, 6],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    [0, 1, 2, 3, 4, 5, 6, 7, 8],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 214, 216, 217],
    [0, 1, 2, 3],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 214, 216, 217],
    [0, 1, 2, 3],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [0, 1, 2, 3, 4, 5, 6, 7, 8],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 200, 201, 202],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 200, 202],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 201, 202],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 200, 201, 202],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 200, 201, 202, 204, 206, 207, 208, 209],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 28, 29, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209],
    [0, 1, 2, 3, 4, 6, 7],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 200]
]

def is_int(v):
    try:
        int(v)
        return True
    except:
        return False

def kkchara_to_vector(kc):
    c = {}
    for k in vector_keys + scalar_keys + categorical_keys:
        keys = list(map(lambda x: int(x) if is_int(x) else x, k.split("_")))
        if len(keys) == 2:
            c[k] = kc.custom[keys[0]][keys[1]]
        elif len(keys) == 3:
            c[k] = kc.custom[keys[0]][keys[1]][keys[2]]
        elif len(keys) == 4:
            c[k] = kc.custom[keys[0]][keys[1]][keys[2]][keys[3]]
    return c

def category_to_onehot(df):
    for k,c in zip(categorical_keys, categories):
        df[k] = pd.Categorical(df[k], categories=c)
    df = pd.get_dummies(df, columns=categorical_keys)
    return df

def dataframe_to_kkchara(df, kc_origin):
    def values_apply(key, value):
        keys = list(map(lambda x: int(x) if is_int(x) else x, key.split("_")))
        if len(keys) == 2:
            kc_origin.custom[keys[0]][keys[1]] = value
        elif len(keys) == 3:
            kc_origin.custom[keys[0]][keys[1]][keys[2]] = value
        elif len(keys) == 4:
            kc_origin.custom[keys[0]][keys[1]][keys[2]][keys[3]] = value

    for s in scalar_keys:
        values_apply(s, df[s].tolist())
    
    for v in vector_keys:
        element_keys = df.index[df.index.str.startswith(v)]
        elements = df[element_keys].values.tolist()
        values_apply(v, elements)
    
    for c in categorical_keys:
        element_keys = df.index[df.index.str.startswith(c)]
        max_element_key = df[element_keys].idxmax()
        id = int(max_element_key.split("_")[-1])
        values_apply(c, id)
    
    return kc_origin

def get_dataframe(kcv, ids=None):
    df = pd.DataFrame(index=ids)
    for s in scalar_keys:
        if not isinstance(kcv[s], list):
            df[s] = [kcv[s]]
        else:
            df[s] = kcv[s]
    
    for v in vector_keys:
        mat = np.array(kcv[v])
        if mat.ndim == 1:
            mat = mat[np.newaxis, :]
        columns = len(mat[0])
        for i in range(columns):
            k = "_".join([v,str(i)])
            df[k] = mat[:,i]

    for c in categorical_keys:
        if not isinstance(kcv[c], list):
            df[c] = [kcv[c]]
        else:
            df[c] = kcv[c]
    return df

def make_dataset():
    ranking = KoikatuWebAPI.get_ranking()

    a = {}
    for i in vector_keys + categorical_keys + scalar_keys:
        a[i] = []

    ids = []
    for filepath in sorted(glob("./kk_chara/*.png")):
        id = int(os.path.splitext(os.path.basename(filepath))[0])

        if not id in ranking["id"].values:
            print("\r\n{} was skipped because that data was deleted from official uploader.".format(filepath))
            continue

        sys.stdout.write("\r"+filepath)
        try:
            kc = KoikatuCharaData.load(filepath)
        except AssertionError:
            print("\r\n{} isn't vaild character data.".format(filepath))
            continue
        except ValueError:
            print("\r\n{} has extra blockdata.".format(filepath))
            continue
        
        if kc.parameter["sex"] != 1:
            print("\r\n{} was skipped. this character is male.".format(filepath))
            continue

        ids.append(id)

        c = kkchara_to_vector(kc)
        for k in vector_keys + categorical_keys + scalar_keys:
            a[k].append(c[k])
    
    df = get_dataframe(a, ids)

    df.to_csv("./kk_charas.csv")

if __name__ == '__main__':
    make_dataset()