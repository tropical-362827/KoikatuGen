import argparse
import json
from os.path import join
import numpy as np
import os.path
from glob import glob

parser = argparse.ArgumentParser("convert weight parameters from json to csv.", add_help=False)
parser.add_argument("folderpath", action="store", type=str, help="path to a parameter folder e.g. './vae_models/20210225_0619'")
args = parser.parse_args()

csv_folder = join(args.folderpath, "csv")
json_folder = join(args.folderpath, "weights")

if not os.path.exists(csv_folder):
    os.mkdir(csv_folder)

for weight_file in glob(join(json_folder, "*.json")):
    epoch = os.path.splitext(os.path.basename(weight_file))[0]
    param_folder = f"epoch_{str(epoch)}"

    if not os.path.exists(join(csv_folder, param_folder)):
        os.mkdir(join(csv_folder, param_folder))

    params = json.load(open(weight_file, "r"))

    weight = np.array(params["weights"])
    bias = np.array(params["bias"])

    np.savetxt(join(csv_folder, param_folder, f"weight.csv"), weight, fmt="%g", delimiter=", ")
    np.savetxt(join(csv_folder, param_folder, f"bias.csv"), bias[np.newaxis, :], fmt="%g", delimiter=", ")