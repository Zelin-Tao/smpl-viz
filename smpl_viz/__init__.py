from os import path
import os

AMASS_DATA_PATH = os.environ.get("AMASS_DATA_PATH", path.expanduser("~/.amass_data"))
SMPLH_PATH = path.join(AMASS_DATA_PATH, "body_models", "smplh")
PROJECT_PATH = path.dirname(path.dirname(__file__))

smpl_model_dict = {}

try:
    import numpy as np
    for gender in ["male", "female", "neutral"]:
        with np.load(path.join(SMPLH_PATH, gender, "model.npz"), allow_pickle=True) as data:
            smpl_model_dict[gender] = {key: data[key] for key in data.files}
except Exception as e:
    print(f"Warning: failed to load SMPLH models: {e}")
    print(f"Make sure body models are placed under: {AMASS_DATA_PATH}/body_models/smplh/")