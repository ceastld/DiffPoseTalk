from .smplx.body_models import SMPLX
from .smplx_driver import SMPLXDriver


import os
file_dir_path = os.path.dirname(os.path.realpath(__file__))
smplx_model_path = os.path.join(file_dir_path, 'SMPLX2020')

smplx_dict_keys = [
    "body_pose",
    "lhand_pose",
    "rhand_pose",
    "jaw_pose",
    "expr",
    "cam_trans",
    "cam_angle",
    "img_ori",
    "seg_img",
    "shape",
    "cam_para",
    "parsing_img",
]
