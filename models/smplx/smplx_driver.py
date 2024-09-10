import os
import tempfile
import warnings
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path
from psbody.mesh import Mesh
from pytorch3d.transforms import so3_exp_map
from . import SMPLX
from data.utils import coef_to_smplx_dict, smplx_dict_to_coef, smplx_dict_to_tensor
from render_utils.render_nvdiff import MeshRenderer_Cuda
from utils.media import convert_video
from utils.renderer import MeshRenderer
from models.flame import FLAMEConfig
from . import smplx_model_path, smplx_dict_keys
from pytorch3d.io import load_obj
from ..adnerf_rendering import (
    adnerf_rendering_part_ids_path,
    adnerf_teeth_mesh_file,
    adnerf_gaussian_info_file,
    template_mesh_file,
    template_noteeth_mesh_file,
)

warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")


class SMPLXDataset(Dataset):
    def __init__(self, smplx_dict):
        self.smplx_dict = smplx_dict
        self.track_save_keys = ["body_pose", "lhand_pose", "rhand_pose", "jaw_pose", "expr", "cam_trans", "cam_angle"]
        assert all(key in smplx_dict for key in self.track_save_keys), "All keys must be present in smplx_dict."
        self.data_length = self.smplx_dict[self.track_save_keys[0]].shape[0]

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        ret_dict = {key: self.smplx_dict[key][idx] for key in self.track_save_keys}
        ret_dict["shape"] = self.smplx_dict["shape"][0]
        ret_dict["cam_para"] = self.smplx_dict["cam_para"][0]
        return ret_dict


class SMPLXDriver:
    def __init__(self, fps=25, size=(480, 480)) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mesh_renderer = MeshRenderer_Cuda()
        shape_dim, expr_dim = FLAMEConfig.n_shape, FLAMEConfig.n_exp
        self.smplx_model = SMPLX(smplx_model_path, num_expression_coeffs=expr_dim, num_betas=shape_dim, use_pca=False).to(self.device).eval()
        # fmt:off
        self.dict_keys = [
            "body_pose","lhand_pose","rhand_pose",
            "jaw_pose","expr",
            "cam_trans","cam_angle",
            "img_ori","seg_img",
            "shape","cam_para","parsing_img",
        ]
        # fmt:on
        self.adnerf_rendering_part_ids = np.loadtxt(adnerf_rendering_part_ids_path, dtype=np.int64)
        if os.path.isfile(template_mesh_file):
            # _, faces, _ = load_obj(template_mesh_file)
            _, faces, _ = load_obj(template_noteeth_mesh_file)
        else:
            _, faces, _ = load_obj(adnerf_teeth_mesh_file)
        self.tris = faces.verts_idx[None, ...].to(self.device).int()
        self.size = size
        self.fps = fps
        pass

    def driving_folder(self, data_dir, out_path):
        smplx_dict = torch.load(f"{data_dir}/body_track/smplx_track.pth")
        self.driving(smplx_dict, out_path)

    def driving_front(self, smplx_dict, out_path):
        smplx_dict = coef_to_smplx_dict(smplx_dict_to_coef(smplx_dict))
        # smplx_dict = smplx_dict_to_tensor(smplx_dict, device=self.device)
        save_dir = Path(out_path).parent
        os.makedirs(save_dir, exist_ok=True)
        tmp_video_file = tempfile.NamedTemporaryFile("w", suffix=".mp4", dir=save_dir)
        writer = cv2.VideoWriter(tmp_video_file.name, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self.size)
        cam_para = torch.tensor([1600, 1600, 240, 240], dtype=torch.float32, device=self.device).unsqueeze(0)
        verts_list = []
        for i in tqdm(range(smplx_dict["body_pose"].shape[0]), "SMPLX"):
            ret_dict = {k: torch.tensor(v[i : i + 1], dtype=torch.float32, device=self.device) for k, v in smplx_dict.items()}
            batch_size = ret_dict["body_pose"].shape[0]
            smplx_out = self.smplx_model.forward(
                betas=ret_dict["shape"],
                body_pose=ret_dict["body_pose"],
                jaw_pose=ret_dict["jaw_pose"],
                expression=ret_dict["expr"],
            )
            verts: torch.Tensor = smplx_out.vertices[:, self.adnerf_rendering_part_ids]  # (b, 5705, 3)
            verts_list.append(verts)

        center = torch.cat(verts_list, dim=0).mean(dim=(0, 1), keepdim=True) + torch.tensor([0, 0.05, 1], device=self.device).float().expand(1, 1, -1)
        print(center)

        for verts in tqdm(verts_list, "Render"):
            render_vis = self.mesh_renderer.forward_visualization_geo(verts - center, self.tris, cam_para=cam_para, img_size=self.size)
            writer.write(cv2.cvtColor(render_vis[0], cv2.COLOR_RGB2BGR))  # 将帧写入视频

        writer.release()
        convert_video(tmp_video_file.name, out_path)
        tmp_video_file.close()

    def driving(self, smplx_dict, out_path):
        # 使用 SMPLXDataset 创建数据集
        smplx_dataset = SMPLXDataset(smplx_dict)
        # batch_size 只能用 1...
        folder_loader = DataLoader(smplx_dataset, batch_size=1, shuffle=False, num_workers=5, pin_memory=True, drop_last=False)
        save_dir = Path(out_path).parent
        os.makedirs(save_dir, exist_ok=True)
        tmp_video_file = tempfile.NamedTemporaryFile("w", suffix=".mp4", dir=save_dir)
        writer = cv2.VideoWriter(tmp_video_file.name, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self.size)

        for ret_dict in tqdm(folder_loader, desc="driving"):
            # batch_size = ret_dict["expr"].shape[0]
            for key in ret_dict.keys():
                ret_dict[key] = ret_dict[key].to(self.device)

            batch_size = ret_dict["body_pose"].shape[0]
            smplx_out = self.smplx_model.forward(
                betas=ret_dict["shape"],
                body_pose=ret_dict["body_pose"],
                # left_hand_pose=ret_dict["lhand_pose"],
                # right_hand_pose=ret_dict["rhand_pose"],
                jaw_pose=ret_dict["jaw_pose"],
                expression=ret_dict["expr"],
            )

            rots = so3_exp_map(ret_dict["cam_angle"])
            # verts_cam = torch.bmm(smplx_out.vertices[:, self.adnerf_rendering_part_ids], rots.permute(0, 2, 1)) + ret_dict["cam_trans"].unsqueeze(1)
            verts_cam = smplx_out.vertices[:, self.adnerf_rendering_part_ids] + ret_dict["cam_trans"].unsqueeze(1)
            for i in range(0, batch_size):
                render_vis = self.mesh_renderer.forward_visualization_geo(verts_cam[i : i + 1], self.tris, ret_dict["cam_para"][i : i + 1], self.size)
                # writer.write_frame(render_vis[0])
                writer.write(cv2.cvtColor(render_vis[0], cv2.COLOR_RGB2BGR))  # 将帧写入视频

        writer.release()
        convert_video(tmp_video_file.name, out_path)
        tmp_video_file.close()
