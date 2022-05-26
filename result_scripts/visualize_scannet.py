import os.path as osp
import numpy as np
import torch
from dkm.utils import *
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from dkm import DKM

imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).cuda()
imagenet_std = torch.tensor([0.229, 0.224, 0.225]).cuda()


def numpy_to_pil(x: np.ndarray):
    """
    Args:
        x: Assumed to be of shape (h,w,c)
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if x.max() <= 1.01:
        x *= 255
    x = x.astype(np.uint8)
    return Image.fromarray(x)


def tensor_to_pil(x, unnormalize=False):
    if unnormalize:
        x = x * imagenet_std[:, None, None] + imagenet_mean[:, None, None]
    x = x.detach().permute(1, 2, 0).cpu().numpy()
    x = np.clip(x, 0.0, 1.0)
    return numpy_to_pil(x)


class ScanNetVisualization:
    def __init__(self, data_root="data/scannet") -> None:
        self.data_root = data_root

    def visualize(self, model):
        model.train(False)
        with torch.no_grad():
            data_root = self.data_root
            tmp = np.load(osp.join(data_root, "test.npz"))
            pairs, rel_pose = tmp["name"], tmp["rel_pose"]
            pair_inds = np.random.choice(
                range(len(pairs)), size=len(pairs), replace=False
            )
            for pairind in tqdm(pair_inds, smoothing=0.9):
                scene = pairs[pairind]
                scene_name = f"scene0{scene[0]}_00"
                im1 = Image.open(
                    osp.join(
                        self.data_root,
                        "scans_test",
                        scene_name,
                        "color",
                        f"{scene[2]}.jpg",
                    )
                )
                im2 = Image.open(
                    osp.join(
                        self.data_root,
                        "scans_test",
                        scene_name,
                        "color",
                        f"{scene[3]}.jpg",
                    )
                )
                im1 = im1.resize((512, 384))
                im2 = im2.resize((512, 384))
                x2 = (torch.tensor(np.array(im2)) / 255).cuda().permute(2, 0, 1)
                dense_matches, dense_certainty = model.match(im1, im2)
                dense_certainty = dense_certainty ** (1 / 5)
                im2_transfer_rgb = F.grid_sample(
                    x2[None],
                    dense_matches[..., 2:][None],
                    mode="bicubic",
                    align_corners=False,
                )[0]
                white_im = torch.ones_like(x2)
                c_b = dense_certainty / dense_certainty.max()
                im1.save(f"vis/scannet/{scene[3]}_{scene[2]}_query.jpg")
                im2.save(f"vis/scannet/{scene[3]}_{scene[2]}_support.jpg")
                tensor_to_pil(
                    c_b * im2_transfer_rgb + (1 - c_b) * white_im, unnormalize=False
                ).save(f"vis/scannet/{scene[3]}_{scene[2]}_warped.jpg")


if __name__ == "__main__":
    dkm = DKM(pretrained=True, version="mega_synthetic")

    yfcc100m_vis = ScanNetVisualization()
    yfcc100m_vis.visualize(dkm)
