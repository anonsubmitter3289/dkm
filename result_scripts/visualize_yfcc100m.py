from dkm import DKM
import numpy as np
import torch
from dkm.utils import *
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
import pickle

imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).cuda()
imagenet_std = torch.tensor([0.229, 0.224, 0.225]).cuda()


class Yfcc100mVisualization:
    def __init__(self, data_root="data/yfcc100m_test") -> None:
        self.scenes = [
            "buckingham_palace",
            "notre_dame_front_facade",
            "reichstag",
            "sacre_coeur",
        ]
        self.data_root = data_root

    def visualize(self, model, r=2):
        model.train(False)
        with torch.no_grad():
            data_root = self.data_root
            meta_info = open(
                f"{data_root}/yfcc_test_pairs_with_gt.txt", "r"
            ).readlines()
            tot_e_t, tot_e_R, tot_e_pose = [], [], []
            for scene_ind in range(len(self.scenes)):
                scene = self.scenes[scene_ind]
                pairs = np.array(
                    pickle.load(
                        open(f"{data_root}/pairs/{scene}-te-1000-pairs.pkl", "rb")
                    )
                )
                scene_dir = f"{data_root}/yfcc100m/{scene}/test/"
                images = open(scene_dir + "images.txt", "r").read().split("\n")
                pair_inds = np.random.choice(range(len(pairs)), size=5, replace=False)
                for pairind in tqdm(pair_inds):
                    idx1, idx2 = pairs[pairind]
                    params = meta_info[1000 * scene_ind + pairind].split()
                    rot1, rot2 = int(params[2]), int(params[3])
                    im1_path = images[idx1]
                    im2_path = images[idx2]
                    im1 = Image.open(scene_dir + im1_path).rotate(
                        rot1 * 90, expand=True
                    )
                    im2 = Image.open(scene_dir + im2_path).rotate(
                        rot2 * 90, expand=True
                    )
                    im1 = im1.resize((512, 384))
                    im2 = im2.resize((512, 384))
                    x2 = (torch.tensor(np.array(im2)) / 255).cuda().permute(2, 0, 1)
                    dense_matches, dense_certainty = model.match(im1, im2)
                    dense_certainty = dense_certainty ** (1 / r)
                    im2_transfer_rgb = F.grid_sample(
                        x2[None],
                        dense_matches[..., 2:][None],
                        mode="bicubic",
                        align_corners=False,
                    )[0]
                    white_im = torch.ones_like(x2)
                    c_b = dense_certainty / dense_certainty.max()
                    im1_name, im2_name = (
                        im1_path.split("/")[-1].split(".")[0],
                        im2_path.split("/")[-1].split(".")[0],
                    )
                    im1.save(f"vis/yfcc100m/{im1_name}_{im2_name}_query.jpg")
                    im2.save(f"vis/yfcc100m/{im1_name}_{im2_name}_support.jpg")
                    tensor_to_pil(
                        c_b * im2_transfer_rgb + (1 - c_b) * white_im, unnormalize=False
                    ).save(f"vis/yfcc100m/{im1_name}_{im2_name}_warped.jpg")


if __name__ == "__main__":
    dkm = DKM(pretrained=True, version="mega_synthetic")

    yfcc100m_vis = Yfcc100mVisualization()
    yfcc100m_vis.visualize(dkm)
