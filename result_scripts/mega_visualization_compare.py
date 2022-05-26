import torch
import numpy as np
import tqdm
import torch.nn.functional as F
from PIL import Image
import kornia.feature as KF
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


class MegadepthVisualization:
    def __init__(self, data_root="data/megadepth") -> None:
        self.scene_names = [
            "0015_0.1_0.3.npz",
            "0015_0.3_0.5.npz",
            "0022_0.1_0.3.npz",
            "0022_0.3_0.5.npz",
            "0022_0.5_0.7.npz",
        ]
        self.scenes = [
            np.load(f"{data_root}/{scene}", allow_pickle=True)
            for scene in self.scene_names
        ]
        self.data_root = data_root

    def visualize(self, dkm, loftr):
        dkm.train(False)
        with torch.no_grad():
            data_root = self.data_root
            for scene_ind in range(len(self.scenes)):
                scene = self.scenes[scene_ind]
                pairs = scene["pair_infos"]
                im_paths = scene["image_paths"]
                pair_inds = np.random.choice(range(len(pairs)), size=10, replace=False)
                for pairind in tqdm.tqdm(pair_inds):
                    idx1, idx2 = pairs[pairind][0]
                    im1_path = im_paths[idx1]
                    im2_path = im_paths[idx2]
                    im1 = Image.open(f"{data_root}/{im1_path}").resize((512, 384))
                    w1, h1 = im1.size
                    im2 = Image.open(f"{data_root}/{im2_path}").resize((512, 384))
                    w2, h2 = im2.size
                    x1 = (torch.tensor(np.array(im1)) / 255).cuda().permute(2, 0, 1)
                    x2 = (torch.tensor(np.array(im2)) / 255).cuda().permute(2, 0, 1)
                    dense_matches, dense_certainty = dkm.match(im1, im2)
                    dense_certainty = dense_certainty ** (1 / 2)
                    kornia_data = {
                        "image0": x1.mean(dim=0, keepdim=True)[None],
                        "image1": x2.mean(dim=0, keepdim=True)[None],
                    }
                    out = loftr(kornia_data)
                    kpts1, kpts2, certainty, batch_inds = (
                        out["keypoints0"],
                        out["keypoints1"],
                        out["confidence"],
                        out["batch_indexes"],
                    )
                    im2_transfer_rgb = F.grid_sample(
                        x2[None],
                        dense_matches[..., 2:][None],
                        mode="bicubic",
                        align_corners=False,
                    )[0]
                    white_im = torch.ones_like(x2)
                    c_b = dense_certainty / dense_certainty.max()
                    im1_name = im1_path.split(".jpg")[0].split("/")[-1]
                    im2_name = im2_path.split(".jpg")[0].split("/")[-1]

                    im1.save(f"vis/mega1500/{im2_name}_{im1_name}.jpg")
                    im2.save(f"vis/mega1500/{im2_name}.jpg")
                    tensor_to_pil(
                        c_b * im2_transfer_rgb + (1 - c_b) * white_im, unnormalize=False
                    ).save(f"vis/mega1500/{im2_name}_{im1_name}_dkm_warped.jpg")
                    warped_im = torch.ones_like(x1)
                    left, right, up, down = (
                        torch.tensor((-1, 0)).cuda(),
                        torch.tensor((1, 0)).cuda(),
                        torch.tensor((0, -1)).cuda(),
                        torch.tensor((0, 1)).cuda(),
                    )
                    kpts1 = torch.cat(
                        (
                            kpts1 + left + up,
                            kpts1 + up,
                            kpts1 + up + right,
                            kpts1 + left,
                            kpts1,
                            kpts1 + right,
                            kpts1 + down + left,
                            kpts1 + down,
                            kpts1 + down + right,
                        )
                    )
                    kpts2 = torch.cat(
                        (kpts2, kpts2, kpts2, kpts2, kpts2, kpts2, kpts2, kpts2, kpts2)
                    )

                    x2_hat = torch.stack(
                        (2 * kpts2[..., 0] / 512 - 1, 2 * kpts2[..., 1] / 384 - 1),
                        dim=-1,
                    )
                    sampled_rgb = F.grid_sample(
                        x2[None],
                        x2_hat[None, None],
                        mode="bicubic",
                        align_corners=False,
                    )[0, :, 0]
                    warped_im[..., kpts1[:, 1].long(), kpts1[:, 0].long()] = sampled_rgb
                    tensor_to_pil(warped_im, unnormalize=False).save(
                        f"vis/mega1500/{im2_name}_{im1_name}_loftr_warped.png"
                    )


if __name__ == "__main__":
    loftr_model = KF.LoFTR(pretrained="outdoor")
    dkm_model = DKM(pretrained=True, version="mega")
    mega_vis = MegadepthVisualization()
    mega_vis.visualize(dkm_model, loftr_model)
