from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
from dkm import DKM
from dkm.utils.utils import tensor_to_pil


scenes = ["sacre_coeur"]
dkm = DKM(pretrained=True, version="mega_synthetic")
for scene in scenes:
    im1 = Image.open(f"assets/{scene}_multimodal_query.jpg").resize((512, 384))
    im2 = Image.open(f"assets/{scene}_multimodal_support.jpg").resize((512, 384))
    im1.save(f"vis/multimodality/{scene}_query.jpg")
    im2.save(f"vis/multimodality/{scene}_support.jpg")
    flow, confidence = dkm.match(im1, im2)
    confidence = confidence ** (1 / 2)
    c_b = confidence / confidence.max()
    x2 = (torch.tensor(np.array(im2)) / 255).cuda().permute(2, 0, 1)
    im2_transfer_rgb = F.grid_sample(
        x2[None], flow[..., 2:][None], mode="bicubic", align_corners=False
    )[0]
    white_im = torch.ones_like(x2)
    tensor_to_pil(
        c_b * im2_transfer_rgb + (1 - c_b) * white_im, unnormalize=False
    ).save(f"vis/multimodality/{scene}_warped.jpg")
