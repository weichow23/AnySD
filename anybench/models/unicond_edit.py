import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{current_dir}/unicond")
from anybench.models.unicond.utils.share import *
import PIL
import anybench.models.unicond.utils.config as config
import cv2
import einops
import numpy as np
import torch
from pytorch_lightning import seed_everything
from anybench.models.unicond.annotator.util import resize_image, HWC3
from anybench.models.unicond.annotator.canny import CannyDetector
from anybench.models.unicond.annotator.mlsd import MLSDdetector
from anybench.models.unicond.annotator.hed import HEDdetector
from anybench.models.unicond.annotator.sketch import SketchDetector
from anybench.models.unicond.annotator.openpose import OpenposeDetector
from anybench.models.unicond.annotator.midas import MidasDetector
from anybench.models.unicond.annotator.uniformer import UniformerDetector
from anybench.models.unicond.annotator.content import ContentDetector
from anybench.models.unicond.models.util import create_model, load_state_dict
from anybench.models.unicond.models.ddim_hacked import DDIMSampler

class Unicond():
    def __init__(self, model_path="shihaozhao/uni-controlnet"):
        self.apply_canny = CannyDetector()
        self.apply_mlsd = MLSDdetector()
        self.apply_hed = HEDdetector()
        self.apply_sketch = SketchDetector()
        self.apply_openpose = OpenposeDetector()
        self.apply_midas = MidasDetector()
        self.apply_seg = UniformerDetector()
        self.apply_content = ContentDetector(model_name="openai/clip-vit-large-patch14")

        self.model = create_model(f'{current_dir}/unicond/configs/uni_v15.yaml').cpu()
        self.model.load_state_dict(load_state_dict(model_path, location='cuda'))
        self.model = self.model.cuda()
        self.ddim_sampler = DDIMSampler(self.model)

    def _process(self, canny_image, mlsd_image, hed_image, sketch_image, openpose_image,
                midas_image, seg_image, content_image, prompt, a_prompt, n_prompt, num_samples,
                image_resolution, ddim_steps, strength, scale, seed, eta,
                low_threshold, high_threshold, value_threshold, distance_threshold,
                alpha, global_strength):

        seed_everything(seed)

        if canny_image is not None:
            anchor_image = canny_image
        elif mlsd_image is not None:
            anchor_image = mlsd_image
        elif hed_image is not None:
            anchor_image = hed_image
        elif sketch_image is not None:
            anchor_image = sketch_image
        elif openpose_image is not None:
            anchor_image = openpose_image
        elif midas_image is not None:
            anchor_image = midas_image
        elif seg_image is not None:
            anchor_image = seg_image
        elif content_image is not None:
            anchor_image = content_image
        else:
            anchor_image = np.zeros((image_resolution, image_resolution, 3)).astype(np.uint8)

        H, W, C = resize_image(HWC3(anchor_image), image_resolution).shape

        with torch.no_grad():
            if canny_image is not None:
                canny_image = cv2.resize(canny_image, (W, H))
                canny_detected_map = HWC3(self.apply_canny(HWC3(canny_image), low_threshold, high_threshold))
            else:
                canny_detected_map = np.zeros((H, W, C)).astype(np.uint8)
            if mlsd_image is not None:
                mlsd_image = cv2.resize(mlsd_image, (W, H))
                mlsd_detected_map = HWC3(self.apply_mlsd(HWC3(mlsd_image), value_threshold, distance_threshold))
            else:
                mlsd_detected_map = np.zeros((H, W, C)).astype(np.uint8)
            if hed_image is not None:
                hed_image = cv2.resize(hed_image, (W, H))
                hed_detected_map = HWC3(self.apply_hed(HWC3(hed_image)))
            else:
                hed_detected_map = np.zeros((H, W, C)).astype(np.uint8)
            if sketch_image is not None:
                sketch_image = cv2.resize(sketch_image, (W, H))
                sketch_detected_map = HWC3(self.apply_sketch(HWC3(sketch_image)))
            else:
                sketch_detected_map = np.zeros((H, W, C)).astype(np.uint8)
            if openpose_image is not None:
                openpose_image = cv2.resize(openpose_image, (W, H))
                openpose_detected_map, _ = self.apply_openpose(HWC3(openpose_image), False)
                openpose_detected_map = HWC3(openpose_detected_map)
            else:
                openpose_detected_map = np.zeros((H, W, C)).astype(np.uint8)
            if midas_image is not None:
                midas_image = cv2.resize(midas_image, (W, H))
                midas_detected_map = HWC3(self.apply_midas(HWC3(midas_image), alpha))
            else:
                midas_detected_map = np.zeros((H, W, C)).astype(np.uint8)
            if seg_image is not None:
                seg_image = cv2.resize(seg_image, (W, H))
                seg_detected_map, _ = self.apply_seg(HWC3(seg_image))
                seg_detected_map = HWC3(seg_detected_map)
            else:
                seg_detected_map = np.zeros((H, W, C)).astype(np.uint8)
            if content_image is not None:
                content_emb = self.apply_content(content_image)
            else:
                content_emb = np.zeros((768))

            detected_maps_list = [canny_detected_map,
                                  mlsd_detected_map,
                                  hed_detected_map,
                                  sketch_detected_map,
                                  openpose_detected_map,
                                  midas_detected_map,
                                  seg_detected_map
                                  ]
            detected_maps = np.concatenate(detected_maps_list, axis=2)

            local_control = torch.from_numpy(detected_maps.copy()).float().cuda() / 255.0
            local_control = torch.stack([local_control for _ in range(num_samples)], dim=0)
            local_control = einops.rearrange(local_control, 'b h w c -> b c h w').clone()
            global_control = torch.from_numpy(content_emb.copy()).float().cuda().clone()
            global_control = torch.stack([global_control for _ in range(num_samples)], dim=0)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=False)

            uc_local_control = local_control
            uc_global_control = torch.zeros_like(global_control)
            cond = {"local_control": [local_control],
                    "c_crossattn": [self.model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)],
                    'global_control': [global_control]}
            un_cond = {"local_control": [uc_local_control],
                       "c_crossattn": [self.model.get_learned_conditioning([n_prompt] * num_samples)],
                       'global_control': [uc_global_control]}
            shape = (4, H // 8, W // 8)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=True)

            self.model.control_scales = [strength] * 13
            samples, _ = self.ddim_sampler.sample(ddim_steps, num_samples,
                                                 shape, cond, verbose=False, eta=eta,
                                                 unconditional_guidance_scale=scale,
                                                 unconditional_conditioning=un_cond, global_strength=global_strength)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=False)

            x_samples = self.model.decode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
            results = [x_samples[i] for i in range(num_samples)]

        return results, detected_maps_list
    def _ulr2np(self, url):
        hed_image = cv2.imread(url)
        hed_image = hed_image.astype(np.uint8)
        return hed_image

    def edit(self, image_url, cond_image_url, prompt, save_path, cond_type):

        content_image = self._ulr2np(image_url)
        canny_image = None
        mlsd_image = None
        hed_image = None
        sketch_image = None
        openpose_image = None
        midas_image = None
        seg_image = None

        if cond_type in ['visual_material_transfer', 'visual_reference', 'visual_bbox']:
            hed_image = self._ulr2np(cond_image_url)
        elif cond_type in ['visual_depth']:
            midas_image = self._ulr2np(cond_image_url)
        elif cond_type in ['visual_scribble']:
            canny_image = self._ulr2np(cond_image_url)
        elif cond_type in ['visual_segment']:
            seg_image = self._ulr2np(cond_image_url)
        elif cond_type in ['visual_sketch']:
            sketch_image = self._ulr2np(cond_image_url)
        else:
            raise ValueError(f"Unknown cond type {cond_type}")

        num_samples = 1
        image_resolution = 512
        strength = 1
        global_strength = 1
        low_threshold = 100
        high_threshold = 200
        value_threshold = 0.1
        distance_threshold = 0.1
        alpha = 6.2
        ddim_steps = 50
        scale = 7.5
        seed = 42
        eta = 0.0
        a_prompt = 'best quality, extremely detailed'
        n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

        image_gallery, cond_gallery = self._process(canny_image, mlsd_image, hed_image, sketch_image, openpose_image,
                                                    midas_image, seg_image, content_image, prompt,
                                                    a_prompt, n_prompt, num_samples, image_resolution,
                                                    ddim_steps, strength, scale, seed, eta, low_threshold,
                                                    high_threshold, value_threshold, distance_threshold,
                                                    alpha, global_strength)

        image = PIL.Image.fromarray(image_gallery[0])
        image.save(save_path)

