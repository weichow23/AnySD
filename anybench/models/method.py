from __future__ import annotations
import PIL
import os
import math
import numpy as np
import torch
import einops
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torch import autocast
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{current_dir}/stable_diffusion")
from anybench.models.stable_diffusion.ldm.util import instantiate_from_config
from anysd.src.model import AnySDEidtPipeline as AnySD
from anybench.models.unicond_edit import Unicond

model_list = ["instructionpix2pix", 'anysd', 'magicbrush', 'ultra', 'nulltext', 'hivew', 'hivec',
              'unicond', 'nulltext-1.5', 'nulltext-xl']

class UniEditModel:
    def __init__(self, name):
        assert name in model_list, f'{name} must in {model_list}'
        if name == 'instructionpix2pix':
            self.model = InstructPix2Pix()
        elif name == 'anysd':
            self.model = AnySD()
        elif name == 'magicbrush':
            self.model = MagicBrush()
        elif name == 'ultra':
            self.model = Ultra()
        elif name == 'hivew':
            self.model = HIVEw()
        elif name == 'hivec':
            self.model = HIVEc()
        elif name == 'unicond':
            self.model = Unicond()
        elif name == 'nulltext':
            self.model = NullText(mode='base')
        elif name == 'nulltext-1.5':
            self.model = NullText(mode='1.5')
        elif name == 'nulltext-xl':
            self.model = NullText(mode='xl')
        else:
            raise NotImplementedError

        self.model_name = name

    def emu_edit(self, batch, bench_name, save_path):
        # for Emu Edit
        if self.model_name in ['nulltext', 'nulltext-1.5', 'nulltext-xl']:
            self.model.edit(image_url=batch['image'],
                   prompt=[batch['input_caption'], batch['output_caption']],
                   save_path=f"./anybench/results/{bench_name}/{self.model_name}/{batch['idx']}.png")
        elif self.model_name == 'unicond':
            pass
        elif self.model_name in ["instructionpix2pix", 'anysd', 'magicbrush', 'ultra', 'hivew', 'hivec']:
            self.model.edit(image_url=[item['image'] for item in batch],
                       prompt=[item['instruction'] for item in batch],
                       save_path=[f"{save_path}/{item['idx']}.png" for item in batch])
        else:
            raise NotImplementedError
    def magic_edit(self, image_path, save_output_img_path, data):
        # for MagicBrush
        if self.model_name in ['nulltext', 'nulltext-1.5', 'nulltext-xl']:
            self.model.edit(image_url=image_path, save_path=save_output_img_path,
                   prompt=[data['input_caption'], data['output_caption']])
        elif self.model_name == 'unicond':
            pass
        elif self.model_name in ["instructionpix2pix", 'anysd', 'magicbrush', 'ultra', 'hivew', 'hivec']:
            self.model.edit(image_url=image_path, save_path=save_output_img_path, prompt=data['instruction'])
        else:
            raise NotImplementedError
    def anybench_edit(self, dim, item, logging, save_path, image_url, cond_image_url):
        # for AnyBench
        if self.model_name != "unicond":
            if self.model_name == 'anysd':
                pass # use anysd/infer.py
            elif self.model_name in ['nulltext', 'nulltext-1.5', 'nulltext-xl']:
                if "input" in item.keys() and "output" in item.keys() and \
                        item["input"] is not None and item["output"] is not None:
                    prompt = [item["input"], item["output"]]
                else:
                    logging.info(f'Null Text not adaptable to {dim}')
                    sys.exit(0)
            else:
                prompt = item['edit']

            self.model.edit(image_url=image_url,
                       prompt=prompt,
                       save_path=save_path)
        else:
            self.model.edit(image_url=image_url,
                       prompt=item['edit'],
                       save_path=save_path,
                       cond_image_url=cond_image_url,
                       cond_type=dim)


class InstructPix2Pix():
    def __init__(self, model_path = "timbrooks/instruct-pix2pix"):
        from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_path,
                                                                      torch_dtype=torch.float16,
                                                                      safety_checker=None)
        self.pipe.to("cuda")
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)

    @torch.inference_mode()
    def edit(self, image_url, prompt, save_path):
        if type(image_url) is str:
            image = PIL.ImageOps.exif_transpose(Image.open(image_url))
            image = image.convert("RGB").resize((512, 512))
        else:
            image = image_url.convert("RGB").resize((512, 512))

        images = self.pipe(prompt, image=image, num_inference_steps=10, image_guidance_scale=1).images
        images[0].save(save_path)


class Ultra():
    def __init__(self, model_path='BleachNick/SD3_UltraEdit_freeform'):
        from diffusers import StableDiffusion3InstructPix2PixPipeline
        self.pipe = StableDiffusion3InstructPix2PixPipeline.from_pretrained(model_path,
                                                                      torch_dtype=torch.float16,
                                                                      safety_checker=None)
        self.pipe.to("cuda")

    @torch.inference_mode()
    def edit(self, image_url, prompt, save_path):
        if type(image_url) is str:
            image = PIL.Image.open(image_url)
            image = PIL.ImageOps.exif_transpose(image)
            image = image.convert("RGB").resize((512, 512))
        else:
            image = image_url.convert("RGB").resize((512, 512))
        images = self.pipe(prompt, image=image, negative_prompt="",
            num_inference_steps=50,
            image_guidance_scale=1.5,
            guidance_scale=7.5).images
        images[0].save(save_path)

class MagicBrush():
    def __init__(self, model_path=f"vinesmsuic/magicbrush-jul7"):
        from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16
        ).to("cuda")
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)

    @torch.inference_mode()
    def edit(self, image_url, prompt, save_path):
        if type(image_url) is str:
            image = PIL.Image.open(image_url)
            image = PIL.ImageOps.exif_transpose(image)
            image = image.convert("RGB").resize((512, 512))
        else:
            image = image_url.convert("RGB").resize((512, 512))
        generator = torch.manual_seed(seed=42)
        images = self.pipe(prompt, image=image, num_inference_steps=20, image_guidance_scale=1.5,
                                  guidance_scale=7, generator=generator).images
        images[0].save(save_path)

class NullText():
    def __init__(self, mode):
        from anybench.models.nulltext.p2p import NullInversion
        from diffusers import StableDiffusionPipeline, DDIMScheduler, DDPMScheduler, StableDiffusionXLPipeline
        # already set the default resolution to be 512 in p2p
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                  clip_sample=False,
                                  set_alpha_to_one=False)
        if mode == 'xl':
            model_path = 'stabilityai/stable-diffusion-xl-base-1.0'

            self.ldm_stable = StableDiffusionXLPipeline.from_pretrained(
                model_path,
                scheduler=scheduler,
                # torch_dtype=torch.float16,
                add_watermarker=False,
            ).to(device)
        else:
            if mode == '1.5':
                model_path = "stable-diffusion-v1-5/stable-diffusion-v1-5"
            else:
                model_path = "CompVis/stable-diffusion-v1-4"
            self.ldm_stable = StableDiffusionPipeline.from_pretrained(model_path, scheduler=scheduler).to(device)

        try:
            self.ldm_stable.disable_xformers_memory_efficient_attention()
        except AttributeError:
            print("Attribute disable_xformers_memory_efficient_attention() is missing")
        self.tokenizer = self.ldm_stable.tokenizer
        self.mode = mode
        self.null_inversion = NullInversion(self.ldm_stable)

    def edit(self, image_url, prompt, save_path):  # zw@zju: Need Grad!!!
        # To avoid "attention replacement edit can only be applied on prompts with the same length but prompt A has 10 words and prompt B has 9 words."
        prompt_1_words = prompt[0].split(' ')
        prompt_2_words = prompt[1].split(' ')
        if len(prompt_1_words) != len(prompt_2_words):
            print("Truncate to the length of the shorter prompt")
            min_length = min(len(prompt_1_words), len(prompt_2_words))
            prompt[0] = ' '.join(prompt_1_words[:min_length])
            prompt[1] = ' '.join(prompt_2_words[:min_length])

        from anybench.models.nulltext.p2p import make_controller, run_and_display
        (image_gt, image_enc), x_t, uncond_embeddings = \
            self.null_inversion.invert(image_url, prompt[0], offsets=(0, 0, 200, 0), verbose=False, mode=self.mode)
        controller = make_controller(prompt, True, cross_replace_steps= {'default_': .8, },
                                     self_replace_steps= .5, blend_words=None,
                                     equilizer_params=None, tokenizer=self.tokenizer)
        images, _ = run_and_display(prompt, controller, run_baseline=False, latent=x_t,
                                    uncond_embeddings=uncond_embeddings, ldm_stable=self.ldm_stable, mode=self.mode)
        pil_image = PIL.Image.fromarray(images[0])
        print(f"Save in {save_path}")
        pil_image.save(save_path)

class CFGDenoiser(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        cfg_z = einops.repeat(z, "1 ... -> n ...", n=3)
        cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=3)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
        }
        out_cond, out_img_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        return out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)


def load_model_from_config(config, ckpt, vae_ckpt=None, verbose=False):
    import k_diffusion as K
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    if vae_ckpt is not None:
        print(f"Loading VAE from {vae_ckpt}")
        vae_sd = torch.load(vae_ckpt, map_location="cpu")["state_dict"]
        sd = {
            k: vae_sd[k[len("first_stage_model.") :]] if k.startswith("first_stage_model.") else v
            for k, v in sd.items()
        }
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    return model

class HIVEw():
    def __init__(self, model_path="./anybench/checkpoints/hive/hive_rw.ckpt"):
        import k_diffusion as K
        self.resolution = 512
        self.steps = 100
        self.seed = 100

        config_path = os.path.join(current_dir, "hive/generate.yaml")
        config = OmegaConf.load(config_path)

        self.model = load_model_from_config(config, model_path, None)
        self.model.eval().cuda()
        self.model_wrap = K.external.CompVisDenoiser(self.model)
        self.model_wrap_cfg = CFGDenoiser(self.model_wrap)
        self.null_token = self.model.get_learned_conditioning([""])

    @torch.inference_mode()
    def edit(self, image_url, prompt, save_path):
        import k_diffusion as K
        if type(image_url) is str:
            input_image = image_url.convert("RGB")
        else:
            input_image = PIL.Image.open(image_url).convert("RGB")

        width, height = input_image.size
        factor = self.resolution / max(width, height)
        factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
        width = int((width * factor) // 64) * 64
        height = int((height * factor) // 64) * 64
        input_image = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)

        with torch.no_grad(), autocast("cuda"), self.model.ema_scope():
            cond = {}
            cond["c_crossattn"] = [self.model.get_learned_conditioning([prompt])]
            input_image = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
            input_image = rearrange(input_image, "h w c -> 1 c h w").to(self.model.device)
            cond["c_concat"] = [self.model.encode_first_stage(input_image).mode()]

            uncond = {}
            uncond["c_crossattn"] = [self.null_token]
            uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]

            sigmas = self.model_wrap.get_sigmas(self.steps)

            extra_args = {
                "cond": cond,
                "uncond": uncond,
                "text_cfg_scale": 7.5,
                "image_cfg_scale": 1.5,
            }
            torch.manual_seed(self.seed)
            z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
            z = K.sampling.sample_euler_ancestral(self.model_wrap_cfg, z, sigmas, extra_args=extra_args)
            x = self.model.decode_first_stage(z)
            x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
            x = 255.0 * rearrange(x, "1 c h w -> h w c")
            edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())
        edited_image.save(save_path)

class HIVEc(HIVEw):
    def __init__(self, model_path="./anybench/checkpoints/hive/hive_rw_condition.ckpt"):
        super().__init__(model_path=model_path)

    @torch.inference_mode()
    def edit(self, image_url, prompt, save_path):
        if type(image_url) is str:
            input_image = Image.open(image_url).convert("RGB")
        else:
            input_image = image_url.convert("RGB")

        import k_diffusion as K
        width, height = input_image.size
        factor = self.resolution / max(width, height)
        factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
        width = int((width * factor) // 64) * 64
        height = int((height * factor) // 64) * 64
        input_image = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)

        with torch.no_grad(), autocast("cuda"), self.model.ema_scope():
            cond = {}
            edit = prompt + ', ' + f'image quality is five out of five'
            cond["c_crossattn"] = [self.model.get_learned_conditioning([edit])]
            input_image = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
            input_image = rearrange(input_image, "h w c -> 1 c h w").to(self.model.device)
            cond["c_concat"] = [self.model.encode_first_stage(input_image).mode()]

            uncond = {}
            uncond["c_crossattn"] = [self.null_token]
            uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]

            sigmas = self.model_wrap.get_sigmas(self.steps)

            extra_args = {
                "cond": cond,
                "uncond": uncond,
                "text_cfg_scale": 7.5,
                "image_cfg_scale": 1.5,
            }
            torch.manual_seed(self.seed)
            z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
            z = K.sampling.sample_euler_ancestral(self.model_wrap_cfg, z, sigmas, extra_args=extra_args)
            x = self.model.decode_first_stage(z)
            x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
            x = 255.0 * rearrange(x, "1 c h w -> h w c")
            edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())
        edited_image.save(save_path)