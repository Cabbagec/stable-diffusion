#!/usr/bin/env python
# coding: utf-8
import json

# ### Load packages

# In[1]:


import logging

import argparse
import os
import shlex
import subprocess
import time
import uuid
from contextlib import nullcontext
from itertools import islice
from pathlib import Path

import cv2
import numpy as np
import torch
from IPython.display import display
from PIL import Image
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from einops import rearrange
from imwatermark import WatermarkEncoder

# display functionalities
from ipywidgets import Output
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torch import autocast
from torchvision.utils import make_grid
from tqdm import tqdm, trange
from transformers import AutoFeatureExtractor, logging as tlogging

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull

# ### Load functions

# In[2]:


logger = logging.getLogger()
logging.basicConfig(
    level=logging.INFO, format='[%(asctime)s]:%(levelname)s: %(message)s'
)
tlogging.set_verbosity_error()
realesrgan_dir = '/Repositories/realesrgan'


@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        # with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
        with redirect_stderr(fnull) as err:
            yield (err,)


# load safety model, return the extractor and checker
def load_safety_module():
    safety_model_id = "CompVis/stable-diffusion-safety-checker"
    safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
    safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)
    return safety_feature_extractor, safety_checker


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False, half=False):
    with suppress_stdout_stderr():
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]
        model = instantiate_from_config(config.model)
        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m)
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u)

        if half:
            model.half()

        model.cuda()
        model.eval()
        return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y) / 255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image, extractor, checker):
    if None in (extractor, checker):
        return x_image, None

    safety_checker_input = extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = checker(
        images=x_image, clip_input=safety_checker_input.pixel_values
    )
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
            print(
                f'One image is replaced by safety check. Use with argument `--unsafe` to keep all results temporarily.'
            )
    return x_checked_image, has_nsfw_concept


def decode_single_sample_from_model(model_samples: torch.Tensor, model):
    # with torch.no_grad(), torch.autocast('cuda'), model.ema_scope():
    decoded = model.decode_first_stage(model_samples)
    clamped_samples = torch.clamp((decoded + 1.0) / 2.0, min=0.0, max=1.0)
    if len(clamped_samples) == 0:
        print(f'No samples generated, skipping...')
        return

    first_clamped_samples = clamped_samples[0]
    rearranged = 255.0 * rearrange(
        first_clamped_samples.cpu().numpy(), 'c h w -> h w c'
    )
    pil_image = Image.fromarray(rearranged.astype(np.uint8))
    return pil_image


# ipywidgets suppresses exceptions, use the custom widget class inherited from the original one.
# see https://github.com/jupyter-widgets/ipywidgets/issues/3208#issuecomment-1070836153
class NoCatchOutput(Output):
    def __exit__(self, *args, **kwargs):
        super().__exit__(*args, **kwargs)


class ProgressDisplayer:
    def __init__(
        self,
        show_progress=True,
        save_progress=False,
        save_path='/tmp',
        displayer_uuid=None,
    ):
        self.show_progress = show_progress
        self.displayer_uuid = (
            str(uuid.uuid4()) if not displayer_uuid else displayer_uuid
        )
        if self.show_progress:
            self.output = NoCatchOutput()
            display(self.output)

        self.displayer = None
        self.save_progress = save_progress
        self.save_path = Path(save_path)
        self.last_img_path = None
        self.index = 0
        self.total_steps = None
        self.abort_flag = False
        self.index_path = {}
        if save_path:
            if not self.save_path.exists() or not self.save_path.is_dir():
                print(
                    f'path {save_path} for saving progress images does not exist, will not save'
                )
                self.save_progress = False
            else:
                self.save_path /= self.displayer_uuid
                self.save_path.mkdir()
                print(f'saving progress files to {self.save_path}')

    def refresh_img(self, image: Image, index=None, total_steps=None):
        self.index = index or self.index
        img_path = self.save_path / f'{index:03d}.png'
        self.last_img_path = img_path
        self.total_steps = total_steps or self.total_steps

        if self.show_progress:
            with self.output:
                if not self.displayer:
                    self.displayer = display(image, clear=True)
                else:
                    self.displayer.display(image)

        if self.save_progress:
            image.save(img_path)
            self.index_path[index] = img_path

    def get_callback(self, model):
        def callback(array, index, total_steps):
            if self.abort_flag:
                torch.cuda.empty_cache()
                raise Exception(f'Job aborted')
            image = decode_single_sample_from_model(array, model)
            self.refresh_img(image, index, total_steps)

        return callback

    def finish(self):
        with open(self.save_path / 'result.json', 'w+') as f:
            json.dump(
                {
                    'job_id': self.displayer_uuid,
                    'last_img': str(self.last_img_path),
                    'last_index': self.index,
                },
                fp=f,
            )

    def abort_on_next(self):
        self.abort_flag = True


# In[3]:


def get_opt(
    prompt, steps=50, height=512, width=512, scale=4.0, seed=None, *args, **kwargs
):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples",
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps", type=int, default=50, help="number of ddim sampling steps"
    )
    parser.add_argument("--plms", action='store_true', help="use plms sampling")
    parser.add_argument(
        "--laion400m", action='store_true', help="uses the LAION400M model"
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument("--n_iter", type=int, default=2, help="sample this often")
    parser.add_argument(
        "--H", type=int, default=512, help="image height, in pixel space"
    )
    parser.add_argument(
        "--W", type=int, default=512, help="image width, in pixel space"
    )
    parser.add_argument("--C", type=int, default=4, help="latent channels")
    parser.add_argument("--f", type=int, default=8, help="downsampling factor")
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows", type=int, default=0, help="rows in the grid (default: n_samples)"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file", type=str, help="if specified, load prompts from this file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="the seed (for reproducible sampling)"
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast",
    )
    parser.add_argument(
        "--half", action='store_true', help='use half of model precision'
    )
    parser.add_argument(
        "--unsafe", action='store_true', help='keep all results temporarily'
    )
    opt = parser.parse_args(shlex.split(argstring))

    update = {
        'prompt': prompt,
        'ddim_steps': int(steps),
        'half': True,
        'unsafe': True,
        'H': height,
        'W': width,
        'n_iter': 1,
        'n_samples': 1,
        'skip_grid': True,
        'skip_save': True,
        'scale': scale,
        'seed': seed,
    }
    opt.__dict__.update(update)
    return opt


def load_model():
    config = OmegaConf.load("configs/stable-diffusion/v1-inference.yaml")
    model = load_model_from_config(
        config, "models/ldm/stable-diffusion-v1/model.ckpt", half=True
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    return model


def run(opt, model, progress_displayer=None):
    opt.x = 'sd'
    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"

    # seed_everything(opt.seed)
    #
    # print(f'------------------ Start loading...')
    # with suppress_stdout_stderr():
    #     config = OmegaConf.load(f"{opt.config}")
    # model = load_model_from_config(config, f"{opt.ckpt}", half=opt.half)
    # print(f'------------------ Model loaded')
    #
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    # print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    # wm = "StableDiffusionV1"
    # wm_encoder = WatermarkEncoder()
    # wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    start_code = None
    if opt.fixed_code:
        pass
        # start_code = torch.randn(
        #     [opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device
        # )

    seed = (uuid.uuid4().int & ((1 << 32) - 1)) if not opt.seed else opt.seed
    print(f'Using random seed of {seed}')
    seed_everything(seed)

    if not opt.unsafe:
        unsafe_extractor, unsafe_checker = load_safety_module()
        print(
            "Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)..."
        )
        wm = "StableDiffusionV1"
        wm_encoder = WatermarkEncoder()
        wm_encoder.set_watermark('bytes', wm.encode('utf-8'))
    else:
        unsafe_extractor, unsafe_checker = None, None
        wm_encoder = None

    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                if not progress_displayer:
                    progress_displayer = ProgressDisplayer(
                        save_progress=True, show_progress=True
                    )
                callback_ = progress_displayer.get_callback(model)

                tic = time.time()
                all_samples = list()
                # progress = tqdm(range(opt.n_iter))
                for n in trange(opt.n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)
                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        samples_ddim, imediates = sampler.sample(
                            S=opt.ddim_steps,
                            conditioning=c,
                            batch_size=opt.n_samples,
                            shape=shape,
                            verbose=False,
                            img_callback=callback_,
                            unconditional_guidance_scale=opt.scale,
                            unconditional_conditioning=uc,
                            eta=opt.ddim_eta,
                            x_T=start_code,
                        )

                        x_samples_ddim = model.decode_first_stage(samples_ddim)

                        # return samples_ddim, x_samples_ddim, imediates, model, torch
                        x_samples_ddim = torch.clamp(
                            (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0
                        )
                        if not opt.unsafe:
                            x_samples_ddim = (
                                x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                            )
                            x_checked_image, has_nsfw_concept = check_safety(
                                x_samples_ddim, unsafe_extractor, unsafe_checker
                            )
                            x_samples_ddim = torch.from_numpy(x_checked_image).permute(
                                0, 3, 1, 2
                            )

                        if not opt.skip_save:
                            for x_sample in x_samples_ddim:
                                x_sample = 255.0 * rearrange(
                                    x_sample.cpu().numpy(), 'c h w -> h w c'
                                )
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                img = put_watermark(img, wm_encoder)
                                img.save(
                                    os.path.join(sample_path, f"{base_count:05}.png")
                                )
                                base_count += 1

                        if not opt.skip_grid:
                            all_samples.append(x_samples_ddim)

                if not opt.skip_grid:
                    # additionally, save as grid
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    grid = make_grid(grid, nrow=n_rows)

                    # to image
                    grid = 255.0 * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    img = Image.fromarray(grid.astype(np.uint8))
                    img = put_watermark(img, wm_encoder)
                    img.save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                    grid_count += 1

                toc = time.time()

    torch.cuda.empty_cache()
    progress_displayer.finish()
    print(
        f"Your samples are ready and waiting for you here: \n{outpath} \n" f" \nEnjoy."
    )


def generate_animation(
    dir_path: Path, fps=24, glob='*.png', encoder='libx264', filename='output.mp4'
):
    logging.info(f'generating animation for pics in {dir_path}')
    if (
        subprocess.run(
            [
                'ffmpeg',
                '-framerate',
                str(fps),
                '-pattern_type',
                'glob',
                '-i',
                glob,
                '-c:v',
                encoder,
                '-pix_fmt',
                'yuv420p',
                filename,
            ],
            cwd=dir_path,
        ).returncode
        == 0
    ):
        animation_filepath = dir_path / filename
        logging.info(f'generated: {animation_filepath}')
        return animation_filepath

    logging.error(f'failed to generate animation for images in path {dir_path}')


def generate_upscaled(image_path: Path, factor: int, filename=None):
    logging.info(f'generating upscaled version for {image_path}')
    factor = 4 if factor not in (2, 3, 4) else factor
    filename = f'output_x{factor}.jpg' if not filename else filename
    upscaled_filepath = image_path.parent / filename
    if (
        subprocess.run(
            [
                './realesrgan-ncnn-vulkan',
                '-i',
                str(image_path),
                '-o',
                upscaled_filepath,
                '-s',
                str(factor),
                '-n',
                'realesrgan-x4plus',
                '-f',
                'ext/jpg',
            ],
            cwd=Path(realesrgan_dir),
        ).returncode
        == 0
    ):
        logging.info(
            f'image upscaled: {image_path}, output: {upscaled_filepath}, factor: {factor}'
        )
        return upscaled_filepath

    logging.error(f'failed to generate upscaled image: {upscaled_filepath}')


# ### Run!

# In[4]:


p1 = "Photo of a raccoon wearing modern spacesuit, glass helmet, highly detailed, hyper realistic, 4k, sharp, pixar animation, photorealistic, unreal engine, artstation"

arg_kv = {
    "prompt": p1,
    "H": 640,
    "W": 448,
    "n_iter": 3,
    "n_samples": 1,
    "ddim_steps": 70,
    "unsafe": "",
    "half": "",
}
argstring = ''
for key, value in arg_kv.items():
    if not value:
        argstring += f' --{key} '
        continue

    if isinstance(value, str):
        argstring += f' --{key} "{value}" '
        continue

    else:
        argstring += f' --{key} {value} '

# if __name__ == '__main__':
#     try:
#         model = load_model()
#         run(get_opt(p1), model)
#     except Exception as e:
#         raise

# In[ ]:
