from os import getenv, listdir, makedirs

try:
  from env import GEMINI_API_KEY
except:
  GEMINI_API_KEY = getenv("GEMINI_API_KEY")

import json
import numpy as np

from PIL import Image as PImage, ImageFilter as PImageFilter
from random import randint, sample
from utils import get_img_description

try:
  import torch
  from diffusers import AutoPipelineForInpainting
except:
  print("no pytorch")


# "runwayml/stable-diffusion-inpainting"
# "stable-diffusion-v1-5/stable-diffusion-inpainting"
# "stabilityai/stable-diffusion-2-inpainting"
class LandscapeGenerator:
  @classmethod
  def get_pipeline(cls, model):
    return AutoPipelineForInpainting.from_pretrained(
      model,
      torch_dtype=torch.float16,
      variant="fp16",
      safety_checker=None
    ).to("cuda")

  @classmethod
  def resize_by_height(cls, src, height):
    iw, ih = src.size
    nw = int(iw * height / ih)
    return src.resize((nw, height)).convert("RGB")

  @classmethod
  def add_mask(cls, img, x, w):
    iw, ih = img.size
    x1 = min(x+w, iw)
    img_np = np.array(img)
    img_np[:, x:x1] = (0, 0, 0)
    return PImage.fromarray(img_np, "RGB")

  @classmethod
  def add_crop(cls, src, sx0, crop_width, dst, mask, dx0):
    if type(dst) == list or type(dst) == tuple:
      dst = PImage.fromarray(255 * np.ones((dst[1], dst[0], 3), dtype=np.uint8), "RGB")
      mask = dst.copy()

    # assume sh == dh
    dw,dh = dst.size
    sw,sh = src.size
    sx1 = min(sx0 + crop_width, sw)
    dx1 = min(dx0 + sx1 - sx0, dw)
    actual_crop_width = min(sx1 - sx0, dx1 - dx0)
    dst.paste(src.crop((sx0, 0, sx0 + actual_crop_width, sh)), (dx0, 0))
    nmask = cls.add_mask(mask, dx0, actual_crop_width)
    return dst, nmask

  @classmethod
  def get_input_images(cls, left_img, keep_width, size, right_img=None):
    left_img = cls.resize_by_height(left_img, height=size[1])
    img, mask = cls.add_crop(left_img, left_img.size[0] - keep_width, keep_width, size, None, 0)

    if right_img is not None:
      right_img = cls.resize_by_height(right_img, height=size[1])
      mask_start = (img.size[0] - right_img.size[0]) if (img.size[0] - right_img.size[0]) > (3 * keep_width) else 3 * keep_width
      img, mask = cls.add_crop(right_img, 0, right_img.size[0], img, mask, mask_start)

    mask_blurred = mask#.filter(PImageFilter.GaussianBlur(radius=(4, 0)))

    return img, mask_blurred

  @classmethod
  def prep_graft(cls, limg, rimg, keep_width=256, right_offset=20, label="grafted"):
    orimg = rimg.crop((right_offset + keep_width, 0, rimg.size[0], rimg.size[1]))
    orimg.save(f"./imgs/{label}b.jpg")

    rimg = rimg.crop((right_offset, 0, right_offset + keep_width, rimg.size[1]))
    img_in, mask_in = cls.get_input_images(limg, keep_width=keep_width, size=(4*keep_width, limg.size[1]), right_img=rimg)
    return img_in, mask_in

  @classmethod
  def build_prompt(cls, prompt_content=None, prompt_style=None):
    content_modifier_options = [
      "things on fire",
      "floods",
      "droughts",
      "environmental crisis",
      "global warming",
      "trash and rubble",
      "environmental crisis",
      "collapsed buildings",
      "vultures and rats"
    ]

    k = randint(2, 4)
    selections = sample(content_modifier_options, k)
    content_modifier = ", ".join(selections)

    if prompt_content is None or prompt_style is None:
      return f"{content_modifier} everywhere."
    else:
      return f"Apocalyptic version of {prompt_content}, with {content_modifier} everywhere. Using the style of {prompt_style}."

  @classmethod
  def stitch_images(cls, dir):
    label = dir.split("/")[-1]
    files = sorted([f for f in listdir(dir) if f.endswith(".jpg")])

    total_w = 0
    total_h = 0
    imgs = []
    for f in files:
      img = PImage.open(f"{dir}/{f}")
      iw,ih = img.size
      imgs.append(img)
      total_w += iw
      total_h = ih

    landscape_np = np.zeros((total_h, total_w, 3), dtype=np.uint8)
    cw = 0
    for img in imgs:
      landscape_np[:, cw:cw + img.size[0]] = np.array(img)[:, :]
      cw += img.size[0]

    landscape_out = PImage.fromarray(landscape_np)
    landscape_out.save(f"{dir}/{label}.jpg")

  def __init__(self, data, model):
    self.data = data
    self.pipe = LandscapeGenerator.get_pipeline(model)

  def gen_image(self, prompt, img_in, mask_in, n_images=1):
    output = self.pipe(
      prompt=prompt,
      negative_prompt="repetitive, distortion, glitch, borders, stretched, frames, breaks, multiple rows, gore, zombies, violence, splits, maps, diagrams, text, font, logos, branding",
      image=img_in,
      mask_image=mask_in,
      width=img_in.size[0], height=img_in.size[1],
      guidance_scale=16.0,
      num_inference_steps=32,
      num_images_per_prompt=n_images,
    )
    return output.images

  def gen_landscape(self, keep_width=256, size=(1440, 512), n=4, label="mural", seed_img=None):
    makedirs(f"./imgs/{label}/", exist_ok=True)

    if seed_img is None:
      img_idx = randint(0, len(self.data) - 1)
      seed_img = self.data[img_idx]["image"]
      seed_img_id = self.data[img_idx]["article_id"]
      prompt_content = self.data[img_idx]["content"][:-1]
      prompt_style = self.data[img_idx]["style"][:-1]
    else:
      description = get_img_description(seed_img)
      prompt_content = description["content"][:-1]
      prompt_style = description["style"][:-1]
      seed_img_id = ""

    seed_img = LandscapeGenerator.resize_by_height(seed_img, size[1])
    seed_img.save(f"./imgs/{label}/{label}_00.jpg")

    landscape_imgs = [seed_img]
    landscape_ids = [seed_img_id]
    landscape_running_width = seed_img.size[0]

    img_in, mask_in = LandscapeGenerator.get_input_images(seed_img, keep_width=keep_width, size=size)
    prompt = LandscapeGenerator.build_prompt(prompt_content=prompt_content, prompt_style=prompt_style)

    for i in range(1, n+1):
      img_out_raw = self.gen_image(prompt, img_in, mask_in)[0]
      img_out_np = np.array(img_out_raw)[:, keep_width:]
      img_out = PImage.fromarray(img_out_np)

      landscape_imgs.append(img_out)
      landscape_running_width += img_out.size[0]
      img_out.save(f"./imgs/{label}/{label}_{('0'+str(i))[-2:]}.jpg")

      img_in, mask_in = LandscapeGenerator.get_input_images(img_out, keep_width=keep_width, size=size)
      prompt = LandscapeGenerator.build_prompt(prompt_content=None, prompt_style=None)

      prompt_rand = randint(0, 100)
      if prompt_rand < 40:
        landscape_ids.append(landscape_ids[-1])
      else:
        img_idx = randint(0, len(self.data) - 1)
        news_img = self.data[img_idx]["image"]
        landscape_ids.append(self.data[img_idx]["article_id"])
        news_img = LandscapeGenerator.resize_by_height(news_img, size[1])
        img_in, mask_in = LandscapeGenerator.get_input_images(img_out, keep_width=keep_width, size=size, right_img=news_img)

    landscape_np = np.zeros((size[1], landscape_running_width, 3), dtype=np.uint8)
    cw = 0
    for img in landscape_imgs:
      landscape_np[:, cw:cw + img.size[0]] = np.array(img)[:, :]
      cw += img.size[0]

    landscape_out = PImage.fromarray(landscape_np)
    landscape_out.save(f"./imgs/{label}/{label}.jpg")

    with open(f"./imgs/{label}/{label}.json", "w") as ofp:
      json.dump(landscape_ids, ofp, separators=(",",":"), ensure_ascii=False)
