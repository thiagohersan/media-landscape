from os import getenv, makedirs

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
  def create_mask(cls, keep_width, size, right_keep_width=None):
    img_in = PImage.new("L", size)
    iw, ih = size
    if right_keep_width is not None:
      rkw = right_keep_width
      mask_start = (iw - rkw) if (iw - rkw) > (3 * keep_width) else 3 * keep_width
      img_in_pxs = [((i % iw) >= keep_width and (i % iw) < mask_start) * 255 for i in range(iw * ih)]
    else:
      img_in_pxs = [(i % iw >= keep_width) * 255 for i in range(iw * ih)]
    img_in.putdata(img_in_pxs)
    return img_in.convert("RGB")

  @classmethod
  def get_input_images(cls, left_img, keep_width, size, right_img=None):
    l_img_np = np.array(cls.resize_by_height(left_img, height=size[1]))

    if right_img is not None:
      r_img_np = np.array(cls.resize_by_height(right_img, height=size[1]))
      bgd_np = np.array(cls.create_mask(keep_width=keep_width, size=size, right_keep_width=right_img.size[0]))
      mask = cls.create_mask(keep_width=keep_width, size=size, right_keep_width=right_img.size[0])
    else:
      bgd_np = np.array(cls.create_mask(keep_width=keep_width, size=size))
      mask = cls.create_mask(keep_width=keep_width, size=size)

    bgd_np[:, :keep_width] = l_img_np[:, -keep_width:]

    if right_img is not None:
      rkw = right_img.size[0]
      right_start = (size[0] - rkw) if (size[0] - rkw) > (3 * keep_width) else 3 * keep_width
      bgd_np[:, right_start:size[0]]= r_img_np[:, :size[0] - right_start]

    img_in = PImage.fromarray(bgd_np)
    mask_blurred = mask.filter(PImageFilter.GaussianBlur(radius=(4, 0)))

    return img_in, mask_blurred

  def __init__(self, data, model):
    self.data = data
    self.pipe = LandscapeGenerator.get_pipeline(model)

  def build_prompt(self, prompt_content=None, prompt_style=None):
    if prompt_content is None:
      prompt_content = self.data[randint(0, len(self.data) - 1)]["content"][:-1]
    if prompt_style is None:
      prompt_style = self.data[randint(0, len(self.data) - 1)]["style"][:-1]

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

    k = randint(1, 3)
    selections = sample(content_modifier_options, k)
    content_modifier = ", ".join(selections)

    return (
      f"Continuous and smooth version of {prompt_content}, but with {content_modifier} everywhere."
      f"Use the style of {prompt_style}."
    )

  def gen_image(self, prompt, img_in, mask_in, n_images=1):
    output = self.pipe(
      prompt=prompt,
      negative_prompt="repetitive, distortion, glitch, borders, stretched, frames, breaks, multiple rows, gore, zombies, violence, splits, maps, diagrams, text, font, logos, branding",
      image=img_in,
      mask_image=mask_in,
      width=img_in.size[0], height=img_in.size[1],
      guidance_scale=16.0,
      num_inference_steps=24,
      num_images_per_prompt=n_images,
    )
    return output.images

  def prep_graft(self, limg, rimg, keep_width=256, right_offset=20, label="grafted"):
    orimg = rimg.crop((right_offset + keep_width, 0, rimg.size[0], rimg.size[1]))
    orimg.save(f"./imgs/{label}_new-right.jpg")

    rimg = rimg.crop((right_offset, 0, right_offset + keep_width, rimg.size[1]))
    description = get_img_description(limg)
    prompt_content = None
    prompt_style = description["style"][:-1]
    img_in, mask_in = LandscapeGenerator.get_input_images(limg, keep_width=keep_width, size=(4*keep_width, limg.size[1]), right_img=rimg)
    prompt = self.build_prompt(prompt_content=prompt_content, prompt_style=prompt_style)
    return prompt, img_in, mask_in

  def gen_landscape(self, keep_width=256, size=(1440, 512), n=4, label="mural", seed_img=None):
    makedirs(f"./imgs/{label}/", exist_ok=True)

    if seed_img is None:
      img_idx = randint(0, len(self.data) - 1)
      seed_img = self.data[img_idx]["image"]
      prompt_content = self.data[img_idx]["content"][:-1]
      prompt_style = self.data[img_idx]["style"][:-1]
      seed_img_id = self.data[img_idx]["article_id"]
    else:
      description = get_img_description(seed_img)
      prompt_content = None
      prompt_style = description["style"][:-1]
      seed_img_id = ""

    seed_img = LandscapeGenerator.resize_by_height(seed_img, size[1])
    seed_img.save(f"./imgs/{label}/{label}_00.jpg")

    landscape_imgs = [seed_img]
    landscape_ids = [seed_img_id]
    landscape_running_width = seed_img.size[0]

    img_in, mask_in = LandscapeGenerator.get_input_images(seed_img, keep_width=keep_width, size=size)
    prompt = self.build_prompt(prompt_content=prompt_content, prompt_style=prompt_style)

    for i in range(1, n+1):
      img_out_raw = self.gen_image(prompt, img_in, mask_in)[0]
      img_out_np = np.array(img_out_raw)[:, keep_width:]
      img_out = PImage.fromarray(img_out_np)

      landscape_imgs.append(img_out)
      landscape_running_width += img_out.size[0]
      img_out.save(f"./imgs/{label}/{label}_{('0'+str(i))[-2:]}.jpg")

      img_in, mask_in = LandscapeGenerator.get_input_images(img_out, keep_width=keep_width, size=size)

      prompt_rand = randint(0, 100)
      if prompt_rand < 30:
        landscape_ids.append(landscape_ids[-1])
        prompt = self.build_prompt(prompt_content=prompt_content, prompt_style=prompt_style)
      else:
        img_idx = randint(0, len(self.data) - 1)
        news_img = self.data[img_idx]["image"]
        prompt_content = self.data[img_idx]["content"][:-1]
        prompt_style = self.data[img_idx]["style"][:-1]
        landscape_ids.append(self.data[img_idx]["article_id"])
        prompt = self.build_prompt(prompt_content=prompt_content, prompt_style=prompt_style)

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
