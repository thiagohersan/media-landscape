# !pip install transformers==4.49 diffusers==0.32.2

import base64
import gc
import io
import json
import numpy as np
import os
import requests

from PIL import Image as PImage

try:
  import torch
  from diffusers import AutoPipelineForInpainting, ControlNetModel
  from torch.cuda import empty_cache
except:
  print("no pytorch")

try:
  from env import GEMINI_API_KEY
except:
  GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


DEFAULT_IMAGE_DESCRIPTION_PROMPT = "Describe this image's style in a way that can I use as a prompt to generate similar images using generative diffusion models. Use 100 words or less. Only describe style, not specific objects. Start the description with 'an image that ...' "  
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

def get64(img):
  if type(img) == str:
    with open(img, "rb") as ifp:
      return base64.b64encode(ifp.read()).decode("utf-8")
  elif isinstance(img, PImage.Image):
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")
  else:
    return ""

def clear_from_gpu(pipe):
  pipe = pipe.to("cpu")
  del pipe
  gc.collect()
  empty_cache()

# "runwayml/stable-diffusion-inpainting",
# "stable-diffusion-v1-5/stable-diffusion-inpainting",
# "stabilityai/stable-diffusion-2-inpainting",

def get_pipeline(model):
  return AutoPipelineForInpainting.from_pretrained(
    model,
    torch_dtype=torch.float16,
    variant="fp16"
  ).to("cuda")

# KEEP_WIDTH = 256
# RESULT_SIZE = (1440, 512)
def resize_by_height(src, height):
  iw, ih = src.size
  nw = int(iw * height / ih)
  return src.resize((nw, height))

def create_mask(keep_width, size):
  img_in = PImage.new("L", size)
  iw, ih = size
  img_in_pxs = [(i % iw >= keep_width) * 255 for i in range(iw * ih)]
  img_in.putdata(img_in_pxs)
  return img_in.convert("RGB")

def get_input_images(img, keep_width, size):
  img_np = np.array(resize_by_height(img.convert("RGB"), height=size[1]))
  bgd_np = np.array(create_mask(keep_width=keep_width, size=size))
  mask = create_mask(keep_width=keep_width, size=size)

  bgd_np[:, :keep_width] = img_np[:, -keep_width:]
  img_in = PImage.fromarray(bgd_np)

  return img_in, mask

# "Describe this image's style in a way that can I use as a prompt to generate similar images using generative diffusion models. Use 100 words or less. Only describe style, not specific objects. Start the description with 'an image that ...' "
def build_prompt(prompt_text, img):
  return {
    "contents": [{
      "parts": [
        { "inline_data": { "mime_type":"image/jpeg", "data": get64(img) } },
        { "text": prompt_text }
      ]
    }],
    "generationConfig": { "temperature": 0.5 }
  }

def get_img_description(img):
  headers = {
    "x-goog-api-key": GEMINI_API_KEY,
    "Content-Type": "application/json",
  }
  post_data = build_prompt(DEFAULT_IMAGE_DESCRIPTION_PROMPT, img)
  res = requests.post(GEMINI_URL, headers=headers, json=post_data)
  res_obj = res.json()
  return res_obj["candidates"][0]["content"]["parts"][0]["text"]

