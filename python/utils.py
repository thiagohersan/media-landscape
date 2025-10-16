# !pip install transformers==4.49 diffusers==0.32.2

import base64
import gc
import io
import json
import os
import pandas as pd
import requests
import string

from PIL import Image as PImage
from sklearn.feature_extraction.text import CountVectorizer

try:
  from torch.cuda import empty_cache
except:
  print("no pytorch")

try:
  from env import GEMINI_API_KEY, NEWSDATA_API_KEY
except:
  GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
  NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY")


GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

DEFAULT_IMAGE_DESCRIPTION_PROMPT_00 = (
  "Describe this image's style in a way that I can use as a prompt to generate "
  "similar images using generative diffusion models. "
  "Use 48 words or less. Only describe style, not specific objects. "
  "Start the description with 'an image that depicts ...'"
)

DEFAULT_IMAGE_DESCRIPTION_PROMPT = (
  "Describe this image's style and content separately but in a way that I can use them "
  "as prompts to generate similar images using generative diffusion models. "
  "Use 24 words or less for each description. "
  "When describing style, don't describe specific objects; "
  "and when describing content, focus on objects and subjects and don't describe style. "
  "Start both descriptions with 'an image that depicts ...'."
  "If the image shows a logo, a map or a diagram, or is just text, use the word 'logo' in the content description."
  "If the image shows a weather report, weatherman or weather woman, use the term 'graphic overlay' in the content description."
)

NEWSDATA_URL = "https://newsdata.io/api/1/latest"
NEWSDATA_EXCLUDE_FIELDS = [
  "link",
  "source_url",
  "source_icon",
  "creator",
  "video_url",
  "pubDateTZ",
  "content",
  "country",
  "language",
  "pubDateTZ",
  "sentiment",
  "sentiment_stats",
  "ai_tag",
  "ai_region",
  "ai_org",
  "ai_summary",
  "ai_content",
]

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

def get_img(url):
  try:
    res = requests.get(url, timeout=10)
    res.raise_for_status()
    return PImage.open(io.BytesIO(res.content)).convert("RGB")
  except:
    return ""

def clear_from_gpu(pipe):
  pipe = pipe.to("cpu")
  del pipe
  gc.collect()
  empty_cache()

def build_description_prompt(prompt_text, img):
  res_schema = {
    "type": "object",
    "properties": {
      "style": { "type": "string" },
      "content": { "type": "string" },
    },
    "required": ["style", "content"]
  }

  return {
    "contents": [{
      "parts": [
        { "inline_data": { "mime_type":"image/jpeg", "data": get64(img) } },
        { "text": prompt_text }
      ]
    }],
    "generationConfig": {
      "responseMimeType": "application/json",
      "responseSchema": res_schema,
      "temperature": 0.25
    }
  }

def get_img_description(img):
  headers = {
    "x-goog-api-key": GEMINI_API_KEY,
    "Content-Type": "application/json",
  }
  post_data = build_description_prompt(DEFAULT_IMAGE_DESCRIPTION_PROMPT, img)
  res = requests.post(GEMINI_URL, headers=headers, json=post_data)
  res_obj = res.json()

  if "candidates" in res_obj:
    return json.loads(res_obj["candidates"][0]["content"]["parts"][0]["text"])
  else:
    print(res_obj)
    return {"content": "", "style": ""}

def get_articles(*, q=None, cat=None, n_articles=10):
  news_params = {
    "apikey": NEWSDATA_API_KEY,
    "language": "en",
    "image": 1,
    "excludefield": ",".join(NEWSDATA_EXCLUDE_FIELDS),
    "removeduplicate": 1
  }

  if q:
    news_params["q"]= q

  if cat:
    news_params["category"] = cat
  else:
    news_params["category"] = "top"
    # news_params["excludecategory"] = "food,lifestyle,sports,tourism,business"

  results = []
  n_queries = n_articles//10 if n_articles%10==0 else n_articles//10+1
  for idx in range(n_queries):
    res = requests.get(NEWSDATA_URL, params=news_params)
    res_obj = res.json()
    if res_obj["status"] == "success":
      results += res_obj["results"]
      news_params["page"] = res_obj["nextPage"]
    else:
      raise Exception("Error in NewsData API", res_obj)

  results_unique = list({x["article_id"]: x for x in results}.values())
  return results_unique[:n_articles]

def clean_text(txt):
  if type(txt) == list:
    txt = " ".join(txt)
  txt = txt.lower()
  txt = txt.replace("/", " ")
  txt = txt.replace("2025", "")
  txt = txt.replace("news", "")
  txt = txt.replace("new", "")
  txt = txt.replace("said", "")
  txt = txt.strip()
  txt = txt.translate(str.maketrans('', '', string.punctuation))
  return txt

def get_articles_with_top_words(articles, *, n_words, n_articles):
  titles = [clean_text(x["title"]) if x["title"] else " " for x in articles]
  keywords = [clean_text(x["keywords"]) if x["keywords"] else " " for x in articles]
  descriptions = [clean_text(x["description"]) if x["description"] else " " for x in articles]

  txt = [f"{t} {k} {d}" for t,k,d in zip(titles, keywords, descriptions)]
  
  mCV = CountVectorizer(stop_words="english", min_df=5, max_df=0.95, max_features=10_000)

  cnt_vct = mCV.fit_transform(txt)
  word_counts = cnt_vct.sum(axis=0).A.reshape(-1)
  word_idxs_by_cnt = (-word_counts).argsort()

  vocab = mCV.get_feature_names_out()
  top_words = vocab[word_idxs_by_cnt[:n_words]]

  article_top_word_cnts = cnt_vct[:, word_idxs_by_cnt[:n_words]].toarray()
  article_idxs_by_top_word_count = (-article_top_word_cnts).argsort(axis=0)[:n_articles]

  return pd.DataFrame(article_idxs_by_top_word_count, columns=top_words)

def get_article_images_by_size(articles, idxs, limit=None):
  imgs = [{"image": get_img(articles[idx]["image_url"]), "idx": idx} for idx in list(set(idxs))]
  imgs = [x for x in imgs if type(x["image"]) != str]
  imgs_by_size = sorted(imgs, key=lambda x: x["image"].size[0]*x["image"].size[1], reverse=True)
  if limit:
    return imgs_by_size[:limit]
  return imgs_by_size
