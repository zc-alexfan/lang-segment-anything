from PIL import Image
from lang_sam import LangSAM
import numpy as np
from glob import glob
import os
import os.path as op
from tqdm import tqdm

text_prompt = "hand."
# text_prompt = "foreground."
model = LangSAM()
fnames = glob('../data/itw_*/images/*')

pbar = tqdm(fnames)
for fname in pbar:
    out_p = fname.replace('/images/', '/processed/masks/').replace('.jpg', '.png')
    pbar.set_description(out_p)
    os.makedirs(op.dirname(out_p), exist_ok=True)
    image_pil = Image.open(fname).convert("RGB")
    results = model.predict([image_pil], [text_prompt])
    mask_np = (results[0]['masks'][0, :, :]*255).astype(np.uint8)
    mask = Image.fromarray(mask_np)
    mask.save(out_p)
    
    out_p = out_p.replace('/masks/', '/mano/')
    os.makedirs(op.dirname(out_p), exist_ok=True)
    mask.save(out_p)
    