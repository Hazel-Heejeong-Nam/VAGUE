import json
from PIL import Image, ImageDraw, ImageFont
font_size = 40
font = ImageFont.load_default(size=font_size)
import os
import glob
from tqdm import tqdm

def read_jsonl_to_dict(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data
#load vcr annotation and aggregate
train = read_jsonl_to_dict('data/train.jsonl')
val = read_jsonl_to_dict('data/val.jsonl')
test = read_jsonl_to_dict('data/test.jsonl')
data = train+val+test


os.makedirs('/mnt/vague/4k_numbered', exist_ok=True)
with open('vcr_10k/4k_ramtag_refined.json', 'r') as file:
    images = json.load(file)
    
    
for idx, image in tqdm(enumerate(images)):
    json_path = '/mnt/vague/vcr_10k/' + image['name'][0] + '.json'
    with open(json_path, 'r') as f:
        meta =json.load(f)
    bbox = []
    mask_rcnn_info = [d for d in data if d["img_fn"].split("/")[-1][:-4]== image['name'][0]][0]
    entities = mask_rcnn_info["objects"]
    for i, entity in enumerate(entities):
        if entity == 'person':
            bbox.append(meta['boxes'][i][:-1])
    
    images[idx]['person_bbox'] = bbox
    images[idx]['width'] = meta['width']
    images[idx]['height'] = meta['height']
    images[idx]['vcr_object'] = entities
    images[idx]['image_name'] = image['name'][0]
    del images[idx]['name']
    
with open('vcr_10k/4k_ram+bbox.json', 'w') as file:
    data = json.dump(images, file, indent=4)
