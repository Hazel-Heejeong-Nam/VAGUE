from ram import inference_ram as inference
from ram.models import ram_plus, ram, tag2text
import torch
from utils import VAGUE_Dataset
import os
import json
import glob
from tqdm import tqdm
# os.environ["CUDA_VISIBLE_DEVICES"]='3'
device = 'cuda:3' if torch.cuda.is_available() else 'cpu'

# with open('sample_with_bbox.json', 'r') as file:
#     data = json.load(file)
# sample_img_names = [d['name'] for d in data]    
    
    
# imglist = ['data/6k_vc/6k_vc_images/'+sample for sample in sample_img_names]
imglist = glob.glob('/mnt/vague/vcr_10k/*.jpg')
print(len(imglist))
dataset = VAGUE_Dataset(imglist)
loader = torch.utils.data.DataLoader(dataset, batch_size = 1, shuffle = False, num_workers = 4,pin_memory = True) 
# !wget https://huggingface.co/spaces/xinyu1205/recognize-anything/resolve/main/ram_swin_large_14m.pth
ram_model = ram(pretrained='models/ram_swin_large_14m.pth', image_size = 384, vit='swin_l').to(device)
ram_model.eval()

ram_tag_list = []
for data in tqdm(loader):
        img, imgname = data[0].to(device), data[1]
        pred_tag = inference(img, ram_model)[0].replace(' |', ',')
        ram_tag_list.append({'imgname': imgname, 'ram_tag': pred_tag})
        
with open('vcr_10k/ramtag.json', 'w') as file:
    json.dump(ram_tag_list, file, indent=4)