import json
from PIL import Image, ImageDraw, ImageFont
font_size = 40
font = ImageFont.load_default(size=font_size)
import os
import glob

with open('vcr_10k/4k_ram_bbox.json', 'r') as file:
    data = json.load(file)
os.makedirs('vcr_10k/annotated_images', exist_ok=True)
    
for d in data:
    path = '/mnt/vague/vcr_10k/'+d['image_name'] +'.jpg'
    image = Image.open(path)
    positions = []
    for i, box in enumerate(d['person_bbox']):
        positions.append(((box[0]+box[2])/2, (box[1]+box[3])/2, str(i+1)))
    draw = ImageDraw.Draw(image)
    for pos in positions:
        x, y, number = pos
        text_size = draw.textlength(number, font=font)
        # print(text_size)
        rectangle_top_left = (x - text_size / 2 - 2, y - text_size / 2 - 2)
        rectangle_bottom_right = (x + text_size / 2 + 2, y + text_size  + 2)
        draw.rectangle([rectangle_top_left, rectangle_bottom_right], fill="black")
        
        text_position = (x - text_size/ 2, y - text_size / 2)
        draw.text(text_position, number, fill="white", font=font)


    output_path = 'vcr_10k/annotated_images/' +d['image_name'] +'_annot.jpg'
    image.save(output_path)
    print('saved')