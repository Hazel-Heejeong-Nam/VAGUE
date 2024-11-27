import os
import pandas as pd
import numpy as np
import  json

objects = pd.read_csv('ramtag_physical_object.csv', header=None)
objects = list(objects[0].values.squeeze())

with open('vcr_10k/ramtag.json', 'r') as file:
    data = json.load(file)
    

ddict = []
namelist = []
for dict in data:
    image_name = dict['imgname']
    orig_entity = dict['ram_tag'].split(', ')
    candidates = []
    candidates.extend(entity for entity in orig_entity if entity in objects)

    ddict.append({'name': image_name, 'num_entity': len(candidates), 'entities' : candidates})
    namelist.append(image_name)
sorted_data= sorted(ddict, key=lambda x: x["num_entity"], reverse=True)
selected_4k = sorted_data[:4000]
nums_selected = [data["num_entity"] for data in selected_4k]
nums= [data["num_entity"] for data in sorted_data]
print('10k :',sum(nums)/len(nums))
print('4k :', sum(nums_selected)/ len(nums_selected))

with open('vcr_10k/10k_ramtag_refined.json', 'w') as file:
    json.dump(sorted_data, file, indent=4)
    
with open('vcr_10k/4k_ramtag_refined.json', 'w') as file:
    json.dump(selected_4k, file, indent=4)


image = [data["name"][0] for data in selected_4k]
with open('vcr_10k/4k_image_name.txt', 'w') as file:
    for string in image:
        file.write(string + '\n')