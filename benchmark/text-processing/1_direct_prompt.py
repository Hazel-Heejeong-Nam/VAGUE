from openai import OpenAI
from IPython.display import Image, display, Audio, Markdown
import base64
import json

file = open("api.txt", "r")
content = file.read()
client = OpenAI(
    api_key=content
)
    
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
def mm_prompt_gen(image, entities):
    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": """
            Your job is to do two things. 

            1. generate a direct complaint based on the image. Your generated prompt must keep these three criteria in mind: 
            a. Specify the recipient: The speaker is the person who is viewing the scene. Specify the recipient as a person in the image (begins with "Hey, person1..."). There is a number tag in the image for each person. 
            b. Generate direct prompts: Your complaint must include the subject, action, and object: it should convey the "WHO should do WHAT action on WHAT object." 
            c. Solvable with Triplet: The object from the triplet--(subject, action, object)--must be a physical object from the provided "Entity" list. 

            2. generate a solution triplet that addresses the prompt. Your generated solution must keep these three criteria in mind: 
            a. Triplet: The format of your output must be in (subject, action, object). 
            b. Problem Mitigation: The generated solution must address the prompt in a way that resolves the complaint in the prompt. 
            """},

        {"role": "user", "content": [
        {"type": "text", "text": f"""
            Entity: {entities}
            Prompt: (One Statement) 
            Solution: (Subject, Action, Object)
            Caption: (2-3 Sentences Describing the Scene)
            """},

        {"type": "image_url", "image_url": {
            "url": f"data:image/png;base64,{image}"}
            }
        ]}
    ],
    temperature=0.0,
    )

    return response.choices[0].message.content


if __name__ == '__main__':

    file_path = '4k_ramtag_refined.json'

    with open(file_path, 'r') as file:
        data = json.load(file)

    import json

    output_data = []
    count = 1

    for item in data[100:1000]:
        print(f"Current at Item: {count}/900" )
        img = f"/mnt/vague/annotated_images/" + item['name'][0] + "_annot.jpg"
        ent = item['entities']
        enc_img = encode_image(img)
        res = mm_prompt_gen(enc_img, ent)

        lines = res.split('\n')

        prompt = ""
        solution = ""
        caption = ""

        for line in lines:
            if line.startswith("Prompt:"):
                prompt = line.replace("Prompt: ", "").strip()
            elif line.startswith("Solution:"):
                solution = line.replace("Solution: ", "").strip()
            elif line.startswith("Caption:"):
                caption = line.replace("Caption: ", "").strip()

        entry = {
            "image": img,
            "prompt": prompt,
            "solution": solution,
            "caption": caption
        }

        output_data.append(entry)

        count += 1

    with open('output.json', 'w') as json_file:
        json.dump(output_data, json_file, indent=4)
