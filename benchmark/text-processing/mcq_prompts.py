from openai import OpenAI
from IPython.display import Image, display, Audio, Markdown
import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def read_file_to_list(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    formatted_lines = [line.strip() for line in lines]
    return formatted_lines

def gen_answer(client, action, obj, direct, indirect ,p):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"""
                Context:
                Your job is to figure out the speaker's true intention based on the given prompt. Your generated response must
                keep these two criteria in mind:
                1. Your answer should include {action}(the action to execute) and {obj}(object of being manipulated).
                2. You should answer to this specific prompt : {direct}
                3. Your answer should not exceed 15 words and start sentence with 'The speaker wants'
                4. Additionally, the response must also be applicable to the following sentence : {indirect}
            """}
        ]
    )

    return response.choices[0].message.content


def gen_wrong_entity(client,image, prompt, interpretation):
    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": """
            You are given an indirect complaint of the situation portrayed in the image and a correct interpretation of the prompt.
            Your job is to intentionally come up with an incorrect interpretations that will serve as an incorrect choice for a multiple choice question.
            The incorrect interpretation should specifically be designed using an object that is reasonable but does not exist anywhere in the image.
            The object, although not in the image, should be one that is very highly be expected to be present in the scene and could be used to resolve the complaint.
            
            Your answer should not exceed 15 words and start sentence with 'The speaker wants'
        """},

        {"role": "user", "content": """
            Prompt: Hey person1, I guess we're all going to be sharing more than just food today.
            Interpretation: The speaker wants person1 to use a fork to pick the food from the platter.
        """},

        {"role": "assistant", "content": """
            The speaker wants person1 to make use of the serving spoon when picking up shared food.
        """},

        {"role": "user", "content": """
            Prompt: Hey person1, I guess we're all going to be sharing more than just food today.
            Interpretation: The speaker wants person1 to use a fork to pick the food from the platter.
        """},

        {"role": "assistant", "content": """
            The speaker wants person1 to use the plastic gloves to avoid sharing germs.
        """},

        {"role": "user", "content": [
        {"type": "text", "text": f"""
            Prompt: {prompt}
            Interpretation: {interpretation}
        """},

        {"type": "image_url", "image_url": {
            "url": f"data:image/png;base64,{image}"}
            }
        ]}
    ],
    temperature=0.0,
    )

    return response.choices[0].message.content


def fake_caption(client,direct, indirect, p):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": f"""
                Context:
                Your job is to generate scene description in 2 sentences.
                Your generated prompt must keep these two criteria in mind:
                
                1. You should include '{p}' in the description.
                2. This sentence needs to be a line that fits with the scene description you create : "{indirect}"
                3. While following 2, your generated sentence must not be suitable to the utterance : "{direct}"
            """}
        ]
    )
    return response.choices[0].message.content


def gen_fake_answer(client,indirect, icls, fake_caption, answer, p):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"""
                Context:
                Your job is to guess underlying intention of the speaker when the situation is given. 
                Your generated response must keep these 5 criteria in mind:
                    1. Current situation : {fake_caption}
                    2. You should answer to this specific prompt : {indirect}
                    3. Your answer should not exceed 15 words and start sentence with 'The speaker wants {p} to'
                    4. Your answer should talk about one physical object in that given situation.
                    5. Your anwer SHOULD NOT HAVE THE SAME MEANING to "{answer}"
            """},
            {"role": "system", "content": f"""
                Example:
                    a. {icls[0]}
                    b. {icls[1]}
                    c. {icls[2]}
            """},
        ]
    )

    return response.choices[0].message.content


def summarize_indirect(client,indirect):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": f"""
                Context:
                Your job is to simplify the given sentence into a fixed form. 
                
                Example 1
                Given sentence : "Hey person 1, it's like a mini tornado hit those shelves, don't you think?"
                Your answer : The speaker is talking about mini tornado.
                
                Example 2
                Given sentence : "Hey person1, keep person2's privacy for her."
                Your answer : The speaker is talking about privacy.
                
                Given sentence : "{indirect}"
                Your answer : The speaker is talking about ( ONLY ONE WORD HERE )
                
                Please output only the sentence which starts with "The speaker ~".
            """}
        ]
    )
    return response.choices[0].message.content

def gen_surface_understanding1(client,caption, summarized_indirect, p, direct):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": f"""
                Context:
                Your job is to guess underlying intention of the speaker when the situation is given.
                Your generated response must keep these criteria in mind:
                    1. Current situation : {caption}
                    2. {summarized_indirect}
                    3. Your answer SHOULD NOT have similar meaning to "{direct}"
                    4. Your answer should not exceed 15 words and start sentence with 'The speaker wants {p} to'
                    
                Again, your answer SHOULD NOT HAVE SIMILAR MEANING to "{direct}"

            """}
        ]
    )
    return response.choices[0].message.content


def gen_surface_understanding2(client,direct, prompt, caption, p):
    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": f"""
            Your job is to generate incorrect interpretations of indirect complaints based on surface-level understanding.
            You will be given a direct complaint, an indirect complaint, and a scene description.
            Focus only on the literal wording of the indirect complaint and the observable elements in the scene. Do not consider any implied or deeper meaning from the indirect complaint.

            The incorrect interpretation must:
            1. Be reasonable within the context of the scene (based on the caption), without being totally nonsensical.
            2. Avoid being closely similar in meaning to the direct complaint. 
            3. Suggest the speaker wants something done, based on a surface understanding of the complaint.
            4. Begin with "The speaker wants {p} to..."
            
            Examples:
            Direct: Hey, person1, please close the fridge door.
            Indirect: Do you feel a sudden breeze in here, or is that just me?
            Caption: In this kitchen scene, person1 is bent over, looking inside the fridge, while person2 stands nearby. The kitchen is equipped with wooden cabinets, a microwave, and various other appliances.
            Interpretation: The speaker wants person1 to change the air conditioning to function at a higher temperature.

            Direct: Hey, person 3, please put out your cigarette.
            Indirect: I wonder why the view here suddenly got a little hazy.
            Caption: In a dimly lit bar, person 1 sits on a stool, looking off into the distance. Person 3, dressed in a business suit and fedora, is lighting a cigarette. The bar is stocked with various bottles and a cash register is visible in the background.
            Interpretation: The speaker wants person3 to adjust the lighting in the bar to be brighter.
            
            Your answer should not exceed 15 words.
        """},

        {"role": "user", "content": f"""
            Direct: {direct}
            Indirect: {prompt}
            Caption: {caption}
        """}
    ],
    temperature=0.0,
    )

    return response.choices[0].message.content
