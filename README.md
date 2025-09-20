# [ICCV 2025] VAGUE: Visual Contexts Clarify Ambiguous Expressions

**Authors**: Heejeong Nam*, Jinwoo Ahn*, Keummin Ka, Jiwan Chung, Youngjae Yu  
**Project Page**: [https://hazel-heejeong-nam.github.io/vague/](https://hazel-heejeong-nam.github.io/vague/)
**Paper Link**: [https://arxiv.org/abs/2411.14137](https://arxiv.org/abs/2411.14137)

![Main Figure](figs/main.jpg)

---

## Dataset Download


**VAGUE 1.0 (2024) has been deprecated. Please use our new VAGUE 2.0 (2025 March ~) 😀**

🤗 HuggingFace Dataset : 
[HuggingFace Dataset Link](https://huggingface.co/datasets/HazelNam/vague-bench)

Also download the dataset using the following link:  
[Dataset Link](https://drive.google.com/drive/folders/1GIoMcmN59PqDoczwcrymqnZ6JY4jgf0r?usp=sharing)

| Feature                | ~~VAGUE 1.0~~ (deprecated) | VAGUE 2.0 |
|------------------------|--------------------------|-----------|
| **Size**              | 3993                     | 1677      |
| **Source**            | [VCR](https://visualcommonsense.com/) | [VCR](https://visualcommonsense.com/), [Ego4D](https://ego4d-data.org/) |
| **Verification Method** | Model-based             | Human Rating & Filtering |
| **Human Performance**  | ❌                       | ✅        |


---

## Inference

### Environments
All the open-source models we used are supported by `vllm`.

```bash
conda create -n vague python=3.12 -y
conda activate vague
pip install bert-score nltk # for eval

# Install vLLM with CUDA 12.1.
pip install vllm 

# Install vLLM with CUDA 11.8.
export VLLM_VERSION=0.6.1.post1
export PYTHON_VERSION=310
pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu118-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux1_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu118

```

## dataset example


### Json format
```
{
    "image_name": "0013_Halloween_00.15.15.492-00.15.17.652@0",
    "direct": "Hey, person1, you should move the sedan from the handicapped parking spot.",
    "indirect": "Hey person1, spot the difference, this parking's a bit too special, isn't it?  ",
    "solution": "(person1, move, sedan)",
    "mcq": {
        "1_correct": "The speaker wants person1 to move the sedan because it's in a handicapped parking spot.",
        "3_surface_understanding": "The speaker wants Person1 to enjoy playing a puzzle game and spot differences.",
        "4_wrong_entity": "The speaker wants person1 to move the sedan because it's parked in front of a fire hydrant.",
        "ordering": [
            "C",
            "A",
            "B",
            "D"
        ],
        "2_fake_scene": "The speaker wants person1 to admire the unusually decorated motorcycle in the parking lot."
    },
    "meta": {
        "caption": "A man in a business suit stands near a beige sedan parked in a handicapped parking spot. The area is surrounded by greenery and a building entrance is visible in the background.",
        "ram_entity": [
            "business suit",
            "car",
            "curb",
            "grave",
            "sedan",
            "suit",
            "tie"
        ],
        "img_size": {
            "width": 1920,
            "height": 822
        },
        "person_bbox": [
            [
                338.989990234375,
                112.2576904296875,
                578.5294799804688,
                717.6659545898438
            ],
            [
                1055.5440673828125,
                233.45152282714844,
                1131.0687255859375,
                288.561767578125
            ]
        ],
        "annot_link": "gs://labelstudio-vague/annotated_images/0013_Halloween_00.15.15.492-00.15.17.652@0_annot.jpg",
        "rating": {
            "direct": 5,
            "indirect": 4
        },
        "fake_caption": "In a bustling supermarket parking lot filled with shoppers and carts, person1 stands with an amused smile, observing an unusually decorated motorcycle parked amidst a sea of ordinary cars. \"Hey person1, spot the difference, this parking's a bit too special, isn't it?\""
    }
}
```

## Citation

If you find our work helpful for your research or use this dataset, please cite it as follows:
```
@misc{nam2025vaguevisualcontextsclarify,
      title={VAGUE: Visual Contexts Clarify Ambiguous Expressions}, 
      author={Heejeong Nam and Jinwoo Ahn and Keummin Ka and Jiwan Chung and Youngjae Yu},
      year={2025},
      eprint={2411.14137},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.14137}, 
}
```
