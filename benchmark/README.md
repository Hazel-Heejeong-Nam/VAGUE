# VAGUE-benchmark 

- ramtag_physical_object.csv 이거 찾아서 옮겨둬야함 @hazel
- Original image source : VCR (110K)

### Image-processing
- Sparse sampling (10K) : @jinwoo 코드 어디에..
- Select images having complex scene by using ramtag (4k) : `image-processing/1_get_ramtag.py`, `image-processing/2_refine_ramtag.py`
- Preprocess images for VAGUE benchmark dataset : `image-processing/3_add_bbox.py`, `4_draw_number.py`

### Text-processing
- Generate direct prompt, solution, and caption : `text-processing/direct_prompt.py`
- Generate indirect prompt : `text-processing/indirect_prompt.ipynb`
- Generateing MCQ : ``
