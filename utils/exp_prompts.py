def plain_mcq(utt, mcq):
    template = f"""
Select the option that best explains the underlying intention of the speaker's utterance based on the given image. 
Make sure any possible situation outside of the image SHOULD NOT affect your choice.
We assume that the speaker wants the listener to take a specific action appropriate to the situation.

Utterance : {utt}
[Choices]
{mcq}

Your answer : (Output only the letter among A,B,C and D)
    """
    return template


def text_mcq(utt, mcq):
    template = f"""
Select the option that best explains the intention of the speaker's utterance. 
We assume that the speaker wants the listener to take a specific action. 

Utterance : {utt}
[Choices]
{mcq}

Your answer : (Output only the letter among A,B,C and D)
    """
    return template


def caption_mcq(cap, utt, mcq):
    template = f"""
Select the option that best explains the underlying intention of the speaker's utterance based on the description of the scene. 
Make sure any possible situation outside of the scene SHOULD NOT affect your choice.
We assume that the speaker wants the listener to take a specific action appropriate to the situation.

Scene Description : {cap}
Utterance : {utt}
[Choices]
{mcq}

Your answer : (Output only the letter among A,B,C and D)
    """
    return template


def caption_icl_mcq(cap, utt, mcq, icl_caps, icl_indirects, icl_mcqs, icl_ans):
    template = f"""
Select the option that best explains the underlying intention of the speaker's utterance based on the description of the scene. 
Make sure any possible situation outside of the scene SHOULD NOT affect your choice.
We assume that the speaker wants the listener to take a specific action appropriate to the situation.

Ex1. Scene Description : {icl_caps[0]}
Utterance : {icl_indirects[0]}
[Choices]
{icl_mcqs[0]}
Your answer : {icl_ans[0]}

Ex2. Scene Description : {icl_caps[1]}
Utterance : {icl_indirects[1]}
[Choices]
{icl_mcqs[1]}
Your answer : {icl_ans[1]}

Ex3. Scene Description : {icl_caps[2]}
Utterance : {icl_indirects[2]}
[Choices]
{icl_mcqs[2]}
Your answer : {icl_ans[2]}

Scene Description : {cap}
Utterance : {utt}
[Choices]
{mcq}

Your answer : (Output only the letter among A,B,C and D)
    """
    return template


def zeroshot_cot_mcq(utt, mcq):
    template = f"""
Select the option that best explains the underlying intention of the speaker's utterance based on the given image. 
Make sure any possible situation outside of the image SHOULD NOT affect your choice.
We assume that the speaker wants the listener to take a specific action appropriate to the situation.
Also, explain reasoning process of your answer.

Utterance : {utt}
[Choices]
{mcq}

Your answer1 (reasoning) : (Output your reasoning process in 2~3 sentences, which starts with "Let's think step by step.")
Your answer2 (intention) : (Output only the letter among A,B,C and D)
    """
    return template


def zeroshot_cot_sm_mcq(cap, utt, mcq):
    template = f"""
Select the option that best explains the underlying intention of the speaker's utterance based on the description of the scene. 
Make sure any possible situation outside of the scene SHOULD NOT affect your choice.
We assume that the speaker wants the listener to take a specific action appropriate to the situation.
Also, explain reasoning process of your answer.

Scene Description : {cap}
Utterance : {utt}
[Choices]
{mcq}

Your answer1 (reasoning) : (Output your reasoning process in 2~3 sentences, which starts with "Let's think step by step.")
Your answer2 (intention) : (Output only the letter among A,B,C and D)
    """
    return template


def caption_icl_cot_mcq(
    cap, utt, mcq, icl_caps, icl_indirects, icl_mcqs, icl_reasonings, icl_ans
):
    template = f"""
Select the option that best explains the underlying intention of the speaker's utterance based on the description of the scene. 
Make sure any possible situation outside of the scene SHOULD NOT affect your choice.
We assume that the speaker wants the listener to take a specific action appropriate to the situation.

Ex1. Scene Description : {icl_caps[0]}
Utterance : {icl_indirects[0]}
[Choices]
{icl_mcqs[0]}
Your answer1 (reasoning) : {icl_reasonings[0]}
Your answer2 (intention) : {icl_ans[0]}

Ex2. Scene Description : {icl_caps[1]}
Utterance : {icl_indirects[1]}
[Choices]
{icl_mcqs[1]}
Your answer1 (reasoning) : {icl_reasonings[1]}
Your answer2 (intention) : {icl_ans[1]}

Ex3. Scene Description : {icl_caps[2]}
Utterance : {icl_indirects[2]}
[Choices]
{icl_mcqs[2]}
Your answer1 (reasoning) : {icl_reasonings[2]}
Your answer2 (intention) : {icl_ans[2]}

Scene Description : {cap}
Utterance : {utt}
[Choices]
{mcq}

Your answer1 (reasoning) : (Output your reasoning)
Your answer2 (intention) : (Output only the letter among A,B,C and D)
    """
    return template


def plain_da(utt, p):
    template = f"""
What do you think is the underlying intention of the speaker's utterance based on the given image?
Make sure any possible situation outside of the image SHOULD NOT affect your answer.
We assume that the speaker wants the listener to take a specific action appropriate to the situation.
Your answer SHOULD NOT exceed 15 words.

Utterance : {utt}

Your answer : (Start your sentence with "The speaker wants {p} to...")
    """
    return template


def text_da(utt, p):
    template = f"""
What do you think is the intention of the speaker's utterance?
We assume that the speaker wants the listener to take a specific action.
Your answer SHOULD NOT exceed 15 words.

Utterance : {utt}

Your answer : (Start your sentence with "The speaker wants {p} to...")
    """
    return template


def caption_da(cap, utt, p):
    template = f"""
What do you think is the underlying intention of the speaker's utterance based on the description of the scene?
Make sure any possible situation outside of the scene SHOULD NOT affect your answer.
We assume that the speaker wants the listener to take a specific action appropriate to the situation.
Your answer SHOULD NOT exceed 15 words.

Scene Description : {cap}
Utterance : {utt}

Your answer : (Start your sentence with "The speaker wants {p} to...")
    """
    return template


def caption_icl_da(cap, utt, p, icl_caps, icl_indirects, icl_ans):
    template = f"""
What do you think is the underlying intention of the speaker's utterance based on the description of the scene?
Make sure any possible situation outside of the scene SHOULD NOT affect your answer.
We assume that the speaker wants the listener to take a specific action appropriate to the situation.
Your answer SHOULD NOT exceed 15 words.

Ex1. Scene Description : {icl_caps[0]}
Utterance : {icl_indirects[0]}
Your answer : {icl_ans[0]}

Ex2. Scene Description : {icl_caps[1]}
Utterance : {icl_indirects[1]}
Your answer : {icl_ans[1]}

Ex3. Scene Description : {icl_caps[2]}
Utterance : {icl_indirects[2]}
Your answer : {icl_ans[2]}

Scene Description : {cap}
Utterance : {utt}

Your answer : (Start your sentence with "The speaker wants {p} to...")
    """
    return template


def zeroshot_cot_da(utt, p):
    template = f"""
What do you think is the underlying intention of the speaker's utterance based on the given image?
Make sure any possible situation outside of the image SHOULD NOT affect your answer.
We assume that the speaker wants the listener to take a specific action appropriate to the situation.
Also, explain reasoning process of your answer.

Utterance : {utt}

Your answer1 (reasoning) : (Output your reasoning process in 2~3 sentences, which starts with "Let's think step by step.")
Your answer2 (intention): (Start your sentence with "The speaker wants {p} to..." and do not exceed 15 words.)
    """
    return template


def zeroshot_cot_sm_da(cap, utt, p):
    template = f"""
Select the option that best explains the underlying intention of the speaker's utterance based on the description of the scene. 
Make sure any possible situation outside of the scene SHOULD NOT affect your choice.
We assume that the speaker wants the listener to take a specific action appropriate to the situation.
Also, explain reasoning process of your answer.

Scene Description : {cap}
Utterance : {utt}

Your answer1 (reasoning) : (Output your reasoning process in 2~3 sentences, which starts with "Let's think step by step.")
Your answer2 (intention): (Start your sentence with "The speaker wants {p} to..." and do not exceed 15 words.)
    """

    return template


def caption_icl_cot_da(cap, utt, p, icl_caps, icl_indirects, icl_reasonings, icl_ans):
    template = f"""
What do you think is the underlying intention of the speaker's utterance based on the description of the scene?
Make sure any possible situation outside of the scene SHOULD NOT affect your answer. 
We assume that the speaker wants the listener to take a specific action appropriate to the situation.
Also, explain reasoning process of your answer.

Ex1. Scene Description : {icl_caps[0]}
Utterance : {icl_indirects[0]}
Your answer1 (reasoning) : {icl_reasonings[0]}
Your answer2 (intention) : {icl_ans[0]}

Ex2. Scene Description : {icl_caps[1]}
Utterance : {icl_indirects[1]}
Your answer1 (reasoning) : {icl_reasonings[1]}
Your answer2 (intention) : {icl_ans[1]}

Ex3. Scene Description : {icl_caps[2]}
Utterance : {icl_indirects[2]}
Your answer1 (reasoning) : {icl_reasonings[2]}
Your answer2 (intention) : {icl_ans[2]}

Scene Description : {cap}
Utterance : {utt}

Your answer1 (reasoning) : (Output your reasoning process in 2~3 sentences.)
Your answer2 (intention): (Start your sentence with "The speaker wants {p} to..." and do not exceed 15 words.)
    """
    return template
