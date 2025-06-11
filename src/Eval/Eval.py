import argparse
import json
import jsonlines
import os
import re
import string
import sys
from tqdm import tqdm
import numpy as np
from rouge import Rouge
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

SPECIAL_TOKEN_LENGTH = 64

# Prompt templates
QA_template = 'Q: {}\nA:'
QA_COT_External_Prompt = """
Please think about the reasoning process in the mind and then provides the user with the answer based on the given background. 
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
You could perform thinking with decomposing, Understanding, Recalling, reflecting, brainstorming, verifying, refining, and revising.
You first need to determine whether the background contains information related to the problem. If not, please answer the question based on general knowledge.
"""

# ========== Utility Functions ==========

def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst

def load_file(input_fp):
    if input_fp.endswith(".json"):
        input_data = json.load(open(input_fp))
    else:
        input_data = load_jsonlines(input_fp)
    return input_data

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def _acc_score(prediction, ground_truth):
    pred_clean = clean_text(prediction)
    gt_clean = clean_text(ground_truth)
    return 1.0 if gt_clean in pred_clean else 0.0

def _rougel_score(prediction, ground_truth):
    if prediction is None:
        prediction = ""
    if ground_truth is None:
        ground_truth = ""
    rouge = Rouge()
    try:
        scores = rouge.get_scores(prediction, ground_truth, avg=True)
    except ValueError:
        return 0.0
    return scores["rouge-l"]["f"]

def extract_answer_content(answer_content):
    """Extract answer from <answer> tags or return the cleaned version."""
    match = re.search(r"<answer>(.*?)</answer>", answer_content, re.DOTALL)
    return match.group(1) if match else re.sub(r"<?answer>", "", answer_content)

def postprocess_output(text):
    """Remove special tokens and leading spaces."""
    return text.replace("</s>", "").lstrip()

# ========== Main Processing Functions ==========

def process_input_data(input_data, args, tokenizer):
    for item in input_data:      
        raw_input = QA_COT_External_Prompt + QA_template.format(item['question'])
        token_input = tokenizer([raw_input])
        input_length = len(token_input.input_ids[0])

        passage = item['passages'][:args.top_n] if len(item['passages']) >= args.top_n else item['passages']
        segments = [entry for entry in passage]
        aug_psg = '\n'.join(segments)
        token_aug_psg = tokenizer([aug_psg])
        token_aug_psg = token_aug_psg.input_ids[0][:args.max_length - SPECIAL_TOKEN_LENGTH - input_length]
        new_passage = tokenizer.decode(token_aug_psg, skip_special_tokens=True)
        item["instruction"] = 'Background:\n' + new_passage + '\n\n' + raw_input
    return input_data

def run_inference(prompts, model, tokenizer, sampling_params):
    inference_prompts = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inference_prompts.append(prompt)

    outputs = model.generate(inference_prompts, sampling_params)
    preds = []
    for pred in outputs:
        pred = pred.outputs[0].text.lstrip()
        preds.append(pred)
    postprocessed_preds = [postprocess_output(pred) for pred in preds]

    return postprocessed_preds, prompts

# ========== Main Script ==========

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the local model directory')
    parser.add_argument('--input_file', type=str, required=True, help='Path to input .jsonl test data file')
    parser.add_argument('--max_tokens', type=int, default=500, help='Max tokens to generate per sample')
    parser.add_argument('--max_length', type=int, default=4096, help='Max tokens length for input')
    parser.add_argument('--metric', type=str, default='accuracy', help="Evaluation metric, Options include 'accuracy', 'rouge'")
    parser.add_argument('--top_n', type=int, default=10, help='The number of top passages')
    parser.add_argument('--task', type=str, default=None, help='Name of the evaluation task')
    parser.add_argument('--batch_size', type=int, default=200, help='Batch size for inference')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save output file')
    parser.add_argument('--exp_name', type=str, default=None, help='Optional experiment name')
    args = parser.parse_args()

    print("The parameter configuration is as follows:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    if args.output_path!=None:
        output_path = os.path.join(args.output_path, args.exp_name)
        os.makedirs(output_path, exist_ok=True)
    else:
        output_path=None

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left",truncation_side="right",is_pretrained_tokenizer=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = LLM(
        model=args.model_path,
        trust_remote_code=True,
        dtype='bfloat16'
    )
                
    sampling_params = SamplingParams(
        n=1,
        best_of=1,
        presence_penalty=1.0,
        frequency_penalty=0.0,
        temperature=1.0,
        top_p=1.0,
        top_k=1,
        use_beam_search=False,
        length_penalty=1,
        early_stopping=False,
        stop=None,
        stop_token_ids=None,
        ignore_eos=False,
        max_tokens=args.max_tokens,
        logprobs=None,
        prompt_logprobs=None,
        skip_special_tokens=True,
    )

    input_data = load_file(args.input_file)
    input_data = process_input_data(input_data, args, tokenizer)

    final_results = []

    for idx in tqdm(range(0, len(input_data), args.batch_size)):
        batch = input_data[idx:idx + args.batch_size]
        processed_batch = [item['instruction'] for item in batch]
        preds, prompts = run_inference(processed_batch, model, tokenizer, sampling_params)

        for j, item in enumerate(batch):
            answer = item["answer"]
            pred = preds[j]
            if "</think>" in pred:
                think_content, answer_content = pred.split("</think>", 1)
            else:
                think_content = ""
                answer_content = pred
            item["output"] = extract_answer_content(answer_content)

            if args.metric == "accuracy":
                metric_result = _acc_score(item["output"], answer)
            elif args.metric == "rouge":
                metric_result = _rougel_score(item["output"], answer)

            item["metric_result"] = metric_result
            final_results.append(item)

    if output_path is not None:
        output_path = os.path.join(output_path, str(args.task)+'output.jsonl')
        with open(output_path, "w") as f:
            for item in final_results:
                json.dump(item, f)
                f.write("\n")

    print(f"Results saved to: {output_path}")
    print(f"overall result: {np.mean([item['metric_result'] for item in final_results]):.4f}")

if __name__ == "__main__":
    main()