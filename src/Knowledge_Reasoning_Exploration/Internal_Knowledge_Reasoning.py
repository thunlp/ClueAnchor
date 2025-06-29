import argparse
import random
import json
import re
import string
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

random.seed(123)

QA_template = 'Question: {}\nAnswer:'
QA_COT_Internal_Prompt = """
Please think about the reasoning process in your mind and then provide the user with the answer.
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
You could perform thinking with decomposing, understanding, recalling, reflecting, brainstorming, verifying, refining, and revising.
"""

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def _acc_score(prediction, ground_truth):
    pred_clean = clean_text(prediction)
    gt_clean = clean_text(ground_truth)
    return 1.0 if gt_clean in pred_clean else 0.0

def read_jsonl(input_path):
    with open(input_path, 'r') as f:
        return [json.loads(line.strip()) for line in f]

def write_jsonl(output_path, data):
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def process_data(data, tokenizer):
    processed_data = []
    for item in data:
        raw_input_internal = QA_COT_Internal_Prompt + QA_template.format(item['question'])
        augment_input_internal = [{"role": "user", "content": raw_input_internal}]
        augment_input_internal = tokenizer.apply_chat_template(
            augment_input_internal, add_generation_prompt=True, tokenize=False
        )
        item['augment_input_internal'] = augment_input_internal
        processed_data.append(item)
    return processed_data

def extract_answer_content(answer_content):
    match = re.search(r"<answer>(.*?)</answer>", answer_content, re.DOTALL)
    return match.group(1) if match else re.sub(r"<?answer>", "", answer_content)

def batch_generate(llm, prompts, sampling_params):
    responses = llm.generate(prompts, sampling_params)
    return [r.outputs[0].text for r in responses]

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    data = read_jsonl(args.input_path)
    processed_data = process_data(data, tokenizer)

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        dtype='bfloat16'
    )

    sampling_params = SamplingParams(
        n=1,
        best_of=1,
        presence_penalty=1.0,
        frequency_penalty=0.0,
        temperature=1.0,
        top_p=0.8,
        top_k=-1,
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

    output_data = []

    for i in tqdm(range(0, len(processed_data), args.batch_size)):
        batch = processed_data[i:i + args.batch_size]
        batch_prompts_internal = [item["augment_input_internal"] for item in batch]
        batch_responses_internal = batch_generate(llm, batch_prompts_internal, sampling_params)

        for item, response_internal in zip(batch, batch_responses_internal):
            answer = item["answer"]
            if "</think>" in response_internal:
                internal_think_content, internal_answer_content = response_internal.split("</think>", 1)
            else:
                internal_think_content = ""
                internal_answer_content = response_internal

            if "<think>" in internal_think_content:
                internal_think_content = internal_think_content.replace("<think>", "").strip()

            extracted_internal_content = extract_answer_content(internal_answer_content)
            item["internal_think"] = internal_think_content
            item["internal_answer"] = extracted_internal_content
            internal_score = _acc_score(extracted_internal_content, answer)
            item["internal_score"] = internal_score

            output_data.append(item)

    write_jsonl(args.output_path, output_data)
    print(f"Internal Knowledge Reasoning complete. Output written to: {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Internal Knowledge Reasoning with VLLM")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the local model directory')
    parser.add_argument('--input_path', type=str, required=True, help='Path to input .jsonl file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save output .jsonl file')
    parser.add_argument('--tensor_parallel_size', type=int, default=4, help='Tensor parallel size for VLLM')
    parser.add_argument('--batch_size', type=int, default=200, help='Batch size for inference')
    parser.add_argument('--max_tokens', type=int, default=500, help='Max tokens to generate per sample')

    args = parser.parse_args()
    main(args)
