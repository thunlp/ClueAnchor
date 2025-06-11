import argparse
import random
import json
import re
import string
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
SPECIAL_TOKEN_LENGTH = 64

random.seed(123)

augment_template = 'Background:\n{}\n\n{}'
QA_template = 'Question: {}\nAnswer:'
Clue_Extract_template = 'Question: {}\nAnswer: {}\nExtracted supporting content:'
Clue_Prompt = """
You are given a question and its correct answer. Carefully read the passage and extract the sentence or sentences that are most relevant to supporting the answer.

Instructions:
1.Only extract content that appears explicitly in the passage.
2.Do not include any reasoning, explanation or inferred information.
3.Output must be faithful to the original wording in the passage, with no paraphrasing or modification.
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

def process_data(data, tokenizer, args):
    processed_data = []
    for item in data:
        raw_input_clue = Clue_Prompt + Clue_Extract_template.format(item['question'], item['answer'])
        token_input = tokenizer([raw_input_clue])
        input_length = len(token_input.input_ids[0])

        passage = item['passages'][:args.top_n] if len(item['passages']) >= args.top_n else item['passages']
        segments = [entry for entry in passage]
        aug_psg = '\n'.join(segments) 
        token_aug_psg = tokenizer([aug_psg])
        token_aug_psg = token_aug_psg.input_ids[0][:args.max_length - SPECIAL_TOKEN_LENGTH - input_length]
        new_psg = tokenizer.decode(token_aug_psg,skip_special_tokens=True)

        raw_input_clue = Clue_Prompt + augment_template.format(new_psg, Clue_Extract_template.format(item['question'], item['answer']))
        augment_input_clue = [{"role": "user", "content": raw_input_clue}]
        augment_input_clue = tokenizer.apply_chat_template(
            augment_input_clue, add_generation_prompt=True, tokenize=False
        )
        item['augment_input_clue'] = augment_input_clue
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
    processed_data = process_data(data, tokenizer, args)

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
        batch_prompts_clue = [item["augment_input_clue"] for item in batch]
        batch_responses_clue = batch_generate(llm, batch_prompts_clue, sampling_params)

        verify_batch_inputs = []
        verify_batch_items = []

        for item, response_clue in zip(batch, batch_responses_clue):
            answer = item["answer"]
            item["clue_extraction"] = response_clue
            clue_pointer_score = _acc_score(response_clue, answer)
            item["clue_score"] = clue_pointer_score

            raw_verify_prompt = augment_template.format(item["clue_extraction"], QA_template.format(item['question']))
            verify_prompt = [{"role": "user", "content": raw_verify_prompt}]
            verify_prompt = tokenizer.apply_chat_template(
                verify_prompt, add_generation_prompt=True, tokenize=False
            )

            verify_batch_inputs.append(verify_prompt)
            verify_batch_items.append(item)

        batch_responses_verify = batch_generate(llm, verify_batch_inputs, sampling_params)

        for item, response_verify in zip(verify_batch_items, batch_responses_verify):
            answer = item["answer"]
            item["verify_answer"] = response_verify
            verify_score = _acc_score(response_verify, answer)
            item["verify_score"] = verify_score

            output_data.append(item)
   
    write_jsonl(args.output_path, output_data)
    print(f"Clue Extration complete. Output written to: {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Internal Knowledge Reasoning with VLLM")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the local model directory')
    parser.add_argument('--input_path', type=str, required=True, help='Path to input .jsonl file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save output .jsonl file')
    parser.add_argument('--tensor_parallel_size', type=int, default=4, help='Tensor parallel size for VLLM')
    parser.add_argument('--top_n', type=int, default=5, help='The number of top passages')
    parser.add_argument('--batch_size', type=int, default=200, help='Batch size for inference')
    parser.add_argument('--max_tokens', type=int, default=100, help='Max tokens to generate per sample')
    parser.add_argument('--max_length', type=int, default=4096, help='Max tokens length for input')
    
    args = parser.parse_args()
    main(args)
