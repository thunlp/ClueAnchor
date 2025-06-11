import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,TrainingArguments)
from functools import partial
import logging
import os
from trl import DPOTrainer, DPOConfig
import transformers
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset
import torch
import transformers
import random

logger = logging.getLogger(__name__)
random.seed(42)

SPECIAL_TOKEN_LENGTH = 64
augment_template = 'Background:\n{}\n\n{}'
QA_template = 'Question: {}\nAnswer:'
QA_COT_Prompt = """
Please think about the reasoning process in the mind and then provides the user with the answer based on the given background. 
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
You could perform thinking with decomposing, Understanding, Recalling, reflecting, brainstorming, verifying, refining, and revising.
You first need to determine whether the background contains information related to the problem. If not, please answer the question based on general knowledge.
"""

@dataclass
class ModelArguments:
    # Arguments related to the model configuration
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the pre-trained model or identifier from the HuggingFace model hub."},
    )
    use_template: bool = field(
        default=True,
        metadata={"help": "Whether to use a predefined template for generating prompts."},
    )

@dataclass
class DataArguments:
    # Arguments related to the dataset
    train_data_path: str = field(
        default=None,
        metadata={"help": "Path to the training dataset in JSON format."},
    )
    eval_data_path: str = field(
        default=None,
        metadata={"help": "Path to the evaluation dataset in JSON format."},
    )
    max_length: int = field(
        default=2000,
        metadata={"help": "Maximum length of the sequences (prompt + completion) in the batch"},
    )
    max_prompt_length: int = field(
        default=1900,
        metadata={"help": "Maximum length of the prompt."},
    )
    top_n: int = field(
        default=5,
        metadata={"help": "Number of top passages to use for retrieval augmentation."},
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # Custom Training Arguments for fine-tuning the model, inherited from transformers
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory to store the cache files for the model and tokenizer."},
    )
    optim: str = field(
        default="adamw_torch",
        metadata={"help": "Optimizer to use during training. Options include 'adamw_torch', 'sgd', etc."},
    )
    output_dir: str = field(
        default=None,
        metadata={"help": "Directory to save the model checkpoints and training outputs."},
    )
    save_steps: int = field(
        default=100,
        metadata={"help": "Number of steps between saving model checkpoints."},
    )
    eval_steps: int = field(
        default=100,
        metadata={"help": "Number of steps between evaluation runs during training."},
    )
    per_device_train_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size for training on each device."},
    )
    per_device_eval_batch_size: int = field(
        default=1, 
        metadata={"help": "Batch size per GPU/TPU/MPS/NPU core/CPU for evaluation."}
    )
    learning_rate: float = field(
        default=5e-5, 
        metadata={"help": "The initial learning rate for AdamW."}
    )
    evaluation_strategy: str = field(
        default='steps',
        metadata={"help": "Evaluation strategy to use during training. Options include 'steps' or 'epoch'."},
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "Number of steps between logging training metrics."},
    )
    logging_dir: str = field(
        default=None,
        metadata={"help": "Directory to save the training logs."},
    )
    bf16: bool = field(
        default=True,
        metadata={"help": "Whether to use bfloat16 (bf16) precision for training."},
    )
    num_train_epochs: int = field(
        default=1, 
        metadata={"help": "Total number of training epochs to perform."},
    )

def load_model_and_tokenizer(
    model_path: str,
    use_lora: bool = True,
    bf16: bool = False,
    fp16: bool = False,
):
    """load model and tokenizer"""
    assert not (bf16 and fp16), "bf16 or fp16, not both"
    if bf16:
        dtype = torch.bfloat16
    elif fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    if use_lora:
        from peft import LoraConfig, TaskType, get_peft_model
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            inference_mode=False,
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    return model, tokenizer

def preprocessing(example,args,tokenizer):
    """Preprocessing function to format input for the model."""
    one_item = {}
    raw_input = QA_COT_Prompt + QA_template.format(example['question'])
    token_input = tokenizer([raw_input])
    input_length = len(token_input.input_ids[0])

    passage = example['passages'][:args.top_n] if len(example['passages']) >= args.top_n else example['passages']
    segments = [entry for entry in passage]
    aug_psg = '\n'.join(segments) 
    token_aug_psg = tokenizer([aug_psg])
    token_aug_psg = token_aug_psg.input_ids[0][:args.max_length - SPECIAL_TOKEN_LENGTH - input_length]
    new_psg = tokenizer.decode(token_aug_psg,skip_special_tokens=True)

    raw_input = QA_COT_Prompt + augment_template.format(new_psg, QA_template.format(example['question']))
    augment_input = [{"role": "user", "content": raw_input}]
    augment_input = tokenizer.apply_chat_template(
        augment_input, add_generation_prompt=True, tokenize=False
    )
    chosen_text = "<think> " + example['chosen_think'] + " </think> " + "<answer> " + example['chosen_answer'] + " </answer>"
    rejext_text = "<think> " + example['reject_think'] + " </think> " + "<answer> " + example['reject_answer'] + " </answer>"

    one_item["prompt"] = augment_input
    one_item["chosen"] = chosen_text
    one_item["rejected"] = rejext_text
    return one_item


def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)
    logger.info("DATA parameters %s", data_args)

    model, tokenizer = load_model_and_tokenizer(
        model_path=model_args.model_name_or_path,
        use_lora=training_args.use_lora,
        bf16=training_args.bf16,
        fp16=training_args.fp16,
    )
    
    partial_preprocess = partial(preprocessing,args=data_args,tokenizer=tokenizer)

    train_dataset = load_dataset("json", data_files=data_args.train_data_path,split="train")
    train_dataset = train_dataset.map(partial_preprocess)

    eval_dataset = load_dataset("json", data_files=data_args.eval_data_path,split="train")
    eval_dataset = eval_dataset.map(partial_preprocess)

    dpo_training_args = DPOConfig(
        optim = training_args.optim,
        output_dir = training_args.output_dir,
        save_steps = training_args.save_steps,
        eval_steps = training_args.eval_steps,
        per_device_train_batch_size = training_args.per_device_train_batch_size,
        evaluation_strategy = training_args.evaluation_strategy,
        logging_steps = training_args.logging_steps,
        logging_dir = training_args.logging_dir,
        learning_rate = training_args.learning_rate,
        bf16 = training_args.bf16,
        num_train_epochs = training_args.num_train_epochs,
        max_length = data_args.max_length, 
        max_prompt_length = data_args.max_prompt_length,
    )

    dpo_trainer = DPOTrainer(
        model,
        ref_model=None,
        args=dpo_training_args,
        beta=0.1,
        train_dataset=train_dataset,
        eval_dataset =eval_dataset,
        tokenizer=tokenizer,
    )
    dpo_trainer.train()
    dpo_trainer.save_model()

if __name__ == '__main__':
    main()