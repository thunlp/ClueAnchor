CUDA_VISIBLE_DEVICES=0,1,2,3 python ../src/Knowledge_Reasoning_Exploration/ClueAnchored_Knowledge_Reasoning.py \
  --model_path /path/to/local/model/ \
  --input_path /path/to/input_data.jsonl \
  --output_path /path/to/output_data.jsonl \
  --tensor_parallel_size 4 \
  --top_n 10 \
  --batch_size 200 \
  --max_length 4096 \
  --max_tokens 500
