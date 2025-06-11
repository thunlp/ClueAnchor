gpu_vis=0,1,2,3,4,5,6,7
MASTER_PORT=2351
deepspeed  --include localhost:$gpu_vis --master_port $MASTER_PORT ../src/Knowledge_Reasoning_Optimization/Knowledge_preference_optimization.py \
    --model_name_or_path /Path/to/local/model/ \
    --train_data_path /Path/to/the/training/dataset/in/JSON/format \
    --eval_data_path /Path/to/the/dev/dataset/in/JSON/format \
    --max_length 2000 \
    --max_prompt_length 1900 \
    --output_dir /Directory/to/save/the/model/checkpoints/and/training/outputs/ \
    --save_steps 100 \
    --eval_steps 100 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 5e-5 \
    --evaluation_strategy steps \
    --logging_strategy steps \
    --logging_steps 10 \
    --logging_dir /Directory/to/save/the/training/logs/ \
    --bf16 True \
    --use_lora True \
    --num_train_epochs 1 \
    --top_n 10 \
    --llama_style True \
    --deepspeed config/ds_config_zero2.json > /Path/to/the/log/file/for/recording/training/progress/run.log 2>&1 &