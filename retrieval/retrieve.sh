python /openmatch/driver/retrieve.py \
 --output_dir /path/to/save/wikipedia/embedding/and/query/embedding \
 --model_name_or_path /path/to/bge/model \
 --per_device_eval_batch_size 512 \
 --query_path /path/to/query/data \
 --query_template "<question>" \
 --trec_save_path /path/to/save/trec \
 --q_max_len 128 \
 --retrieve_depth \
 --use_gpu True