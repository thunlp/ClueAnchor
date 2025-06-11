torchrun --nproc_per_node 8 -m openmatch.driver.build_index \
 --output_dir  /path/to/save/wikipedia/embedding \
 --model_name_or_path /path/to/bge/model \
 --per_device_eval_batch_size 2048 \
 --corpus_path /path/to/wikipedia/corpus \
 --doc_template "<text>" \
 --q_max_len 64 \
 --p_max_len 256 \
 --max_inmem_docs 1000000