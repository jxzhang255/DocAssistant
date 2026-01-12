CUDA_VISIBLE_DEVICES=0 swift sft \
    --model_type mini-internvl-chat-4b-v1_5 \
    --dataset /home/jxzhang/datasets/DUE_Benchmark/DUE_jsons/docvqa_rationales_new_llava.json \
    --max_length 4096