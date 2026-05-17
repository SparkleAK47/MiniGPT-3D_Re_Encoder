评估配置文件 eval_configs/benchmark_evaluation_paper.yaml 中已经默认指向了官方权重路径（params_weight/MiniGPT_3D_stage_3 和 params_weight/MiniGPT_3D_stage_4）

export PYTHONPATH=$PWD
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/opt/miniconda3/envs/minigpt_3d/lib:$LD_LIBRARY_PATH

export WANDB_MODE=disabled
CUDA_VISIBLE_DEVICES=0 python train.py --cfg-path ./train_configs/MiniGPT_3D/stage_1.yaml > log_stage_1.txt
CUDA_VISIBLE_DEVICES=0 python train.py --cfg-path ./train_configs/MiniGPT_3D/stage_2.yaml > log_stage_2.txt
CUDA_VISIBLE_DEVICES=0 python train.py --cfg-path ./train_configs/MiniGPT_3D/stage_3.yaml > log_stage_3.txt
CUDA_VISIBLE_DEVICES=0 python train.py --cfg-path ./train_configs/MiniGPT_3D/stage_4.yaml > log_stage_4.txt
CUDA_VISIBLE_DEVICES=0 python train.py --cfg-path ./train_configs/MiniGPT_3D/stage_5.yaml > log_stage_5.txt

python UI_demo.py --cfg-path ./eval_configs/MiniGPT_3D_conv_UI_demo.yaml --gpu-id 0

# Prompt 0
CUDA_VISIBLE_DEVICES=0 python pointllm/eval/eval_objaverse.py \
    --out_path ./output/hybrid \
    --task_type classification \
    --cfg-path ./eval_configs/benchmark_evaluation_paper.yaml \
    --prompt_index 0

# Prompt 1
CUDA_VISIBLE_DEVICES=0 python pointllm/eval/eval_objaverse.py \
    --out_path ./output/hybrid \
    --task_type classification \
    --cfg-path ./eval_configs/benchmark_evaluation_paper.yaml \
    --prompt_index 1

# Prompt 0
CUDA_VISIBLE_DEVICES=0 python pointllm/eval/eval_modelnet_cls.py \
    --out_path ./output/hybrid \
    --cfg-path ./eval_configs/benchmark_evaluation_paper.yaml \
    --prompt_index 0

# Prompt 1
CUDA_VISIBLE_DEVICES=0 python pointllm/eval/eval_modelnet_cls.py \
    --out_path ./output/hybrid \
    --cfg-path ./eval_configs/benchmark_evaluation_paper.yaml \
    --prompt_index 1

CUDA_VISIBLE_DEVICES=0 python pointllm/eval/eval_objaverse.py \
    --out_path ./output/hybrid \
    --task_type captioning \
    --cfg-path ./eval_configs/benchmark_evaluation_paper.yaml \
    --prompt_index 2



CUDA_VISIBLE_DEVICES=0 python pointllm/eval/traditional_evaluator.py \
    --results_path ./output/hybrid/evaluation/PointLLM_brief_description_val_200_GT_Objaverse_captioning_prompt2.json
    
CUDA_VISIBLE_DEVICES=0 python pointllm/eval/traditional_evaluator.py \
    --results_path ./output/hybrid/evaluation/PointLLM_brief_description_val_200_GT_Objaverse_classification_prompt1.json
    
CUDA_VISIBLE_DEVICES=0 python pointllm/eval/traditional_evaluator.py \
    --results_path ./output/hybrid/evaluation/PointLLM_brief_description_val_200_GT_Objaverse_classification_prompt0.json
    
    


# API

# python ./pointllm/eval/evaluator_opensource_llm_QwenAPI.py \
#     --results_path ./output/official/evaluation/PointLLM_brief_description_val_200_GT_Objaverse_classification_prompt0.json \
#     --eval_type open-free-form-classification \
#     --model_type qwen2-72b-instruct \
#     --parallel --num_workers 4

export PYTHONPATH=$PWD

python ./pointllm/eval/evaluator_opensource_llm_QwenAPI.py  \
        --results_path ./output/hybrid/evaluation/PointLLM_brief_description_val_200_GT_Objaverse_classification_prompt0.json  \
        --eval_type open-free-form-classification  \
        --model_type qwen-flash \
        --parallel --num_workers 4


python ./pointllm/eval/evaluator_opensource_llm_QwenAPI.py  \
        --results_path ./output/hybrid/evaluation/PointLLM_brief_description_val_200_GT_Objaverse_classification_prompt1.json  \
        --eval_type open-free-form-classification  \
        --model_type qwen-flash \
        --parallel --num_workers 4

python ./pointllm/eval/evaluator_opensource_llm_QwenAPI.py  \
    --results_path ./output/hybrid/evaluation/ModelNet_classification_prompt0.json  \
    --eval_type modelnet-close-set-classification  \
    --model_type qwen-flash \
    --parallel --num_workers 4

python ./pointllm/eval/evaluator_opensource_llm_QwenAPI.py  \
    --results_path ./output/hybrid/evaluation/ModelNet_classification_prompt1.json  \
    --eval_type modelnet-close-set-classification  \
    --model_type qwen-flash \
    --parallel --num_workers 4

python ./pointllm/eval/evaluator_opensource_llm_QwenAPI.py  \
        --results_path ./output/hybrid/evaluation/PointLLM_brief_description_val_200_GT_Objaverse_captioning_prompt2.json  \
        --eval_type object-captioning  \
        --model_type qwen-flash \
        --parallel --num_workers 4

        
/output/official/evaluation

python ./pointllm/eval/evaluator_opensource_llm_QwenAPI.py  \
        --results_path ./output/pcpmae/evaluation/PointLLM_brief_description_val_200_GT_Objaverse_classification_prompt0.json  \
        --eval_type open-free-form-classification  \
        --model_type qwen-flash \
        --parallel --num_workers 4

python ./pointllm/eval/evaluator_opensource_llm_QwenAPI.py  \
        --results_path ./output/pcpmae/evaluation/PointLLM_brief_description_val_200_GT_Objaverse_classification_prompt1.json  \
        --eval_type open-free-form-classification  \
        --model_type qwen-flash \
        --parallel --num_workers 4

python ./pointllm/eval/evaluator_opensource_llm_QwenAPI.py  \
    --results_path ./output/pcpmae/evaluation/ModelNet_classification_prompt0.json  \
    --eval_type modelnet-close-set-classification  \
    --model_type qwen-flash \
    --parallel --num_workers 4

python ./pointllm/eval/evaluator_opensource_llm_QwenAPI.py  \
    --results_path ./output/pcpmae/evaluation/ModelNet_classification_prompt1.json  \
    --eval_type modelnet-close-set-classification  \
    --model_type qwen-flash \
    --parallel --num_workers 4

python ./pointllm/eval/evaluator_opensource_llm_QwenAPI.py  \
        --results_path ./output/pcpmae/evaluation/PointLLM_brief_description_val_200_GT_Objaverse_captioning_prompt2.json  \
        --eval_type object-captioning  \
        --model_type qwen-flash \
        --parallel --num_workers 4