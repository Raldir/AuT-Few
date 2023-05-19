datasets="emotion cb sst5 amazon_counterfactual_en wsc wic rte sentevalcr anli-r1 anli-r2 anli-r3 enron_spam"
num_shot=$1
for dataset in $datasets
  do
    for seed in 1 1024 42 0 32
    do
        CUDA_VISIBLE_DEVICES=$2 python3.9 t_few/run_tfew.py \
        --experiment_name tfew_baseline \
        --seed ${seed} \
        --dataset ${dataset} \
        --num_shot ${num_shot} \
        --few_shot_seed ${seed} \
        --ignore_save_path \
        --no_validation \
        --paper_comparison \
        --gradient_checkpointing \
        --efficient_finetune ia3 \
        --balanced_sampling \
        --backbone google/flan-t5-large
    done
done
