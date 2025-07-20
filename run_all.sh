for dataset in mt-bench med-safe helmet implicit_hate mrpc #persuasion 
do
    for method in importance_sampling #control_variate uniform_sampling llm_only  control_variate control_variate_importance_sampling
    do
        echo "Running $method on $dataset"
        python methods/$method.py --dataset $dataset --step_size 50 --repeat 200 --max_human_labels 2000 --save_dir results/$method/
    done
done