
export DEBUG=1

reference_models="Qwen/Qwen1.5-7B-chat,meta-llama/Meta-Llama-3-8B-Instruct,mistralai/Mistral-Nemo-Instruct-2407"

python generate_for_alpaca_eval_test.py \
    --model="Qwen/Qwen1.5-7B-chat" \
    --output-path="outputs/Qwen1.5-7B-round-3_MoA-Lite.json" \
    --reference-models=${reference_models} \
    --layers 4\
    --num-proc 1

 alpaca_eval --model_outputs outputs/Qwen1.5-7B-round-3_MoA-Lite.json --reference_outputs alpaca_eval/results/model_outputs.json --output_path /home/wh174/Mixture_test/MoA/alpaca_eval/results

# model_outputs : A path to a json file for the outputs of the model to add to the leaderboard. Each dictionary should contain the keys instruction and output.
# annotators_config: This is the annotator to use. We recommend using weighted_alpaca_eval_gpt4_turbo ( default for AlpacaEval 2.0),
# reference_outputs: The outputs of the reference model. Same format as model_outputs. By default, this is gpt4_turbo for AlpacaEval 2.0
# output_path: Path for saving annotations and leaderboard.
