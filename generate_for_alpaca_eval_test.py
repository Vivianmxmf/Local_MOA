import json
import datasets
from fire import Fire
from loguru import logger
import asyncio
import torch
import transformers
import os
import gc
from huggingface_hub import login

# Ensure Hugging Face token is available for model loading
token = os.getenv('HUGGINGFACE_TOKEN')
if token:
    login(token=token)
else:
    raise ValueError("Hugging Face token not found in environment variables.")

# Define the model class for local inference
class LocalLLM:
    def __init__(self, model_id):
        self.model_id = model_id
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id, padding_side='left', device_map="auto")
        self.model = transformers.AutoModelForCausalLM.from_pretrained(self.model_id, device_map="auto", torch_dtype=torch.float16)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    async def get_response(self, query, max_length=1024, temperature=0.7):
        model_inputs = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True)
        model_inputs = {k: v.to("cuda:0") for k, v in model_inputs.items()}
        model_inputs.pop("token_type_ids", None)

        generated_ids = self.model.generate(**model_inputs, max_length=max_length, max_new_tokens=100, temperature=temperature)
        response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return response

# Proposed_layers
async def process_fn_ly(item, model, temperature=0.7, max_tokens=2048, layers=None):
    # Initial prompt
    if layers == 1:
        user_prompt = item["instruction"]
    else:
        user_prompt = item["concatenated_output"]
   
    result = await model.get_response(query=user_prompt, temperature=temperature, max_length=max_tokens)
    output_data = {
        "instruction": user_prompt,
        "dataset": item.get("dataset", "unknown"),  
        "output": result,
        "generator": model.model_id   
    }

    # Convert the output_data dictionary to a string with formatting similar to the one in the image
    output_str = json.dumps(output_data, indent=4)

    print(output_str)

    # Append the formatted output to the file
    with open(f"model_{model.model_id}_layer{layers}_response.txt", "a") as f:
        f.write(output_str + "\n") 


# Concatenate the proposed answers from a signle proposed_layer
async def interoutput_fn_ly(reference_models=None, layers=None):
    instructions_dict = {}
    datasets = {}

    for model in reference_models:
        if "Llama" in model:
            dir_name = "model_meta-llama"
        elif "Mistral" in model:
            dir_name = "model_mistralai"
        elif "Qwen" in model:
            dir_name = "model_Qwen"
    
        # Format the file name with directory and model information
        file_name = f"{dir_name}/{model}_layer{layers}_response.txt"
        try:
            with open(file_name, "r") as f:
                file_content = f.read()
                json_blocks = file_content.split("}\n{")

                # Reassemble the blocks into proper JSON strings
                for i in range(len(json_blocks)):
                    if i == 0:
                        json_blocks[i] += "}"
                    elif i == len(json_blocks) - 1:
                        json_blocks[i] = "{" + json_blocks[i]
                    else:
                        json_blocks[i] = "{" + json_blocks[i] + "}"

                    # Parse each JSON block
                    data = json.loads(json_blocks[i])

                    # Extract the instruction and output parts
                    instruction = data.get("instruction", "")
                    dataset = data.get("dataset", "")
                    output = data.get("output", "")

                    if instruction:
                        if instruction not in instructions_dict:
                            instructions_dict[instruction] = []
                        instructions_dict[instruction].append(output)
                    if instruction not in datasets:
                        datasets[instruction] = dataset

        except FileNotFoundError:
            print(f"File not found: {file_name}")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in file {file_name}: {e}")

    # Prepare the final JSON output structure
    output_list = []
    for instruction, outputs in instructions_dict.items():
        output_data = {
            "instruction": instruction,
            "dataset": datasets.get(instruction, "unknown"),
            "concatenated_output": " ".join(outputs).strip()  # Join outputs with a space separator
        }
        output_list.append(output_data)

    # Convert the output list to a JSON string
    output_str = json.dumps(output_list, indent=4)
    
    # Print the JSON output string (can be removed if not needed)
    print(output_str)
    
    # Save the concatenated outputs to a file
    with open(f"Interoutput_from_layer{layers}_response.json", "w") as f:
        f.write(output_str + "\n")
            

# The final aggregation layer
async def process_fn_agg(item, model, temperature=0.7, max_tokens=2048,output_path= None):
    user_prompt = """You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability. Responses from models:"""+item["instruction"]

    
    final_output = await model.get_response(query=user_prompt, max_length=max_tokens)

    output_data = {
        "instruction": user_prompt,
        "dataset": item.get("dataset", "unknown"), 
        "output": final_output,
        "generator": model.model_id + "-together",
    }
    logger.info(f"Saving outputs to {output_path}.")
    return output_data

async def main(
    model: str,
    output_path: str,
    reference_paths: str = None,
    reference_models: str = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    layers: int = None,
    num_proc: int = 16,
):

    # Handling Reference Paths and Models
    if reference_paths is None:
        reference_paths = []
    else:
        print(reference_paths)
        reference_paths = reference_paths.split(",")

    if reference_models is None:
        reference_models = []
    else:
        reference_models = reference_models.split(",")

    # Loading the Dataset
    eval_set = datasets.load_dataset(
        "tatsu-lab/alpaca_eval", "alpaca_eval_gpt4_baseline", trust_remote_code=True
    )["eval"]
    eval_set = eval_set.remove_columns(["output", "generator"])

    if len(reference_paths):

        logger.info(f"`reference_paths` provided: {reference_paths}")

        references = []
        for reference_path in reference_paths: #loops through each file in reference_paths.
            with open(reference_path) as f:
                reference_responses = json.load(f)
                logger.info(f"Reading reference outputs: {reference_path} ({len(reference_responses)})")
                for i_reference_response, reference_response in enumerate(reference_responses):
                    if len(references) <= i_reference_response:
                        references.append([reference_response["output"]])
                    else:
                        references[i_reference_response].append(reference_response["output"])

        eval_set = eval_set.add_column(f"references", references) # Add References to Dataset

    elif len(reference_models):
        logger.info(f"`reference_models` provided: {reference_models}. Will generate reference responses on-the-fly.")

   
    #Iterate over each reference model (layer 1)
    for reference_model in reference_models:

        agent = LocalLLM(reference_model)  # Load the agent once

        #Process all items in the dataset with the loaded agent
        results_layer_1 = await asyncio.gather(*[
            process_fn_ly(item, model=agent, temperature=temperature, max_tokens=max_tokens, layers=1)
            for item in eval_set
        ])
        del agent
        gc.collect()
    
    reference_models_list = ["Qwen1.5-7B-chat","Meta-Llama-3-8B-Instruct","Mistral-Nemo-Instruct-2407"]
    # Concatenate the proposed answers from first proposed layer
    await asyncio.gather (*[interoutput_fn_ly(reference_models= reference_models_list, layers= 1)])

    # The following proposed layers
    for ly in range(2,layers):
        
        for reference_model in reference_models:

            agent = LocalLLM(reference_model)  # Load the agent once

        # Process all items in the dataset with the loaded agent
            with open(f"Interoutput_from_layer{2}_response.json", "r") as file:
                file_content = json.load(file)
        # Step 2: Replace eval_set with file_content
            results_layer_fl = await asyncio.gather(*[
            process_fn_ly(item, model=agent, temperature=temperature, max_tokens=max_tokens, layers=3)
                for item in file_content  # Replace eval_set with file_content
            ])
            del agent
            gc.collect()
        

        await asyncio.gather (*[interoutput_fn_ly(reference_models= reference_models_list, layers= 3)])
    


    # Last layer (aggregated layer)
    agent = LocalLLM(model)
    with open(f"Interoutput_from_layer{layers-1}_response.json", "r") as file:
                file_content = json.load(file)

    final_output = await asyncio.gather(*[
        process_fn_agg(item, model=agent, temperature=temperature, max_tokens=max_tokens,output_path= output_path)
        for item in file_content  
        ])
    
    with open(output_path, "w") as f:
        json.dump(list(final_output), f, indent=2)    
    
    del agent
    gc.collect()

    # Adjust the prompt to the initial prompts
    with open(f"Interoutput_from_layer1_response.json", "r") as file:
            Original_instruction = json.load(file)

    with open(output_path, "r") as file:
            Current_instruction = json.load(file)
    
    for Current, Original in zip(Current_instruction, Original_instruction):
        Current['instruction'] = Original['instruction']


    with open(output_path, 'w') as file_a_modified:
        json.dump(Current_instruction, file_a_modified, indent=4)

    print("The 'instruction' keys have been successfully updated.")

if __name__ == "__main__":
    Fire(main)

