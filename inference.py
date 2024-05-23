import os
import jax
import jax.numpy as jnp
import numpy as np
import argparse
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from modeling_flax_indictrans import FlaxIndicTransForConditionalGeneration
from IndicTransTokenizer import IndicTransTokenizer, IndicProcessor
import json
import nltk
nltk.download('punkt')
import time

def load_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def main(model, params, data, batch_size, shard):
        
    t = time.time()

    local_device_count = jax.local_device_count()
    inputs = []
    indices = []
    input_ids = []
    attention_mask = []
    
    for i in data:
        indices.extend(i['indices'])
        input_ids.extend(i['tokenized_input']['input_ids'])
        attention_mask.extend(i['tokenized_input']['attention_mask'])

    assert len(indices) == len(input_ids)
    assert len(input_ids) == len(attention_mask)
    
    def padding_fn(
        batch,
        keys_to_pad=[
                ("input_ids", 1),
                ("attention_mask", 0),
            ]
        ):

        batch_out = {key: [] for key in batch.keys()}
    
        for key in batch_out.keys():
            batch_out[key] += batch[key]
    
        for key, value_to_pad_with in keys_to_pad:

            len_list = list(map(lambda x: len(x), batch_out[key]))

            padding_length = max(len_list)

            if padding_length > 256:
                
                print(padding_length)
                return None
            
            array_list = []
            for i, x in enumerate(batch_out[key]):

                if len(x) < padding_length:
                    padded_array = np.concatenate([np.full((padding_length - len(x)), value_to_pad_with), np.array(x)])
                    array_list.append(padded_array)
                else:
                    array_list.append(np.array(x))

            batch_out[key] = np.stack(array_list)

        return batch_out
    
    for i in range(0, len(input_ids), batch_size):
        
        input = {
            "input_ids": input_ids[i : i + batch_size],
            "attention_mask": attention_mask[i : i + batch_size]
        }
        
        input = padding_fn(input)
        if input and len(input) % local_device_count==0:
            inputs.append(input)

    # @jax.jit
    def generate(
            batch,
            params,
        ):
            model.params = params
            return model.generate(
                **batch,
                num_beams=1,
                num_return_sequences=1,
                max_length=256,
                do_sample=False,
            ).sequences

    p_generate = jax.pmap(generate) 

    # @jax.jit
    def run_inference_step(batch, params, run_ds):
        
        input_batch = {
            "input_ids": shard(jnp.array(batch["input_ids"])),
            "attention_mask": shard(jnp.array(batch["attention_mask"]))
        }
        
        output = []
        try:
            output = p_generate(input_batch, params)

            output = output.block_until_ready()

            if local_device_count != 1:
                output = output.reshape(-1, *output.shape[2:])
            else:
                output = output[0]

        except:
            print("!Error in inference step")

        return output

    outputs = []

    for input in inputs:
        output = run_inference_step(input, params, None)
        outputs.append(output.tolist())

    print("Inference completed!")
    print(time.time() - t)
    
    with open(f'{subset}_output_{shard}.json', 'w') as f:
        json.dump(outputs, f)


if __name__ =='__main__':

    parser = argparse.ArgumentParser(description="Tanslate tokenized sentences")
    parser.add_argument("--subset", type=str, default=None, required=True)
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")

    args = parser.parse_args()
    subset = args.subset
    batch_size = args.batch_size

    curr_dir = os.getcwd()
    file_path = f'{curr_dir}/{subset}.json'
    model_path = f'{curr_dir}/flax_weights/200m'

    if not os.path.isdir(model_path):
        os.system("mkdir flax_weights")
        os.system(f'gsutil cp -R gs://indic-llama-data/indic-llama/flax_weights/200m {curr_dir}/flax_weights/')

    #download the file from google storage if file does not exist
    if not os.path.isfile(file_path):
        os.system(f'gsutil cp gs://indic-llama-data/indic-llama/{subset}.json {subset}.json')

    
    shard = 1

    data = load_json_file(file_path=file_path)

    for i in range (0, len(data), 10000):

        data = data[i : i + 10000]

        model = FlaxIndicTransForConditionalGeneration.from_pretrained(
            model_path, 
            local_files_only=True,
            dtype=jnp.float16,
        )
        print("model loaded")

        params = replicate(model.params)
        print("model replicated")

        main(model, params, data, batch_size, shard)

        shard = shard + 1

        del model, params
    
