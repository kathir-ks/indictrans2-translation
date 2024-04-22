import os
import sys
import subprocess
import torch
from torch.utils.data import DataLoader
import jax
import jax.numpy as jnp
import numpy as np
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from modeling_flax_indictrans import FlaxIndicTransForConditionalGeneration
from IndicTransTokenizer import IndicTransTokenizer, IndicProcessor

if __name__ == '__main__':
    # Load model
    

    model = FlaxIndicTransForConditionalGeneration.from_pretrained(
        model_path, 
        local_files_only=True,
        dtype=jnp.float16,
    )

    params = replicate(model.params)

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

    def run_inference_step(batch, params, run_ds):

        input_batch = {
            "input_ids": shard(jnp.array(batch["input_ids"])),
            "attention_mask": shard(jnp.array(batch["attention_mask"]))
        }
        
        outputs = p_generate(input_batch, params)

        outputs = outputs.block_until_ready()

        if local_device_count != 1:
            outputs = outputs.reshape(-1, *outputs.shape[2:])
        else:
            outputs = outputs[0]
        
        return outputs



    tokenizer = IndicTransTokenizer(direction="en-indic")
    ip = IndicProcessor(inference=True)
    
    out = tokenizer.batch_decode(np.asarray(outputs), src=False)
    out = ip.postprocess_batch(out, lang="tam_Taml")  
