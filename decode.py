import json
import nltk
import time
import os
import argparse
import numpy as np
from IndicTransTokenizer import IndicTransTokenizer, IndicProcessor
nltk.download('punkt')


def load_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="decode inference completed output")
    parser.add_argument("--file", type=str, default=None, required=True)
    parser.add_argument("--lang", type=str, required=True)
    parser.add_argument("--direction", type=str,default="en-indic", required=False)
    parser.add_argument("--batch_size", type=int, default=64, required=False)

    args = parser.parse_args()
    file = args.file
    lang = args.lang
    direction = args.direction
    batch_size = args.batch_size

    curr_dir = os.getcwd()

    file_path = f'{curr_dir}/{file}.json'

    batches = []
    sentences = []

    tokenizer = IndicTransTokenizer(direction=direction)
    ip = IndicProcessor(inference=True)

    data = load_json_file(file_path)

    tokens = []
    for d in data:
        tokens.extend(d)

    del data

    for i in range(0, len(tokens), batch_size):
        batches.append(tokens[i : i + batch_size])

    del tokens

    for batch in batches:
        output = tokenizer.batch_decode(np.asarray(batch), src=False)
        placeholder_entity_maps = [{}] * len(batch)
        outputs = ip.postprocess_batch(output, lang=lang, placeholder_entity_maps=placeholder_entity_maps)
        for output in outputs:
            if "<ID" not in output:
                sentences.extend(output)

    with open(f'{file}_sentences.json', 'w') as f:
        json.dump(sentences, f)
    
    os.system(f'rm {file}.json')
