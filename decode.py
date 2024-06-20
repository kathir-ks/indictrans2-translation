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
    # parser.add_argument("--lang", type=str, required=True)
    parser.add_argument("--direction", type=str,default="en-indic", required=False)
    parser.add_argument("--batch_size", type=int, default=64, required=False)

    args = parser.parse_args()
    file = args.file
    lang = 'hin_Deva'
    direction = args.direction
    batch_size = args.batch_size

    curr_dir = os.getcwd()

    file_path = f'{curr_dir}/{file}.json'

    batches = []
    sentences = []

    tokenizer = IndicTransTokenizer(direction=direction)
    ip = IndicProcessor(inference=True)

    data = load_json_file(file_path)

    for i in range(0, len(data), batch_size):
        batches.append(data[i : i + batch_size])

    del data

    for batch in batches:
        out = tokenizer.batch_decode(np.asarray(batch), src=False)
        out = ip.postprocess_batch(out, lang=lang, placeholder_entity_maps={})
        for i in range(0, len(out)):
            if "<ID" not in out[i]:
                sentences.extend(out[i])

    with open(f'{file}_sentences.json', 'w') as f:
        json.dump(sentences, f)
    
    os.system(f'rm {file}.json')
