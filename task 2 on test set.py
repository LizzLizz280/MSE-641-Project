import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import html
import re

def loadData(filename):
    data = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                proceed = json.loads(line)
                data.append(proceed)
    return data

def combineSpoiler(txt):
    if isinstance(txt, str):
        return txt
    else:
        return ' '.join(txt)

def dataPrep(raw_data):
    data = html.unescape(raw_data)
    data_new = re.sub(r'\s+', ' ', data)
    return ' '.join(data_new.split())

def getFeaturesSpoiler(record):
    post  = ' '.join(record.get('postText', []))
    title = record.get('targetTitle', '')
    paras = ' '.join(record.get('targetParagraphs', []))

    raw_feature = ' '.join([post, title, paras])
    return dataPrep(raw_feature)

# def preprocess(input_text):
#     model_inputs = tokenizer(
#         input_text["text"],
#         max_length=400,
#         truncation=True,
#         padding="max_length",
#     )
#     if "label" in input_text:
#         with tokenizer.as_target_tokenizer():
#             labels = tokenizer(
#                 input_text["label"],
#                 max_length=64,
#                 truncation=True,
#                 padding="max_length",
#             )["input_ids"]
#         model_inputs["labels"] = labels
#     return model_inputs

if __name__ == "__main__":
    test_raw = loadData('test.jsonl')

    test_ids = []
    for sub_id in test_raw:
        querried = sub_id['id']
        test_ids.append(querried)

    test_id_list = []
    test_text_list = []
    for sub_id in test_raw:
        obtained_id = sub_id['id']
        obtained_text = getFeaturesSpoiler(sub_id)
        test_id_list.append(obtained_id)
        test_text_list.append(obtained_text)

    test_dict = {
        "id": test_id_list,
        "text": test_text_list,
    }
    test_ds = Dataset.from_dict(test_dict)

    tokenizer = AutoTokenizer.from_pretrained('task2_t5_small', use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained('task2_t5_small').to(
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    model.eval()
    model.config.use_cache = True

    predictions = []
    for start in range(0, len(test_ds), 8):
        batch_texts = test_ds['text'][start: start + 8]
        inputs = tokenizer(
            batch_texts,
            max_length=400,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        ).to(model.device)

        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_length=64,
                num_beams=4,
                early_stopping=True,
            )
        preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        pred_res = []
        for raw_text in preds:
            stripped_text = raw_text.strip()
            pred_res.append(stripped_text)
        predictions.extend(pred_res)

    out_path = Path('prediction_task2.csv')

    output_dict = {
        'id': test_ids,
        'spoiler': predictions
    }

    df = pd.DataFrame(output_dict)
    df.to_csv(out_path, index=False, encoding='utf-8')