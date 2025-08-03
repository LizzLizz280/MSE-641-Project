import json
from pathlib import Path
import numpy as np
import pandas as pd
import re
import html
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from evaluate import load as load_metric

def dummyMetrics(eval_pred):
    return {}

def loadData(fname):
    data = []
    with open(fname, encoding="utf-8") as f:
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


def preprocess(input_text):
    model_inputs = tokenizer(
        input_text["text"],
        max_length=400,
        truncation=True,
        padding="max_length",
    )
    if "label" in input_text:
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                input_text["label"],
                max_length=64,
                truncation=True,
                padding="max_length",
            )["input_ids"]
        model_inputs["labels"] = labels
    return model_inputs

if __name__ == "__main__":
    training = loadData("train.jsonl")
    validation = loadData("val.jsonl")

    train_texts = []
    train_labels = []

    for record in training:
        text = getFeaturesSpoiler(record)
        label = combineSpoiler(record["spoiler"])
        train_texts.append(text)
        train_labels.append(label)

    train_dict = {
        'text': train_texts,
        'label': train_labels
    }
    train_ds = Dataset.from_dict(train_dict)

    val_texts = []
    val_labels = []

    for record in validation:
        text = getFeaturesSpoiler(record)
        label = combineSpoiler(record["spoiler"])
        val_texts.append(text)
        val_labels.append(label)

    val_dict = {
        'text': val_texts,
        'label': val_labels
    }
    val_ds = Dataset.from_dict(val_dict)

    tokenizer = AutoTokenizer.from_pretrained('t5-small', use_fast=True)

    train_tok = train_ds.map(preprocess, batched=True, remove_columns=["text", "label"])
    val_tok = val_ds.map(preprocess, batched=True, remove_columns=["text", "label"])

    model = AutoModelForSeq2SeqLM.from_pretrained('t5-small')
    model.gradient_checkpointing_enable()

    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    args = Seq2SeqTrainingArguments(
        output_dir=str(Path("task2_t5_small")),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=5,
        learning_rate=5e-5,
        eval_strategy="no",
        save_strategy="no",
        predict_with_generate=True,
        fp16=False,
        logging_steps=50,
        dataloader_pin_memory=False,
        max_grad_norm=1.0,
        generation_max_length=64
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=dummyMetrics
    )

    trainer.train()

    meteor = load_metric("meteor")
    pred = trainer.predict(val_tok, max_length=64)
    logits_or_ids = pred.predictions
    n_dims = logits_or_ids.ndim


    if n_dims == 3:
        pred_ids = np.argmax(logits_or_ids, axis=-1)
    else:

        pred_ids = logits_or_ids

    pred_ids = np.where(
        (pred_ids < 0) | (pred_ids >= tokenizer.vocab_size),
        tokenizer.eos_token_id,
        pred_ids.astype("int32"),
    )

    pred_text = tokenizer.batch_decode(pred_ids.tolist(), skip_special_tokens=True)

    predictions = []
    for sub_pred in pred_text:
        predictions.append(sub_pred.strip())

    references = []
    for sub_ref in val_ds["label"]:
        references.append(sub_ref.strip())

    score = meteor.compute(
        predictions=predictions,
        references=references,
    )

    print('val set meteor:', score['meteor'])

    trainer.save_model(str(Path("task2_t5_small")))
    tokenizer.save_pretrained(str(Path("task2_t5_small")))
    print('finish')