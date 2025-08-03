import json
import re
import numpy as np
import pandas as pd
import html
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, EarlyStoppingCallback

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

def loadData(filename):
    data = []
    with open(filename, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def dataPrep(raw_data):
    data = html.unescape(raw_data)
    data_new = re.sub(r"\s+", " ", data)
    data_output = " ".join(data_new.split())
    return data_output

def getFeatures(record):
    post = record["postText"][0]
    title = record.get("targetTitle") or ""
    paragraph_list = record.get("targetParagraphs") or []
    paragraph = " ".join(paragraph_list)
    output_feature = " ".join([post, title, paragraph])
    return dataPrep(output_feature)

def featuresList(records):
    out = []
    for item in records:
        out.append(getFeatures(item))
    return out

def getLabel(records):
    id_dict = {
        "phrase": 0,
        "passage": 1,
        "multi": 2
    }
    label_list = []
    for item in records:
        label = item["tags"][0]
        label_list.append(id_dict[label])
    return np.array(label_list, dtype="int32")

tad_dict = {"phrase": 0, "passage": 1, "multi": 2}

id_convert_dict = {}
for label_name, label_numeric in tad_dict.items():
    id_convert_dict[label_numeric] = label_name

def computeMetrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }

def main():
    train_raw = loadData("train.jsonl")
    val_raw = loadData("val.jsonl")
    test_raw = loadData("test.jsonl")

    train_texts = featuresList(train_raw)
    train_labels = getLabel(train_raw)
    train_dict = {
        "text": train_texts,
        "label": train_labels
    }
    train = Dataset.from_dict(train_dict)

    val_texts = featuresList(val_raw)
    val_labels = getLabel(val_raw)

    val_dict = {
        "text": val_texts,
        "label": val_labels
    }
    val = Dataset.from_dict(val_dict)

    test_texts = featuresList(test_raw)
    test_ids = []
    for rec in test_raw:
        test_ids.append(rec["id"])

    test_dict = {
        "text": test_texts,
        "id": test_ids
    }
    test = Dataset.from_dict(test_dict)

    tok = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')

    def tokenize(batch):
        texts = batch["text"]


        tokenized_outputs = tok(
            texts,
            truncation=True,
            max_length=256,
            padding=False
        )
        return tokenized_outputs

    train = train.map(tokenize, batched=True, remove_columns=["text"])
    val = val.map(tokenize, batched=True, remove_columns=["text"])
    test = test.map(tokenize, batched=True, remove_columns=["text"])

    collator = DataCollatorWithPadding(tok)

    ytrain = np.array(train["label"])
    num_training = np.bincount(ytrain, minlength=3)
    weights = torch.tensor(num_training.sum() / (3 * num_training), dtype=torch.float)

    class WeightedTrainer(Trainer):

        def __init__(self, class_weights, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.loss_func = torch.nn.CrossEntropyLoss(
                weight=class_weights.to(self.args.device)
            )

        def computeLoss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = self.loss_func(outputs.logits, labels)
            return (loss, outputs) if return_outputs else loss

    model = AutoModelForSequenceClassification.from_pretrained(
        'microsoft/deberta-v3-base',
        num_labels=3
    )

    args = TrainingArguments(
        output_dir="task1_deberta_v3_base",
        num_train_epochs=5,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.005,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        logging_steps=50,
        report_to="none",
    )

    trainer = WeightedTrainer(
        class_weights=weights,
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset=val,
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=computeMetrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()
    print("val set:", trainer.evaluate())


    test_logits = trainer.predict(test).predictions
    test_pred = test_logits.argmax(1)

    ids = test["id"]

    spoiler_types = []
    for pred in test_pred:
        mapped_label = id_convert_dict[pred]
        spoiler_types.append(mapped_label)

    df_dict = {
        "id": ids,
        "spoilerType": spoiler_types
    }
    df = pd.DataFrame(df_dict)
    df.to_csv("prediction_task1_transformer.csv", index=False)
    val_out = trainer.predict(val)
    ytrue = val_out.label_ids
    ypred = val_out.predictions.argmax(-1)

    cm = confusion_matrix(ytrue, ypred, labels=[0, 1, 2])
    print(cm)

if __name__ == "__main__":
    main()

