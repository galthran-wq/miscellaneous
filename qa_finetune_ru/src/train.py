import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, BertForQuestionAnswering, AutoModel, AutoModelForTokenClassification
from transformers import default_data_collator
from transformers import TrainingArguments, Trainer
from datasets import Dataset

from dataset import QA
from train_utils import (
    prepare_train_features, prepare_validation_features,
    squad_predictions_from_logits
)

# checkpoint = "AlexKay/xlm-roberta-large-qa-multilingual-finedtuned-ru"
checkpoint = "surdan/LaBSE_ner_nerel"
checkpoint_name = checkpoint.split("/")[1]
model = AutoModelForQuestionAnswering.from_pretrained(
    checkpoint, num_labels=2
    # "AndrewChar/model-QA-5-epoch-RU", from_tf=True
    # "surdan/LaBSE_ner_nerel"
    # "DeepPavlov/rubert-base-cased",
)

for name, param in model.named_parameters():
    if name.startswith("bert") or name.startswith("roberta"):
        param.requires_grad = False
        print(f"{name} -- freezed")
    else:
        print(name)

tokenizer = AutoTokenizer.from_pretrained(
    checkpoint
    # "AndrewChar/model-QA-5-epoch-RU", from_tf=True
    # "surdan/LaBSE_ner_nerel"
    # "DeepPavlov/rubert-base-cased"
)

data = QA()
generator = torch.Generator().manual_seed(42)
train, val = torch.utils.data.random_split(
    data, [0.8, 0.2],
    generator=generator
)

train = Dataset.from_list(train)
val = Dataset.from_list(val)

tokenized_train = train.map(
    prepare_train_features, 
    batched=True, 
    remove_columns=train.column_names,
    fn_kwargs={"tokenizer": tokenizer, "max_length": 400, "doc_stride": 30}
)
tokenized_val = val.map(
    prepare_train_features, batched=True, remove_columns=val.column_names,
    fn_kwargs={"tokenizer": tokenizer, "max_length": 400, "doc_stride": 30}
)
tokenized_val_for_evaluation = val.map(
    prepare_validation_features, batched=True, remove_columns=val.column_names,
    fn_kwargs={"tokenizer": tokenizer, "max_length": 400, "doc_stride": 30}
)

import evaluate
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    """
    TODO Problem: In order to compute squad metrics one needs to 
        reference the dataset.
    """
    
    logits, labels = eval_pred
    start_logits = logits[0]
    end_logits = logits[1]
    start_positions = labels[0]
    end_positions = labels[1]
    start_preds = start_logits.argmax(axis=-1)
    end_preds = end_logits.argmax(axis=-1)

    # TODO: problem
    tokenized_val_for_evaluation.set_format(
        type=tokenized_val_for_evaluation.format["type"], 
        columns=list(tokenized_val_for_evaluation.features.keys())
    )

    return {
        "start_accuracy": accuracy.compute(
            predictions=start_preds,
            references=start_positions
        )["accuracy"],
        "end_accuracy": accuracy.compute(
            predictions=end_preds,
            references=end_positions
        )["accuracy"],
        "f1": squad_predictions_from_logits(
            logits=logits,
            # TODO: problem
            ds=val,
            tokenized_ds=tokenized_val_for_evaluation,
            tokenizer=tokenizer
        )["f1"]
    }

training_args = TrainingArguments(
    logging_dir="logs",
    output_dir="test",
    learning_rate=1e-4,
    save_strategy="epoch",
    # save_steps=1,
    evaluation_strategy="epoch",
    # eval_steps=1,
    logging_strategy="epoch",
    # logging_steps=30,
    dataloader_pin_memory=False,
    per_device_eval_batch_size=64,
    per_device_train_batch_size=32,
    num_train_epochs=180,
    lr_scheduler_type="constant",
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics
)
trainer.train("./test/checkpoint-3608")
