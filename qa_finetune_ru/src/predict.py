import json
import argparse
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, Trainer
from datasets import Dataset

from train_utils import (
    prepare_validation_features, postprocess_qa_predictions,
    squad_predictions_from_logits
)
from dataset import QA


def build_args():
    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    parser.add_argument('--model-checkpoint', default="galthran/rubert-base-cased-qa")      
    parser.add_argument('--tokenizer-checkpoint', default="DeepPavlov/rubert-base-cased")      
    parser.add_argument(
        '--to-evaluate',
        action=argparse.BooleanOptionalAction, 
        default=True
    )
    parser.add_argument('-o', "--output", default="predictions")      
    # parser.add_argument('--dataset', default=None)      
    return parser


def predictions2output_format(predictions):
    results = []
    for id, prediction_dict in predictions.items():
        results.append({
            "id": id,
            "text": prediction_dict["document"],
            "label": prediction_dict["label"],
            "extracted_part": {
                "text": [prediction_dict["text"]],
                "answer_start": [prediction_dict["answer_start"]],
                "answer_end": [prediction_dict["answer_end"]],
            }
        })
    return results

if __name__ == "__main__":
    args = build_args().parse_args()
    model = AutoModelForQuestionAnswering.from_pretrained(
        args.model_checkpoint
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_checkpoint
    )
    # ds = QA(partition="test")
    ds = QA(partition="train")
    def ds_gen():
        for i in range(len(ds)):
            yield ds[i]
    ds = Dataset.from_generator(ds_gen)
    tokenized_ds = ds.map(
        prepare_validation_features, 
        batched=True,
        remove_columns=ds.column_names,
        fn_kwargs={"tokenizer": tokenizer, "max_length": 400, "doc_stride": 100}
    )
    trainer = Trainer(model=model)
    raw_predictions =  trainer.predict(tokenized_ds).predictions
    # The Trainer hides the columns that are not used by the model 
    # (here example_id and offset_mapping which we will need for our post-processing), so we set them back:
    tokenized_ds.set_format(
        type=tokenized_ds.format["type"], 
        columns=list(tokenized_ds.features.keys())
    )
    predictions = postprocess_qa_predictions(
        ds, tokenized_ds, raw_predictions, tokenizer, max_answer_length=100
    )

    #
    with open(f'{args.output}.json', 'w') as outfile:
        json.dump(
            predictions2output_format(predictions), 
            outfile,
            ensure_ascii=False,
            indent=2
        )
    
    # requires labels
    if args.to_evaluate:
        print(
            squad_predictions_from_logits(
                logits=raw_predictions,
                ds=ds, tokenized_ds=tokenized_ds,
                tokenizer=tokenizer,
            )
        )