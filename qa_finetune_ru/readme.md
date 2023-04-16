### train.json format 
- `id`: int - document id
-  `text`: str - document text which might contain the relevant (according to `label`) part.
- `label`: str - Takes two values: `обеспечение исполнения контракта` or `обеспечение гарантийных обязательств`
- `extracted_part`: dict :
    ```
    {
        'text': [text fragment from `text`, corresponding to ```label```], 
        'answer_start': [char index of the start ],
        'answer_end': [char index of the end]
    }
   ```

### results

3 uptrained on QA checkpoints

| Name     | start_accuracy | end_accuracy | f1 | 
| ---      | ---       | 
| AlexKay/xlm-roberta-large-qa-multilingual-finedtuned-ru | 0.07 | |  |
| AlexKay/xlm-roberta-large-qa-multilingual-finedtuned-ru | 0.07 | |  |
| **DeepPavlov/rubert-base-cased** | 0.85 | 0.65 | -- |

The best model is at https://huggingface.co/galthran/rubert-base-cased-qa

