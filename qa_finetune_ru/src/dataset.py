import json
from torch.utils.data import Dataset
import pandas as pd

class QA(Dataset):
    PATH = "./dataset"

    def __init__(self, partition="train") -> None:
        super().__init__()

        self.partition = partition
        self.init_data = open(f"/home/leo/nlp_test_task/solution/dataset/{partition}.json").read()
        self.init_data = json.loads(self.init_data)
        keys_except_extracted = set(self.init_data[0].keys()).difference(set(["extracted_part"]))

        # convert to { feature: list of values }
        self.data = {
            k: [
                self.init_data[i][k]
                for i in range(len(self.init_data))
            ]
            for k in keys_except_extracted
        }
        if partition == "train":
            # Add extracted
            extracted_parts = [ entry["extracted_part"] for entry in self.init_data ]
            for k in extracted_parts[0].keys():
                self.data[f"extracted_{k}"] = [
                    extracted_parts[i][k][0]
                    for i in range(len(extracted_parts))
                ]
            
        self.data = pd.DataFrame(self.data).reset_index()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = {
            "id": self.data.loc[index, "id"],
            "document": self.data.loc[index, "text"],
            "question": self.data.loc[index, "label"],
        }
        if self.partition == "train":
            item.update({
                "answer": self.data.loc[index, "extracted_text"],
                "answer_start": self.data.loc[index, "extracted_answer_start"],
                "answer_end": (
                    self.data.loc[index, "extracted_answer_start"] + 
                    len(self.data.loc[index, "extracted_text"])
                ),
            })
        return item