from dataclasses import dataclass

@dataclass
class amharic_dataset:
    dataset: str = "amharic_dataset"
    train_split : str = "train"
    test_split: str = "val"
    data_path: str = "data/amharic_finetune.json"

'''
dataset: A string attribute with a default value of "amharic_dataset". This likely represents the name or identifier of the dataset.
train_split: A string attribute with a default value of "train". This specifies the identifier for the training split of the dataset.
test_split: A string attribute with a default value of "val". This specifies the identifier for the validation (or test) split of the dataset.
data_path: A string attribute with a default value of "data/amharic_finetune.json". This indicates the file path where the data for this dataset is stored, presumably in JSON format.

'''