import torch

from functools import partial

from finetune_datasets import get_amharic_dataset

DATASET_PREPROC = {
    "amharic_dataset": partial(get_amharic_dataset, max_words=672)
}

def get_preprocessed_dataset(
    tokenizer, dataset_config, split: str = "train"
) -> torch.utils.data.Dataset:
    if not dataset_config.dataset in DATASET_PREPROC:
        raise NotImplementedError(f"{dataset_config.dataset} is not (yet) implemented")

    def get_split():
        return (
            dataset_config.train_split
            if split == "train"
            else dataset_config.test_split
        )
    
    return DATASET_PREPROC[dataset_config.dataset](
        dataset_config,
        tokenizer,
        get_split(),
    )


'''
'from ft_datasets import get_amharic_dataset' :- imports a specific function get_amharic_dataset, a custom function designed to fetch and 
preprocess AMharic Language datasets

DATASET_PREPROC = {
    "amharic_dataset": partial(get_amharic_dataset, max_words=672)
}  -- The function should use a maximum of 672 words, this could mean limiting the number of words per data instance, trimming data, or 
configuring the dataset loading in some way that incorporates this contrain

The get_preprocessed dataset function is designed to fetch and preprocess a dataset accoding to a specific configurations and for a specified split (either training or testing)


Function Definition:

    tokenizer: This parameter is expected to be an instance of a tokenizer, which is used to convert raw text data into a format suitable for machine learning models (typically converting text to numerical tokens).
    dataset_config: This is a configuration object or dictionary that contains settings and paths related to the dataset (like names, paths, and splits).
    split: A string parameter that defaults to "train", indicating which part of the dataset the function should process (training or testing part).

 Check Dataset Availability:

    The function first checks if the dataset specified in dataset_config.dataset is available in the DATASET_PREPROC dictionary. 
    If it's not found, it raises a NotImplementedError. This error handling ensures that the function only processes datasets that have been predefined and configured properly in the DATASET_PREPROC dictionary.   

    
Determine the Appropriate Dataset Split:

    get_split(): This nested function determines which dataset split to use based on the split argument. It uses a conditional expression to return either the training split (dataset_config.train_split) or the testing split (dataset_config.test_split) from the dataset_config.
    This allows the function to dynamically select the appropriate part of the dataset based on the intended use (training or evaluation)

Fetch and Preprocess the Dataset:

    The final return statement calls the function associated with the dataset in the DATASET_PREPROC dictionary. It passes dataset_config, tokenizer, and the result of the get_split() function as arguments.
    This call is expected to execute the preconfigured function (set up with partial in the DATASET_PREPROC dictionary) which fetches and preprocesses the dataset. 
    The preprocessing could include operations like tokenization, truncation, padding, or any other necessary transformations to make the dataset suitable for training or testing with a machine learning model.

'''