import copy
import json
import torch


from torch.utils.data import Dataset

class InstructionDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", max_words=50):
        self.ann = json.load(open(dataset_config.data_path))
        if partition == "train":
            self.ann = self.ann
        else:
            self.ann = self.ann[:200]

        self.max_words = max_words
    
        self.tokenizer = tokenizer

    '''

    Parameters:
        dataset_config: An object that contains configuration details for the dataset, such as file paths. It is expected to have an attribute data_path which is used to locate the dataset file.
        tokenizer: A tokenizer object used to convert text into numerical tokens that machine learning models can process.
        partition: A string that defaults to "train". This parameter determines which part of the dataset to load. If not "train", only a subset (first 200 entries) is used, possibly intended for validation or testing.
        max_words: An integer that sets the upper limit for the number of words/tokens in processed dataset entries. This is used to standardize the length of input sequences, important for batching in neural networks.
        Loading and Processing Data:

    self.ann = json.load(open(dataset_config.data_path)): Loads the dataset from a JSON file specified by dataset_config.data_path. The file is opened and its contents are parsed into a Python object (self.ann).
    Conditional Data Loading:
        If partition is "train", it retains the full dataset loaded into self.ann.
        If partition is not "train", it limits self.ann to the first 200 entries. This is typically used to handle smaller sets for validation or testing to prevent overfitting and to speed up evaluation.
    
    Storing Configuration:
   
        self.max_words: Stores the max_words parameter as an instance variable, which will later dictate the maximum length of tokenized sequences.
        self.tokenizer: Stores the tokenizer passed to the constructor for use in tokenizing text when the dataset is accessed.
            
    
    '''


    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]

        prompt = ann["input"]
        example = prompt + ann["output"]
 
        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )

        example = self.tokenizer.encode(example)

        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )

        '''
        The 'getitem' method in the 'InstructionDataset' class is a crucial component for handling
        data retrieval and preprocessing in a machine learning context

        ann = self.ann[index]: This line retrieves the dataset entry at the specified index. 'ann' contains the data for a single example, 
        which in this context likely includes both an "input" and an "output" component. 

        Construct Inputs :- 
            prompt = ann["input"]: Extracts the "input" part of the dataset entry. This text will be used as the input prompt for the model 
            example = prompt + ann["output"] :- Concatenates the "input" with the "output" to form a full example. This combination likely represents a sequence that the model should 
            learn to generate from the given prompt. 

        prompt = torch.tensor(
                    self.tokenizer.encode(prompt), dtype=torch.int64
                ) -- This is tokenizing the input, which converts the "input" text to a sequence of 
        token IDs using the tokenizer's encode function, which translates text into a list of numerical identifiers that the model can process. 
        These token IDs are then converted into a PyTorch tensor with a data type of 64-bit integers.         

        Tokenize and Extend Example :- 

            example = self.tokenizer.encode(example) :- similar to the prompt this line encodes the concatenated "input" and "output" text into a sequence of token IDs
            example.append(self.tokenizer.eos_token_id):- Appends the end of the sequence (EOS) token ID to the end of the "example" token list. The EOS token is a special marker used in 
            language models to indicate the end of a sequence, helping the model distinguish where a sentence or output should conclude
            example = torch.tensor(example, dtype = torch.int64) :- converts the list of token IDs (including the EOS token) into a PyTorch tensor, also with a datatype of 64-bit integers. 
            
        '''

        padding = self.max_words - example.shape[0]

        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[: self.max_words]

        labels = copy.deepcopy(example)

        labels[: len(prompt)] = -1 # masking out

        '''
            Calculate padding :- 
                padding  = self.max_words = example.shape[0] :- This line calculates the amount of padding needed by subtracting the current number of 
                tokens in the 'example' tensor from the 'max_words' limit, which is the maximum allowed length of sequences, This maximum length is enforced to ensure consistency in tensor sizes across different data points, 
                which is crucial for batch processing in neural networks. 

            Adjust example tensor length :- 
                If padding is needed (if padding > 0):- if the calculated "padding" value is positive, it means that the "example" tensor 
                has fewer tokens than "max_words". To rectify this, additional padding tokens are added. "torch.zeros(padding, dtype = torch.int64) - 1" creates a tensor of zeros with 
                length equal to "padding" and then subtracts 1 from each element, effectively creating a tensor of -1 values. This tensor of -1 is then concatenated to the "example" tensor using torch.cat, extending its length to "max_words"

                If Trimming is Needed (elif padding < 0 ):- If padding is negative, the "example" tensor exceeds the "max_words" limit. The tensor is then truncated to "max_words" by slicing it ('example[:self.max_words]') 

            Create Labels Tensor for Training:- 

                labels = copy.deepcopy(example): creates a deep copy of the example tensor to use as labels, which will be used for calculating loss during model training. Deep Copying is essential here to ensure that changes to labels do not affect "example"

                labels[:len(prompt)] = -1 : set the elements of the labels tensor that correspond to the prompt portion to -1. This operation effectively masks out the prompt part in the loss calculation, telling the training process to ignore the prompt's tokens and focus only on the output
                tokens for training. This is commonly done in tasks like conditional generation, where the model should not be penalized for its reproduction of the input sequence. 

        '''


        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = 0
        example_mask = example_mask.float()
        label_mask = label_mask.float()

        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask":example_mask,
        }
    


    '''
    This method completes the preparation of the data for training by creatnig masks for the tensors and ensuring that 
    any negative values (which are used for padding and to mask out certain tokens ) are properly handled. 
    
    
    
    
    
    
    '''