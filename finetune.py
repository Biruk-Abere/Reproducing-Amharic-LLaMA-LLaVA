import torch 
from contextlib import nullcontext

from transformers import(
    LlamaForCausalLM,
    LlamaTokenizer,
    TrainerCallback,
    default_data_collator,
    Trainer, 
    TrainingArguments

)

'''
LlamaForCausalLM :- This is a model class in the 'transformers' library specifically 
designed to work with Llama models for causal language modeling (CLM). Causal language modeling is a type of language model 
where the prediction of the next word only depends on the previously seen words, not on any words that come after it

LlamaTokenizer:- This tokenizer is used to convert text into a format that is understandable by the Llama Model. It handles tasks such  splitting text into tokens
, converting tokens into numerical IDs, and managing special tokens that are used by language models 
(like start-of-sequence and end-of-sequence tokens)

Trainercallback :- This is a base class for creating callbacks that can be used during the training process of a model. Callbacks are a way to execute 
custom logic at specific points in the training loop, such as the start or end of training, before or after each batch or after each epoch

default_data_collator :- In the context of machine learning, a data collator is responsible for batching
together examples and preparing them for input into a model. The default_data_collator handles common batch 
preparation steps for language modeling tasks.  

Trainer :- This class provides an API for feature-complete training of transformers models. It is designed to be flexible and easily extendable to enable simple and efficient 
model training. It handles training loops, evaluation, model saving, device placement and more


TrainingArguments :- This class is used to store the hyperparameters and other arguments needed for training a model. It includes settings like the 
directory to save the model, the batch_size, the number of training epochs, learning rate and other training specific options. 


'''

from peft import(

    LoraConfig,
    TaskType,
    prepare_model_for_int8_training,
    PeftModel 
)

from pathlib import Path
from utils.dataset_utils import get_preprocessed_dataset
from configs.datasets import amharic_dataset

def print_trainable_parameters(model):
    print("Trainable Parameters: ")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f" - {name}")


'''
This function is used to display the names of all trainable parameters in the model
So the function then enters to a loop where it iterates over each parameter in the model, which returns an iterator of tuples, each 
containing the name of the parameter (as a string) and the parameter itself.

check for trainability :- Within the loop, there's a conditional check 'if param.requires_grad', this checks whether the parameter is trainable. 
requires_grad is a property of tensors that indicates whether gradients should be computed for this tensor during backpropagation, if requires_grad is True, it means the parameter is part of the model 
that will be updated during training

'''


def finetune():
    LLAMA_DIR = '/path/to/llama/weights'
    PT_DIR = '/path/to/pt/weights'
    OUTPUT_DIR = '/path/to/output'


    tokenizer = LlamaTokenizer.from_pretrained(LLAMA_DIR)
    model = LlamaForCausalLM.from_pretrained(LLAMA_DIR, load_in_8bit = False, device_map = 'auto', torch_dtype = torch.float16)

    train_dataset = get_preprocessed_dataset(tokenizer, amharic_dataset, 'train')
    model.train()

    embedding_size = model.get_input_embeddings().weight.shape[0]

    if len(tokenizer)!= embedding_size:
        print("resize the embedding size by the size of the tokenizer")
        model.resize_token_embeddings(len(tokenizer))
    
    print("loading the pre-trained model from config")

    model = prepare_model_for_int8_training(model)
    model = PeftModel.from_pretrained(model, PT_DIR)
    model.print_trainable_parameters()

    lora_config = LoraConfig(
        task_type = TaskType.CAUSAL_LM,
        inference_mode = False,
        r = 8, 
        lora_alpha = 32, 
        lora_dropout = 0.05,
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj","down_proj", "up_proj"]

    )

    enable_profiler = False
    config = {

        'lora_config': lora_config, 
        'learning_rate': 1e-4,
        'num_train_epochs': 1,,
        'gradient_accumulation_steps':1,
        'per_device_train_batch_size':2,
        'gradient_checkpointing':False,

    }

     # Set up profiler
    if enable_profiler:
        wait, warmup, active, repeat = 1, 1, 2, 1
        total_steps = (wait + warmup + active) * (1 + repeat)
        schedule =  torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)
        profiler = torch.profiler.profile(
            schedule=schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f"{OUTPUT_DIR}/logs/tensorboard"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True)

        class ProfilerCallback(TrainerCallback):
            def __init__(self, profiler):
                self.profiler = profiler

            def on_step_end(self, *args, **kwargs):
                self.profiler.step()

        profiler_callback = ProfilerCallback(profiler)
    else:
        profiler = nullcontext()


    # define training args 
    training_args = TrainingArguments(
        OUTPUT_DIR = OUTPUT_DIR, 
        overwrite_OUTPUT_DIR = True, 
        bf16 = True, # use BF16 if available, 
        # logging strategies 
        logging_dir = f"{OUTPUT_DIR}/logs",
        logging_strategy = "steps",
        logging_steps = 10, 
        save_strategy = "steps",
        logging_steps = 10,
        save_strategy = "steps",
        save_steps = 1000,
        save_total_limit = 1,
        warmup_ratio = 0.03,
        optim = "adamw_torch_fused",
        max_steps = total_steps if enable_profiler else -1,
        **{k:v for k,v in config.items() if k!= "lora_config"}

    )

    with profiler: 
        # create Trainer instance 
        trainer = Trainer(
            model = model,
            args = training_args,
            train_dataset = train_dataset,
            data_collator = default_data_collator, 
            callbacks=[profiler_callback] if enable_profiler else [], 
        )

        print_trainable_parameters(model)

        # start training 
        trainer.train()

    model.save_pretrained(OUTPUT_DIR)


#  function call 
finetune()


    







