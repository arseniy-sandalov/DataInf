from tqdm import tqdm
import pickle
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    BitsAndBytesConfig,
    T5ForConditionalGeneration,
    T5Tokenizer,
    DataCollatorForSeq2Seq
)
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model
)
from datasets import Dataset
import evaluate

class LORAEngine(object):
    def __init__(self, 
                model_name_or_path="roberta-large",
                target_modules=["value"],
                train_dataloader=None,
                eval_dataloader=None,
                device="cuda",
                num_epochs=10,
                lr=3e-4,
                low_rank=2,
                task="mrpc"):
        self.model_name_or_path=model_name_or_path
        self.target_modules=target_modules
        self.train_dataloader=train_dataloader
        self.eval_dataloader=eval_dataloader
        self.device=device
        self.num_epochs=num_epochs
        self.lr=lr
        self.task=task
        self.low_rank=low_rank
        
    def build_LORA_model(self):
        '''
        This function fine-tunes a model for classification tasks. 
        For text generation tasks, please see `notebooks/Influential_Data_Identification-Llama2-Math.ipynb`.
        '''
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path,
                                                                        return_dict=True)
        self.model.config.use_cache = False
        self.model.config.pad_token_id = self.model.config.eos_token_id
            
        peft_config = LoraConfig(task_type="SEQ_CLS",
                                 inference_mode=False, 
                                 target_modules=self.target_modules,
                                 r=self.low_rank,
                                 lora_alpha=self.low_rank, 
                                 lora_dropout=0.05)
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

    def train_LORA_model(self):
        '''
        This function fine-tunes a model for GLUE classification tasks. 
        For text generation tasks, please see `notebooks/Influential_Data_Identification-Llama2-Math.ipynb`.
        '''
        metric = evaluate.load("glue", self.task)
        optimizer = AdamW(params=self.model.parameters(), lr=self.lr)

        # Instantiate scheduler
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0.06*(len(self.train_dataloader)*self.num_epochs),
            num_training_steps=(len(self.train_dataloader)*self.num_epochs),
        )

        self.model.to(self.device)
        for epoch in range(self.num_epochs):
            self.model.train()
            for step, batch in enumerate(tqdm(self.train_dataloader)):
                batch.to(self.device)
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            self.model.eval()
            for step, batch in enumerate(tqdm(self.eval_dataloader)):
                batch.to(self.device)
                with torch.no_grad():
                    outputs = self.model(**batch)
                predictions = outputs.logits.argmax(dim=-1)
                predictions, references = predictions, batch["labels"]
                metric.add_batch(
                    predictions=predictions,
                    references=references,
                )

            eval_metric = metric.compute()
            print(f"Epoch {(epoch+1)}:", eval_metric)


    def compute_gradient(self, tokenized_datasets, collate_fn):
        train_dataloader_stochastic = DataLoader(tokenized_datasets["train"], 
                                                  shuffle=False,
                                                  collate_fn=collate_fn,
                                                  batch_size=1)
        val_dataloader_stochastic = DataLoader(tokenized_datasets["validation"], 
                                                  shuffle=False,
                                                  collate_fn=collate_fn,
                                                  batch_size=1)
        # Compute the gradient
        self.model.eval()
        tr_grad_dict = {}
        for step, batch in enumerate(tqdm(train_dataloader_stochastic)):
            self.model.zero_grad() # zeroing out gradient
            batch.to(self.device)
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            
            grad_dict={}
            for k, v in self.model.named_parameters():
                if 'lora_A' in k:
                    grad_dict[k]=v.grad.cpu()
                elif 'lora_B' in k:
                    # first index of shape indicates low-rank
                    grad_dict[k]=v.grad.cpu().T
                elif 'modules_to_save.default.out_proj.weight' in k:
                    grad_dict[k]=v.grad.cpu()
                else:
                    pass
            tr_grad_dict[step]=grad_dict
            del grad_dict
            
        val_grad_dict = {}
        for step, batch in enumerate(tqdm(val_dataloader_stochastic)):
            self.model.zero_grad() # zeroing out gradient
            batch.to(self.device)
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            
            grad_dict={}
            for k, v in self.model.named_parameters():
                if 'lora_A' in k:
                    grad_dict[k]=v.grad.cpu()
                elif 'lora_B' in k:
                    # first index of shape indicates low-rank
                    grad_dict[k]=v.grad.cpu().T
                elif 'modules_to_save.default.out_proj.weight' in k:
                    grad_dict[k]=v.grad.cpu()
                else:
                    pass
            val_grad_dict[step]=grad_dict    
            del grad_dict
            
        return tr_grad_dict, val_grad_dict


class LORAEngineGeneration(object):
    def __init__(self, 
                base_path,
                project_path,
                train_path,
                test_path,
                tokenizer,
                dataset_name,
                adapter_path,
                device="cuda"):
        self.base_path = base_path
        self.project_path = project_path
        self.train_path = train_path
        self.test_path = test_path
        self.tokenizer = tokenizer
        self.adapter_path = adapter_path
        self.dataset_name = dataset_name
        self.device=device
        self.load_pretrained_network(tokenizer)
        self.load_datasets()

    def load_pretrained_network(self, tokenizer):
        # setup tokenizer
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "right"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # load a base model
        quantization_config = BitsAndBytesConfig(load_in_8bit=True, load_in_4bit=False)
        base_model = T5ForConditionalGeneration.from_pretrained(
            self.base_path,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            offload_folder="offload",
            offload_state_dict=True,
        )

        # load a pre-trained model.
        self.model = PeftModel.from_pretrained(base_model, self.adapter_path, is_trainable=True)
        self.finetuned_config = LoraConfig.from_pretrained(pretrained_model_name_or_path=self.adapter_path)

    def load_datasets(self):
        self.train_dataset = Dataset.load_from_disk(self.train_path)
        self.validation_dataset = Dataset.load_from_disk(self.test_path)

    def create_tokenized_datasets(self):
        def tokenize_func(examples):
            # Format inputs with a clear prefix for QA task
            prompts = [f"answer this question: {q}" for q in examples["question"]]
            answers = examples["answer"]
            
            # Tokenize inputs
            model_inputs = self.tokenizer(
                prompts,
                max_length=512,  # Increased for longer questions/context
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )
            
            # Tokenize answers (labels)
            labels = self.tokenizer(
                answers,
                max_length=128,  # Shorter for answers
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            ).input_ids
            
            # Replace padding token id with -100 for loss calculation
            labels[labels == self.tokenizer.pad_token_id] = -100
            
            # Add labels to model inputs
            model_inputs['labels'] = labels
            
            return model_inputs

        # Create tokenized datasets
        tokenized_datasets = {}
        
        # Remove columns we don't need
        column_list = [col for col in self.train_dataset.column_names if col not in ["question", "answer"]]
        
        tokenized_datasets["train"] = self.train_dataset.map(
            tokenize_func,
            batched=True,
            remove_columns=column_list,
        )
        
        tokenized_datasets["validation"] = self.validation_dataset.map(
            tokenize_func,
            batched=True,
            remove_columns=column_list,
        )
        
        # Use DataCollatorForSeq2Seq instead of simple padding
        collate_fn = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=None,  # Will be set during training
            padding=True,
            return_tensors="pt"
        )
        
        return tokenized_datasets, collate_fn
        
################################################################################################

    def compute_gradient(self, tokenized_datasets, collate_fn):
        train_dataloader_stochastic = DataLoader(tokenized_datasets["train"], 
                                                  shuffle=False,
                                                  collate_fn=collate_fn,
                                                  batch_size=1)
        val_dataloader_stochastic = DataLoader(tokenized_datasets["validation"], 
                                                  shuffle=False,
                                                  collate_fn=collate_fn,
                                                  batch_size=1)
        
        def check_batch_integrity(dataloader, device):
            print("Checking batch integrity...")
            for step, batch in enumerate(dataloader):
                # Move batch to device
                print(f"Step {step} Batch Keys: {batch.keys()}")
                batch = {k: v.to(device) for k, v in batch.items()}

                # Check if keys exist
                if 'input_ids' not in batch or 'labels' not in batch:
                    print(f"Step {step}: Missing 'input_ids' or 'labels' in the batch.")
                    continue

                # Check shapes of tensors
                input_ids_shape = batch['input_ids'].shape
                labels_shape = batch['labels'].shape
                print(f"Step {step}: input_ids shape: {input_ids_shape}, labels shape: {labels_shape}")

                # Ensure tensors are not empty
                if batch['input_ids'].numel() == 0 or batch['labels'].numel() == 0:
                    print(f"Step {step}: Empty tensors detected in 'input_ids' or 'labels'.")
                    continue

                # Check for data mismatch
                if input_ids_shape != labels_shape:
                    print(f"Step {step}: Mismatch in shapes: input_ids ({input_ids_shape}) vs labels ({labels_shape}).")
                else:
                    print(f"Step {step}: Batch integrity verified.")

                # Limit the number of batches to check (optional)
                if step >= 5:  # Check the first 5 batches only
                    print("...stopping after 5 batches.")
                    break
                
        check_batch_integrity(train_dataloader_stochastic, self.device)
        check_batch_integrity(val_dataloader_stochastic, self.device)
        
        
        
        
        # Compute the gradient
        self.model.eval()
        tr_grad_dict = {}
        for step, batch in enumerate(tqdm(train_dataloader_stochastic)):
            self.model.zero_grad() # zeroing out gradient
            batch['labels'] = batch['input_ids']
            batch.to(self.device)
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            
            grad_dict={}
            for k, v in self.model.named_parameters():
                if 'lora_A' in k:
                    grad_dict[k]=v.grad.cpu()
                elif 'lora_B' in k:
                    # first index of shape indicates low-rank
                    grad_dict[k]=v.grad.cpu().T
                else:
                    pass
            tr_grad_dict[step]=grad_dict
            del grad_dict
            
        val_grad_dict = {}
        for step, batch in enumerate(tqdm(val_dataloader_stochastic)):
            self.model.zero_grad() # zeroing out gradient
            batch['labels'] = batch['input_ids']
            batch.to(self.device)
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            
            grad_dict={}
            for k, v in self.model.named_parameters():
                if 'lora_A' in k:
                    grad_dict[k]=v.grad.cpu()
                elif 'lora_B' in k:
                    # first index of shape indicates low-rank
                    grad_dict[k]=v.grad.cpu().T
                else:
                    pass
            val_grad_dict[step]=grad_dict    
            del grad_dict
            
        return tr_grad_dict, val_grad_dict

################################################################################################