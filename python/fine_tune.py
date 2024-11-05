import pandas as pd
from datasets import DatasetDict, load_from_disk
from transformers import AutoTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling

pretrained_model_path = "./pretrained_model"  
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
fine_tune_model = AutoModelForMaskedLM.from_pretrained(pretrained_model_path)
fine_tune_dataset = load_from_disk('./fine_tune_dataset')

def mask_if_condition(example):
    function_code = example['function_code']
    if 'if ' in function_code:
        function_code = function_code.replace('if ', '[MASK] ', 1)
    tokens = tokenizer(function_code, padding="max_length", truncation=True, max_length=128)
    return tokens

tokenized_fine_tune_train = fine_tune_dataset['train'].map(mask_if_condition, remove_columns=["function_code"])
tokenized_fine_tune_val = fine_tune_dataset['validation'].map(mask_if_condition, remove_columns=["function_code"])
tokenized_fine_tune_test = fine_tune_dataset['test'].map(mask_if_condition, remove_columns=["function_code"])


fine_tune_dataset = DatasetDict({
    "train": tokenized_fine_tune_train,
    "validation": tokenized_fine_tune_val,
    "test": tokenized_fine_tune_test
})

data_collator_fine_tune = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.0
)

training_args_fine_tune = TrainingArguments(
    output_dir="./fine_tuned_model",
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    num_train_epochs=1,  
    weight_decay=0.01,
    no_cuda=True,  
    logging_steps=1000,
    disable_tqdm=True
)

trainer_fine_tune = Trainer(
    model=fine_tune_model,
    args=training_args_fine_tune,
    train_dataset=fine_tune_dataset['train'],
    eval_dataset=fine_tune_dataset['validation'],
    data_collator=data_collator_fine_tune,
)
trainer_fine_tune.train()

results = trainer_fine_tune.evaluate(eval_dataset=fine_tune_dataset['test'])
print("Evaluation Results on Test Set:", results)

fine_tune_model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

#sample_code = "def check_positive(num): if num > 0: return True else: return False"
#masked_code = sample_code.replace("if ", "[MASK] ", 1)

#inputs = tokenizer(masked_code, return_tensors="pt")
#outputs = fine_tune_model(**inputs)
#predictions = outputs.logits.argmax(dim=-1)

#predicted_text = tokenizer.decode(predictions[0], skip_special_tokens=True)
#print("Original code:", sample_code)
#print("Masked code:", masked_code)
#print("Predicted code:", predicted_text)
