import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling

df = pd.read_csv('processed_files.csv')

if_functions = df[df['has_if'] == True]
non_if_functions = df[df['has_if'] == False]

sample_size = int(len(if_functions) * 0.05)
non_if_sample = non_if_functions.sample(n=sample_size, random_state=42)

balanced_df = pd.concat([if_functions, non_if_sample])

pre_train_df = balanced_df.sample(frac=0.75, random_state=42)
fine_tune_df = balanced_df.drop(pre_train_df.index)

fine_tune_eval_df = fine_tune_df.sample(frac=0.10, random_state=42)
fine_tune_test_df = fine_tune_df.drop(fine_tune_eval_df.index).sample(frac=0.10, random_state=42)
fine_tune_train_df = fine_tune_df.drop(fine_tune_eval_df.index).drop(fine_tune_test_df.index)

pre_train_dataset = Dataset.from_pandas(pre_train_df)
fine_tune_dataset = DatasetDict({
    "train": Dataset.from_pandas(fine_tune_train_df),
    "validation": Dataset.from_pandas(fine_tune_eval_df),
    "test": Dataset.from_pandas(fine_tune_test_df)
})
fine_tune_dataset.save_to_disk('./fine_tune_dataset')
model_name = "distilbert-base-uncased" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)


def tokenize_for_mlm(examples):
    return tokenizer(examples['function_code'], padding="max_length", truncation=True, max_length=128)

tokenized_pre_train = pre_train_dataset.map(tokenize_for_mlm, batched=True, remove_columns=["function_code"])
data_collator_pre_train = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

training_args_pre_train = TrainingArguments(
    output_dir="./pretrained_model",
    evaluation_strategy="no",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    num_train_epochs=1,
    no_cuda=True,
    logging_steps=1000,
    disable_tqdm=True  
)

trainer_pre_train = Trainer(
    model=model,
    args=training_args_pre_train,
    train_dataset=tokenized_pre_train,
    data_collator=data_collator_pre_train,
)

trainer_pre_train.train()


model.save_pretrained("./pretrained_model")
tokenizer.save_pretrained("./pretrained_model")




