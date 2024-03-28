# create virtual environment
# conda create --name train transformers transformers[torch]  accelerate --channel conda-

# homo/
# - train/train.py
# -      /train.txt
# - gpt-neo-125m/
# - (models)

# we are in train. one level above is ../gpt

from transformers import AutoTokenizer, AutoModelForCausalLM, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments, pipeline

tokenizer = AutoTokenizer.from_pretrained("/scratch/network/fc1991/homo/train/gpt-neo-125m", local_files_only=True)
model = AutoModelForCausalLM.from_pretrained("/scratch/network/fc1991/homo/train/gpt-neo-125m", local_files_only=True)

train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path='/scratch/network/fc1991/homo/train/train.txt',
    block_size=128
)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)
training_args = TrainingArguments(
    output_dir= '/',
    # overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=4,
)
trainer = Trainer(
    model=model,
    # args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

trainer.train()

tokenizer.save_pretrained('../models')
trainer.save_model('../models', 'pytorch_model')
