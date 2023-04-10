import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# Load the pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Load and preprocess your dataset
train_file = "train.txt"
val_file = "val.txt"

train_dataset = TextDataset(tokenizer=tokenizer, file_path=train_file, block_size=128)
val_dataset = TextDataset(tokenizer=tokenizer, file_path=val_file, block_size=128)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Define training configuration
training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    eval_steps=400,
    save_steps=800,
    warmup_steps=200,
    prediction_loss_only=True,
)

# Train the model
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

# Save the fine-tuned model
trainer.save_model("./output")