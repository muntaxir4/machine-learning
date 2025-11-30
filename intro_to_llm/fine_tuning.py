from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# Create the small DataCamp toy dataset (train: 4 rows, test: 1 row)
train_data = Dataset.from_dict(
    {
        "interaction": [
            "I'm really upset with the delays on delivering this item. Where is it?",
            "The support I've had on this issue has been terrible and really unhelpful. Why will no one help me?",
            "I have a question about how to use this product. Can you help me?",
            "This product is listed as out of stock. When will it be available again?",
        ],
        "risk": ["high risk", "high risk", "low risk", "low risk"],
    }
)

test_data = Dataset.from_dict(
    {
        "interaction": ["You charged me twice for the one item. I need a refund."],
        "risk": ["high risk"],
    }
)


# Keep the human-readable 'risk' column but add an integer 'labels' column
# Trainer expects a 'labels' column containing ints for computing loss.
def add_labels(batch):
    # batched=True yields lists of strings, map should return dict with lists
    batch["labels"] = [1 if r == "high risk" else 0 for r in batch["risk"]]
    return batch


train_data = train_data.map(add_labels, batched=True)
test_data = test_data.map(add_labels, batched=True)

# Load the model and tokenizer
# Binary classification -> tell the model there are 2 labels so it returns loss
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


# Tokenize using Dataset.map (keeps label alignment)
def tokenize_fn(batch):
    # Return tokenized lists (not tensors) so the Trainer/data collator will handle
    # conversion to tensors when batching. Returning 'pt' tensors here makes the
    # dataset contain tensor objects which breaks the Trainer's collate behavior.
    return tokenizer(batch["interaction"], truncation=True, padding=True, max_length=20)


tokenized_training_data = train_data.map(tokenize_fn, batched=True)
tokenized_test_data = test_data.map(tokenize_fn, batched=True)

print(tokenized_training_data[0])

# Set up an instance of TrainingArguments
training_args = TrainingArguments(
    output_dir="./finetuned2",
    # Set the evaluation strategy
    eval_strategy="epoch",
    # Specify the number of epochs
    num_train_epochs=3,
    learning_rate=2e-5,
    # Set the batch sizes
    per_device_train_batch_size=3,
    per_device_eval_batch_size=3,
    weight_decay=0.01,
    use_cpu=False,
)

trainer = Trainer(
    model=model,
    # Assign the training arguments and tokenizer
    args=training_args,
    train_dataset=tokenized_training_data,
    eval_dataset=tokenized_test_data,
    # tokenizer=tokenizer,
)

# Train the model
trainer.train()
