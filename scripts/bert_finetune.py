# https://mccormickml.com/2019/07/22/BERT-fine-tuning/

import datasets
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
import torch


# Load a pre-trained BERT model for German text
model_name = "bert-base-german-cased"
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=7)
tokenizer = BertTokenizer.from_pretrained(model_name)
# TODO check for: do_lower_case = False


# Load prepared train test data set from disk (scipts/bert_prep_train_test.py)
train_df = datasets.load_dataset("csv", data_files="data/reviews/train.csv")


# Tokenize the text
def tokenize_dataset(data):
    return tokenizer(
        data["text"], truncation=True, padding="max_length", max_length=256
    )


train_tokenized = train_df.map(tokenize_dataset, batched=True)


# Split the preprocessed train dataset again in train and evaluate datasets
train_dataset, eval_dataset = (
    train_tokenized["train"]
    .class_encode_column("label")
    .train_test_split(test_size=0.1, stratify_by_column="label", seed=161)
    .values()
)

# https://huggingface.co/santiviquez/amazon-reviews-finetuning-bert-base-sentiment
# optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
# lr_scheduler_type: linear
# num_epochs: 2

# https://huggingface.co/deepset/gbert-large-sts
# batch_size = 16
# n_epochs = 4
# warmup_ratio = 0.1
# learning_rate = 2e-5
# lr_schedule = LinearWarmup

# Define training arguments
training_args = TrainingArguments(
    output_dir="data/bert_finetuning",
    eval_strategy="epoch",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    warmup_steps=500,  # Warm-up steps for learning rate scheduler
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Data collator to dynamically pad the inputs received
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Fine-tune the model
trainer.train()

# Save the model
trainer.save_model("./fine_tuned_model")


################################################################################
# test fine-tuned model


# Load the fine-tuned model and tokenizer
model_name = "data/bert_finetuning/colab_240924"  # path to fine-tuned model
model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Define a Trainer for evaluation
trainer = Trainer(model=model, tokenizer=tokenizer)


# Load the test data (assuming it's in a CSV file)
test_dataset = datasets.load_dataset(
    "csv", data_files={"test": "data/reviews/reviews_sent_hc_06.csv"}
)


# Tokenize the text
def preprocess_data(examples):
    return tokenizer(
        examples["text"], truncation=True, padding="max_length", max_length=128
    )


tokenized_test_dataset = test_dataset.map(preprocess_data, batched=True)


# Make predictions
predictions = trainer.predict(tokenized_test_dataset["test"])

# Retrive predicted labels
predicted_labels = torch.argmax(torch.tensor(predictions.predictions), axis=1)
# TODO correct manually: befor was numpy.ndarray -> why error

reviews_hc["labels_predicted"] = predicted_labels.tolist()

# %view reviews_hc

# Compare with true labels
true_labels = tokenized_test_dataset["test"]["labels"]

tokenized_test_dataset["test"]["attention_mask"][0]


# Calculate additional metrics (optional)
from sklearn.metrics import classification_report

report = classification_report(
    true_labels,
    predicted_labels,
    target_names=["class0", "class1", "class2", "class3", "class4", "class5", "class6"],
)
print(report)
