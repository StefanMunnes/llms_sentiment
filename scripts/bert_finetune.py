import datasets
from transformers import (
    BertTokenizer,
    XLMRobertaTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)


# Load prepared train test data set from disk (scipts/bert_prep_train_test.py)
train_df = datasets.load_dataset("csv", data_files="data/reviews/train.csv")


# Define models to finetune
models = [
    "google-bert/bert-base-german-cased",
    "deepset/gbert-base",
    "FacebookAI/xlm-roberta-base",
    "distilbert/distilbert-base-german-cased",
]


# Define function for tokenization
def tokenize_dataset(data):
    return tokenizer(
        data["text"], truncation=True, padding="max_length", max_length=256
    )


for model_name in models:
    print(model_name)

    # Load the pre-trained BERT (like) model
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=7)

    # Load the tokenizer
    if model_name == "FacebookAI/xlm-roberta-base":
        tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
    else:
        tokenizer = BertTokenizer.from_pretrained(model_name)

    # Tokenize the text
    train_tokenized = train_df.map(tokenize_dataset, batched=True)

    # Split the tokenized train dataset in train and evaluate datasets
    train_dataset, eval_dataset = (
        train_tokenized["train"]
        .class_encode_column("label")
        .train_test_split(test_size=0.1, stratify_by_column="label", seed=161)
        .values()
    )

    # Data collator to dynamically pad the inputs received
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=f"data/{model_name.split('/')[1]}",
        eval_strategy="epoch",
        num_train_epochs=8,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

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
    trainer.save_model(f"data/fine_tuned_model/{model_name.split('/')[1]}")

    !zip -r f"{model_name.split('/')[1]}.zip" f"data/fine_tuned_model/{model_name.split('/')[1]}"


# https://mccormickml.com/2019/07/22/BERT-fine-tuning/

# https://huggingface.co/deepset/gbert-large-sts
# batch_size = 16
# n_epochs = 4
# warmup_ratio = 0.1
# learning_rate = 2e-5
# lr_schedule = LinearWarmup
