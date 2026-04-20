import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

# 1. Load our cleaned data
train_df = pd.read_csv('data/train_cleaned.csv').dropna()
test_df = pd.read_csv('data/test_cleaned.csv').dropna()

# 2. Setup the Tokenizer
# 'uncased' means it doesn't care about Capital Letters
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class NewsDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=128, return_tensors="pt")
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Prepare datasets
train_dataset = NewsDataset(train_df['text'], train_df['label'].values)
test_dataset = NewsDataset(test_df['text'], test_df['label'].values)

# 3. Load the Model
# We add 'num_labels=2' because we only have two categories: Real or Fake
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# 4. Define Training Settings (The Developer Logic)
training_args = TrainingArguments(
    output_dir='./models/results',
    num_train_epochs=1,
    per_device_train_batch_size=4,   # Lowered this slightly to be safe
    eval_strategy="steps",           # Fixed the name here
    eval_steps=100,
    logging_dir='./logs',
)

# 5. The Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

print("Starting training... Go grab a coffee, this takes a bit!")
trainer.train()
