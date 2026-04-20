import pandas as pd
import re
from sklearn.model_selection import train_test_split

# 1. Load the data
print("Loading datasets...")
true_df = pd.read_csv('../data/True.csv')
fake_df = pd.read_csv('../data/Fake.csv')

# 2. Label the data (0 for Real, 1 for Fake)
true_df['label'] = 0
fake_df['label'] = 1

# 3. Combine them into one big list
df = pd.concat([true_df, fake_df])

# 4. Clean the text
# This removes the "Reuters" tags so the AI doesn't cheat!
def clean_text(text):
    text = text.lower()
    text = re.sub(r'^[^-]*-\s*', '', text) # Removes city/agency headers
    text = re.sub(r'\W', ' ', text)        # Removes special characters
    return text.strip()

print("Cleaning text (this might take a minute)...")
df['text'] = df['text'].apply(clean_text)

# 5. Split for training and testing
# We'll use 80% to train the AI and 20% to test it later
train, test = train_test_split(df, test_size=0.2, random_state=42)

# 6. Save the cleaned versions
train.to_csv('../data/train_cleaned.csv', index=False)
test.to_csv('../data/test_cleaned.csv', index=False)

print("Success! Cleaned data saved to the data/ folder.")
