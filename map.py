import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import fbeta_score 
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

# ================================================================
# 1. CONFIGURATION
# ================================================================

# CHECK THIS PATH: IT MUST MATCH THE SLUG OF THE DATASET YOU ATTACH!
BASE_MODEL_DIR = "/kaggle/input/deberta-starter-cv-0-930/ver_1/best"
KAGGLE_DATA_PATH = "/kaggle/input/map-charting-student-math-misunderstandings/"

MAX_LEN = 512
SEED = 42
TRAIN_BATCH_SIZE = 6
EPOCHS = 4
TOP_K = 3

torch.manual_seed(SEED)
np.random.seed(SEED)

# --- Check for attached datasets and set LOCAL_MODEL_PATH ---
print("--- Checking Kaggle Input Directory Contents ---")
try:
    print(os.listdir("/kaggle/input"))
    
    # 🚨 CRITICAL FIX: Since the model files are likely directly attached
    # at the BASE_MODEL_DIR level, we'll use it directly if it exists.
    if "deberta-v3-small-offline" in os.listdir("/kaggle/input"):
        LOCAL_MODEL_PATH = BASE_MODEL_DIR
    else:
        # Fallback to the subfolder logic (less common for DeBERTa on Kaggle)
        subfolders = [f.path for f in os.scandir(BASE_MODEL_DIR) if f.is_dir()]
        LOCAL_MODEL_PATH = subfolders[0] if subfolders else BASE_MODEL_DIR

except FileNotFoundError:
    print(f"FATAL ERROR: Model directory not found at {BASE_MODEL_DIR}. You MUST click '+ Add Input' and attach the model weights dataset.")
    raise # Stop execution as we cannot run offline without the files.

print(f"📁 Using local model path: {LOCAL_MODEL_PATH}")
print("----------------------------------------------")

# ================================================================
# 2. DATA LOADING & PREPROCESSING
# ================================================================

train_df = pd.read_csv(os.path.join(KAGGLE_DATA_PATH, "train.csv"))
test_df = pd.read_csv(os.path.join(KAGGLE_DATA_PATH, "test.csv"))

# Create unique target label
train_df['Misconception'] = train_df['Misconception'].fillna('NA')
train_df['Target'] = train_df['Category'] + ":" + train_df['Misconception']
train_targets_lists = [[label] for label in train_df['Target'].tolist()]

mlb = MultiLabelBinarizer()
Y_train = mlb.fit_transform(train_targets_lists).astype(np.float32)
all_labels = mlb.classes_
num_labels = len(all_labels)

def format_input(row):
    return(
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        f"Explanation: {row['StudentExplanation']}"
    )

train_df['input_text'] = train_df.apply(format_input, axis=1)
test_df['input_text'] = test_df.apply(format_input, axis=1)
X_train = train_df['input_text'].tolist()
X_test = test_df['input_text'].tolist()

# ================================================================
# 3. TOKENIZATION (OFFLINE)
# ================================================================

# This will now successfully load the tokenizer config and vocabulary offline
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)

raw_train_encodings = tokenizer(
    X_train, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt"
)

raw_test_encodings = tokenizer(
    X_test, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt"
)

# ================================================================
# 4. DATASET CLASS & METRICS (Simplified & Corrected)
# ================================================================

class MisconceptionDataset(Dataset):
    def _init_(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def _getitem_(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float) 
        return item

    def _len_(self):
        return len(self.encodings["input_ids"])

train_dataset = MisconceptionDataset(raw_train_encodings, Y_train)
dummy_test_labels = np.zeros((len(X_test), num_labels), dtype=np.float32)
test_dataset = MisconceptionDataset(raw_test_encodings, dummy_test_labels)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def compute_metrics(p):
    logits = p.predictions
    probas = 1.0 / (1.0 + np.exp(-logits))
    preds = (probas > 0.5).astype(int)
    f2 = fbeta_score(p.label_ids, preds, beta=2, average='micro', zero_division=0)
    return {"f2_score_micro": f2}

# ================================================================
# 5. MODEL & TRAINER (OFFLINE)
# ================================================================

model = AutoModelForSequenceClassification.from_pretrained(
    LOCAL_MODEL_PATH,
    num_labels=num_labels,
    problem_type="multi_label_classification",
    local_files_only=True
)
USE_GPU = torch.cuda.is_available()

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=TRAIN_BATCH_SIZE * 2,
    warmup_steps=500,
    weight_decay=0.05,
    learning_rate=1.5e-5,
    logging_dir="./logs",
    logging_steps=50,
    save_strategy="epoch",
    report_to="none",
    fp16=True if USE_GPU else False,
)

trainer = Trainer(
    model=model, 
    args=training_args, 
    train_dataset=train_dataset, 
    data_collator=data_collator
)

# ================================================================
# 6. TRAINING & SUBMISSION (FIXED)
# ================================================================

print("\n--- Starting Model Training (Offline) ---")
trainer.train()
print("✅ Training Complete.")

# Prediction
raw_predictions = trainer.predict(test_dataset).predictions
probabilities = torch.sigmoid(torch.tensor(raw_predictions)).numpy()

final_predictions = []
for row_probs in probabilities:
    top_indices = np.argsort(row_probs)[::-1][:TOP_K]
    top_labels = [all_labels[i] for i in top_indices]
    final_predictions.append(" ".join(top_labels))

# FIXED: Create the submission DataFrame AFTER the loop is complete
submission_df = pd.DataFrame({
    "row_id": test_df["row_id"],
    "Category:Misconception": final_predictions
})

submission_file = "submission.csv"
submission_df.to_csv(submission_file, index=False)
try:
    print(submission_df.head().to_string()) # Use .to_string() for explicit printing
except Exception as e:
    print(f"Warning: Could not print submission head due to error: {e}")
    
# Add cleanup to free memory right after training
import gc
import torch
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# print(f"\n✅ Submission file '{submission_file}' generated successfully!")
# print("--- Sample Submission ---")
# print(submission_df.head())