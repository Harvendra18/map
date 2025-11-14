# 🗺️ Misconception Mapping using DeBERTa (Kaggle Competition)

This repository contains the full pipeline for a multi-label classification task, specifically addressing the Kaggle competition **"Misconception Mapping: Charting Student Math Misunderstandings"**.

The goal is to correctly classify student responses based on their mathematical misconception (`Category:Misconception`), treating it as a **multi-label classification** problem to predict the top-K most likely labels.

## 🚀 Overview

The core script, `map.py`, leverages the **Hugging Face `transformers`** library, specifically using a **DeBERTa-v3-small** base model for fine-tuning. The entire process is designed to run efficiently within a Kaggle notebook environment, including the crucial step of loading model weights and tokenizer **offline** from an attached input dataset.

### Key Features
* **Offline Model Loading:** Uses `local_files_only=True` for robust execution in Kaggle's environment without requiring an internet connection during the run phase.
* **Multi-Label Classification:** Configures the model and training arguments for multi-label prediction.
* **Custom Dataset:** Implements `MisconceptionDataset` to correctly handle the text inputs and the one-hot encoded label matrix.
* **F2-Score Focus:** Defines a `compute_metrics` function to track the $F_\beta$ score (with $\beta=2$) which is typically the metric used in this type of competition.
* **Top-K Prediction:** Generates the final submission by selecting the **top 3** most probable labels from the sigmoid outputs.

## 📦 Prerequisites

To run this script successfully, you will need the following:

1.  **Python 3.x**
2.  **Required Libraries:** `pandas`, `numpy`, `torch`, `scikit-learn`, and `transformers`.
    ```bash
    pip install pandas numpy torch scikit-learn transformers
    ```
3.  **Kaggle Environment:** This script is optimized for the Kaggle notebook environment.

    * **Data:** The official competition data must be available at `/kaggle/input/map-charting-student-math-misunderstandings/`.
    * **Pre-trained Weights:** You **must** attach a Kaggle Dataset containing the DeBERTa model weights (e.g., the output from a previous training run or a dedicated model weights dataset). The script expects them at the configured path:
        ```python
        BASE_MODEL_DIR = "/kaggle/input/deberta-starter-cv-0-930/ver_1/best"
        ```

## ⚙️ Configuration (in `map.py`)

The script utilizes several configurable constants:

| Variable | Description | Value (Default) |
| :--- | :--- | :--- |
| `BASE_MODEL_DIR` | Path to the local model weights (must be attached). | `/kaggle/input/...` |
| `MAX_LEN` | Maximum sequence length for tokenization. | `512` |
| `EPOCHS` | Number of training epochs. | `4` |
| `TRAIN_BATCH_SIZE` | Batch size per device during training. | `6` |
| `TOP_K` | Number of top predicted labels for submission. | `3` |

## 📝 Script Walkthrough (`map.py`)

### 1. Data Preparation
* The `Category` and `Misconception` columns are combined into a single `Target` label (e.g., `Category:Misconception`).
* `MultiLabelBinarizer` is used to create the one-hot encoded training labels (`Y_train`).
* The input text is formatted by concatenating the **Question**, **MC Answer**, and **Student Explanation** into a single sequence for the model.

### 2. Tokenization & Dataset
* The `AutoTokenizer` is loaded offline.
* The `MisconceptionDataset` class is instantiated to manage input IDs, attention masks, and the float-type label tensors.

### 3. Model & Trainer
* The `AutoModelForSequenceClassification` is loaded offline, configured with the correct number of labels (`num_labels`) and `problem_type="multi_label_classification"`.
* A `Trainer` instance is set up with standard **TrainingArguments**, including `fp16` for faster training if a GPU is available.

### 4. Training & Prediction
* `trainer.train()` executes the fine-tuning process.
* `trainer.predict(test_dataset)` generates raw logits.
* The logits are passed through the **sigmoid function** to obtain probabilities.
* For each prediction, `np.argsort` is used to find the indices of the **top 3** highest probabilities, which are then mapped back to the original `Category:Misconception` labels using the `mlb.classes_` array.
* Finally, a `submission.csv` file is created with the required format.

## 🎯 How to Run

1.  Clone this repository:
    ```bash
    git clone [YOUR_REPO_URL]
    ```
2.  Create a new Kaggle Notebook.
3.  Add the necessary **Data** and **Model Weights** (output) as inputs.
4.  Copy the contents of `map.py` into a code cell and run it.

The script will handle the training and automatically generate a `submission.csv` file, ready for submission to the competition.
