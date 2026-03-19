# modules/bert_classifier.py
import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.optim import AdamW
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class LIARDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length     = config.BERT_MAX_LENGTH,
            truncation     = True,
            padding        = "max_length",
            return_tensors = "pt"
        )
        return {
            "input_ids"      : encoding["input_ids"].squeeze(),
            "attention_mask" : encoding["attention_mask"].squeeze(),
            "label"          : torch.tensor(self.labels[idx], dtype=torch.long)
        }


def load_processed_data():
    train_df = pd.read_csv(os.path.join(config.DATA_PROCESSED, "train.csv"))
    val_df   = pd.read_csv(os.path.join(config.DATA_PROCESSED, "val.csv"))
    test_df  = pd.read_csv(os.path.join(config.DATA_PROCESSED, "test.csv"))

    train_df = train_df.dropna(subset=["binary_label", "combined_text"])
    val_df   = val_df.dropna(subset=["binary_label", "combined_text"])
    test_df  = test_df.dropna(subset=["binary_label", "combined_text"])

    # DEV MODE — small sample for fast iteration
    if config.DEV_MODE:
        train_df = train_df.sample(config.DEV_SAMPLE_SIZE, random_state=config.RANDOM_SEED)
        val_df   = val_df.sample(100, random_state=config.RANDOM_SEED)
        test_df  = test_df.sample(100, random_state=config.RANDOM_SEED)
        print(f"DEV MODE: using {config.DEV_SAMPLE_SIZE} train samples")

    print(f"Train : {len(train_df)} rows")
    print(f"Val   : {len(val_df)} rows")
    print(f"Test  : {len(test_df)} rows")

    return train_df, val_df, test_df


def train(model, dataloader, optimizer, device):
    """One full pass through training data = one epoch."""
    model.train()
    total_loss = 0

    # tqdm wraps the dataloader and shows a live progress bar
    progress = tqdm(dataloader, desc="  Training", leave=False)

    for batch in progress:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            labels         = labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        # Show live loss in progress bar
        progress.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluates model. Returns accuracy, F1, predictions, labels."""
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  Evaluating", leave=False):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            outputs = model(
                input_ids      = input_ids,
                attention_mask = attention_mask
            )

            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="weighted")

    return acc, f1, all_preds, all_labels


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    print("\nLoading processed data...")
    train_df, val_df, test_df = load_processed_data()

    print("\nLoading RoBERTa tokenizer and model...")
    tokenizer = RobertaTokenizer.from_pretrained(config.BERT_MODEL_NAME)
    model     = RobertaForSequenceClassification.from_pretrained(
        config.BERT_MODEL_NAME,
        num_labels = 2
    )
    model.to(device)

    train_dataset = LIARDataset(
        train_df["combined_text"].tolist(),
        train_df["binary_label"].astype(int).tolist(),
        tokenizer
    )
    val_dataset = LIARDataset(
        val_df["combined_text"].tolist(),
        val_df["binary_label"].astype(int).tolist(),
        tokenizer
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size = config.BERT_BATCH_SIZE,
        shuffle    = True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size = config.BERT_BATCH_SIZE,
        shuffle    = False
    )

    optimizer = AdamW(model.parameters(), lr=config.BERT_LR)

    print(f"\nFine-tuning RoBERTa for {config.BERT_EPOCHS} epochs...")
    print("-" * 50)

    best_f1    = 0
    best_epoch = 0

    for epoch in range(config.BERT_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.BERT_EPOCHS}")

        train_loss            = train(model, train_loader, optimizer, device)
        val_acc, val_f1, _, _ = evaluate(model, val_loader, device)

        print(f"  Train Loss : {train_loss:.4f}")
        print(f"  Val Acc    : {val_acc:.4f}")
        print(f"  Val F1     : {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1    = val_f1
            best_epoch = epoch + 1
            os.makedirs(config.MODELS_DIR, exist_ok=True)
            model.save_pretrained(os.path.join(config.MODELS_DIR, "roberta_liar"))
            tokenizer.save_pretrained(os.path.join(config.MODELS_DIR, "roberta_liar"))
            print(f"  ✓ Best model saved (F1={best_f1:.4f})")

    print(f"\nBest model: Epoch {best_epoch} with F1 = {best_f1:.4f}")

    # Final test evaluation
    print("\nLoading best model for test evaluation...")
    best_model = RobertaForSequenceClassification.from_pretrained(
        os.path.join(config.MODELS_DIR, "roberta_liar")
    )
    best_model.to(device)

    test_dataset = LIARDataset(
        test_df["combined_text"].tolist(),
        test_df["binary_label"].astype(int).tolist(),
        tokenizer
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size = config.BERT_BATCH_SIZE,
        shuffle    = False
    )

    test_acc, test_f1, test_preds, test_labels = evaluate(
        best_model, test_loader, device
    )

    print("\n" + "="*50)
    print("  FINAL TEST RESULTS")
    print("="*50)
    print(f"  Accuracy : {test_acc:.4f}")
    print(f"  F1 Score : {test_f1:.4f}")
    print("\nDetailed Report:")
    print(classification_report(
        test_labels, test_preds,
        target_names=config.LABEL_NAMES
    ))

    os.makedirs(config.TABLES_DIR, exist_ok=True)
    results = {
        "model"     : config.BERT_MODEL_NAME,
        "accuracy"  : round(test_acc, 4),
        "f1_score"  : round(test_f1, 4),
        "epochs"    : config.BERT_EPOCHS,
        "best_epoch": best_epoch
    }
    pd.DataFrame([results]).to_csv(
        os.path.join(config.TABLES_DIR, "bert_results.csv"),
        index=False
    )
    print("\nResults saved to results/tables/bert_results.csv")