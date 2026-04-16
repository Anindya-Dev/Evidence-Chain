"""Stacking ensemble that fuses classifier and RAG signals."""

import os
import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from transformers import RobertaTokenizer, RobertaForSequenceClassification

import config
from src.preprocessing.preprocessor import build_model_input, extract_claim_text
from src.rag.decomposer import ClaimDecomposer
from src.rag.reasoner import EvidenceReasoner
from src.rag.retriever import Retriever


class StackingEnsemble:
    """
    Meta-classifier that combines BERT and RAG outputs.

    Input features:
      1. bert_prob          - BERT probability of REAL (0-1)
      2. rag_confidence     - RAG verdict confidence (0-1)
      3. max_similarity     - best evidence similarity score
      4. source_weight_avg  - average source credibility
      5. hallucination_flag - 1 if hallucinated, 0 if not
      6. sub_claim_count    - how many sub-claims were found

    Why Logistic Regression as meta-classifier?
    Simple, interpretable, works well with small feature sets.
    SHAP values are meaningful on LR - directly shows which
    feature drove the final decision.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.classifier = LogisticRegression(
            random_state=config.RANDOM_SEED,
            max_iter=1000,
            C=1.0
        )
        self.is_trained = False
        self.feature_names = [
            "bert_prob",
            "rag_confidence",
            "max_similarity",
            "source_weight_avg",
            "hallucination_flag",
            "sub_claim_count"
        ]

    def build_feature_vector(self, bert_prob, rag_result):
        """
        Builds input feature vector from BERT and RAG outputs.
        """

        hallucination = 1.0 if rag_result.get("hallucination_flag") else 0.0

        rag_conf = rag_result.get("final_confidence", 0.5)
        if rag_result.get("final_verdict") == "FALSE":
            rag_conf = 1.0 - rag_conf

        return np.array([[
            bert_prob,
            rag_conf,
            rag_result.get("max_similarity", 0.0),
            rag_result.get("source_weight_avg", 0.5),
            hallucination,
            float(rag_result.get("sub_claim_count", 1))
        ]])

    def train(self, feature_matrix, labels):
        """
        Trains the meta-classifier on validation set features.
        """

        print("Training stacking ensemble...")
        print(f"  Samples  : {len(labels)}")
        print(f"  Features : {self.feature_names}")

        X_scaled = self.scaler.fit_transform(feature_matrix)
        self.classifier.fit(X_scaled, labels)
        self.is_trained = True

        train_preds = self.classifier.predict(X_scaled)
        train_acc = accuracy_score(labels, train_preds)
        train_f1 = f1_score(labels, train_preds, average="weighted")

        print(f"  Train Acc : {train_acc:.4f}")
        print(f"  Train F1  : {train_f1:.4f}")

        print("\n  Feature Coefficients:")
        for name, coef in zip(self.feature_names, self.classifier.coef_[0]):
            direction = "-> REAL" if coef > 0 else "-> FAKE"
            print(f"    {name:<22} : {coef:+.4f}  {direction}")

    def predict(self, feature_matrix):
        """
        Predicts final verdict for given features.
        """

        if not self.is_trained:
            raise ValueError("Ensemble not trained yet. Call train() first.")

        X_scaled = self.scaler.transform(feature_matrix)
        predictions = self.classifier.predict(X_scaled)
        probabilities = self.classifier.predict_proba(X_scaled)[:, 1]

        return predictions, probabilities

    def predict_single(self, bert_prob, rag_result):
        """
        Predicts verdict for a single claim.
        Convenience wrapper for pipeline use.
        """

        features = self.build_feature_vector(bert_prob, rag_result)
        preds, probs = self.predict(features)

        verdict = "REAL" if preds[0] == 1 else "FAKE"
        confidence = float(probs[0]) if preds[0] == 1 else 1.0 - float(probs[0])

        return {
            "final_verdict"  : verdict,
            "confidence"     : round(confidence, 4),
            "bert_prob"      : round(bert_prob, 4),
            "rag_verdict"    : rag_result.get("final_verdict"),
            "rag_confidence" : rag_result.get("final_confidence"),
            "hallucination"  : rag_result.get("hallucination_flag")
        }

    def save(self, path=None):
        """Saves trained ensemble to disk."""
        if path is None:
            path = config.get_preferred_ensemble_path()

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "classifier": self.classifier,
                "scaler"    : self.scaler,
                "features"  : self.feature_names
            }, f)
        print(f"Ensemble saved to {path}")

    def load(self, path=None):
        """Loads trained ensemble from disk."""
        if path is None:
            path = config.get_ensemble_path()

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.classifier = data["classifier"]
        self.scaler = data["scaler"]
        self.feature_names = data["features"]
        self.is_trained = True
        print(f"Ensemble loaded from {path}")


def _resolve_validation_assets(dataset=None):
    """Returns the validation split and checkpoint paths for a dataset."""

    dataset = config._normalize_dataset_name(dataset)
    data_path = config.get_processed_split_path("val", dataset)
    model_path = config.get_roberta_model_dir(dataset)

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Missing {dataset.upper()} validation data: {data_path}"
        )
    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"Missing {dataset.upper()} RoBERTa checkpoint: {model_path}"
        )

    return data_path, model_path


def _resolve_text_column(df):
    """Returns the primary claim column for feature-table inspection."""

    if "statement" in df.columns:
        return "statement"
    if "title" in df.columns:
        return "title"
    if "combined_text" in df.columns:
        return "combined_text"
    raise KeyError("Expected 'statement', 'title', or 'combined_text'.")


def _get_bert_probability(text, model, tokenizer, device, dataset=None):
    """Returns RoBERTa probability of REAL for one claim."""

    model_input = build_model_input(text, dataset=dataset)
    encoding = tokenizer(
        model_input,
        max_length=config.BERT_MAX_LENGTH,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(
            input_ids=encoding["input_ids"].to(device),
            attention_mask=encoding["attention_mask"].to(device)
        )

    probs = torch.softmax(outputs.logits, dim=1)
    return probs[0][1].item()


def _run_rag_branch(example, decomposer, retriever, reasoner, dataset=None):
    """
    Runs the claim decomposition + retrieval + reasoning branch for one
    dataset example and returns an aggregated RAG result.
    """

    clean_claim = extract_claim_text(example, dataset=dataset)
    sub_claims = decomposer.decompose(clean_claim)
    sub_results = []

    for sub_claim in sub_claims:
        evidence = retriever.retrieve(sub_claim, top_k=config.TOP_K_RETRIEVAL)
        result = reasoner.reason(sub_claim, evidence)

        if evidence:
            result["source_weight_avg"] = round(
                sum(e["source_weight"] for e in evidence) / len(evidence), 4
            )
            result["max_similarity"] = max(e["similarity"] for e in evidence)
        else:
            result["source_weight_avg"] = 0.3
            result["max_similarity"] = 0.0

        sub_results.append(result)

    rag_result = reasoner.aggregate_sub_claim_verdicts(sub_results)
    rag_result["max_similarity"] = round(
        sum(r["max_similarity"] for r in sub_results) / len(sub_results), 4
    )
    rag_result["source_weight_avg"] = round(
        sum(r["source_weight_avg"] for r in sub_results) / len(sub_results), 4
    )

    return rag_result


def train_ensemble_on_validation(sample_size=None, dataset=None):
    """
    Generates real validation features from the active validation set,
    trains the ensemble, and saves both the trained model and feature
    table for inspection.
    """

    dataset = config._normalize_dataset_name(dataset)
    data_path, model_path = _resolve_validation_assets(dataset)

    if sample_size is None:
        if config.ENSEMBLE_SAMPLE_SIZE > 0:
            sample_size = config.ENSEMBLE_SAMPLE_SIZE
        elif config.CHEAP_MODE:
            sample_size = config.CHEAP_EVAL_SAMPLE_SIZE

    print("=" * 65)
    print(
        f"  TRAINING STACKING ENSEMBLE ON {dataset.upper()} VALIDATION FEATURES"
    )
    print("=" * 65)
    print(f"Validation data : {data_path}")
    print(f"RoBERTa path    : {model_path}")
    if sample_size:
        print(f"Sample size     : {sample_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device          : {device}")

    val_df = pd.read_csv(data_path)
    text_column = _resolve_text_column(val_df)
    val_df = val_df.dropna(subset=["binary_label", text_column]).reset_index(drop=True)

    if sample_size:
        val_df = val_df.sample(sample_size, random_state=config.RANDOM_SEED)
        val_df = val_df.reset_index(drop=True)

    print(f"Validation rows : {len(val_df)}")

    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    bert_model = RobertaForSequenceClassification.from_pretrained(model_path)
    bert_model.to(device)
    bert_model.eval()

    decomposer = ClaimDecomposer()
    retriever = Retriever()
    reasoner = EvidenceReasoner()
    ensemble = StackingEnsemble()

    feature_rows = []

    for idx, row in val_df.iterrows():
        example = row
        label = int(row["binary_label"])

        try:
            rag_result = _run_rag_branch(
                example, decomposer, retriever, reasoner, dataset=dataset
            )
            bert_prob = _get_bert_probability(
                example, bert_model, tokenizer, device, dataset=dataset
            )
            feature_vector = ensemble.build_feature_vector(
                bert_prob, rag_result
            )[0]

            feature_rows.append({
                "claim"              : extract_claim_text(example, dataset=dataset),
                "label"              : label,
                "bert_prob"          : feature_vector[0],
                "rag_confidence"     : feature_vector[1],
                "max_similarity"     : feature_vector[2],
                "source_weight_avg"  : feature_vector[3],
                "hallucination_flag" : feature_vector[4],
                "sub_claim_count"    : feature_vector[5],
                "rag_verdict"        : rag_result["final_verdict"],
                "rag_raw_confidence" : rag_result["final_confidence"]
            })

            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{len(val_df)}")

        except Exception as exc:
            print(f"  Warning: failed on validation row {idx} - {exc}")

    if not feature_rows:
        raise RuntimeError("No validation features were generated.")

    features_df = pd.DataFrame(feature_rows)
    X = features_df[ensemble.feature_names].to_numpy(dtype=float)
    y = features_df["label"].to_numpy(dtype=int)

    holdout_metrics = {
        "holdout_samples": 0,
        "holdout_accuracy": None,
        "holdout_f1": None,
    }

    # Report a small honest holdout estimate before refitting on the full
    # validation feature table that we save for downstream inference.
    class_counts = np.bincount(y) if len(y) else np.array([])
    if len(y) >= 20 and len(class_counts) >= 2 and np.min(class_counts) >= 2:
        X_train, X_holdout, y_train, y_holdout = train_test_split(
            X,
            y,
            test_size=0.2,
            stratify=y,
            random_state=config.RANDOM_SEED
        )

        holdout_ensemble = StackingEnsemble()
        holdout_ensemble.train(X_train, y_train)
        holdout_preds, _ = holdout_ensemble.predict(X_holdout)
        holdout_metrics = {
            "holdout_samples": int(len(y_holdout)),
            "holdout_accuracy": round(accuracy_score(y_holdout, holdout_preds), 4),
            "holdout_f1": round(
                f1_score(y_holdout, holdout_preds, average="weighted"), 4
            ),
        }

        print("\nHoldout metrics before final refit:")
        for key, value in holdout_metrics.items():
            print(f"  {key:<16}: {value}")

    ensemble.train(X, y)
    preds, _ = ensemble.predict(X)

    metrics = {
        "dataset"  : dataset,
        "samples"  : int(len(features_df)),
        "accuracy" : round(accuracy_score(y, preds), 4),
        "f1"       : round(f1_score(y, preds, average="weighted"), 4),
        **holdout_metrics,
    }

    print("\nValidation metrics after ensemble training:")
    for key, value in metrics.items():
        print(f"  {key:<10}: {value}")

    os.makedirs(config.TABLES_DIR, exist_ok=True)
    features_path = os.path.join(
        config.TABLES_DIR, f"ensemble_features_{dataset}.csv"
    )
    metrics_path = os.path.join(
        config.TABLES_DIR, f"ensemble_results_{dataset}.csv"
    )

    features_df.to_csv(features_path, index=False)
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    ensemble.save(config.get_preferred_ensemble_path(dataset))

    print(f"\nSaved features to {features_path}")
    print(f"Saved metrics  to {metrics_path}")

    return features_df, metrics


if __name__ == "__main__":
    train_ensemble_on_validation()
