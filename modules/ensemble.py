# modules/ensemble.py
# Stacking ensemble meta-classifier.
# Combines BERT and RAG outputs into final verdict.
#
# Why stacking?
# Simple averaging: final = 0.5*BERT + 0.5*RAG
# This assumes both are equally reliable always â€” wrong.
# Stacking trains a meta-classifier to learn optimal weights
# from validation data â€” data-driven, not hand-tuned.

import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from transformers import RobertaTokenizer, RobertaForSequenceClassification

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from modules.preprocessor import preprocess_text
from modules.retriever import Retriever
from modules.decomposer import ClaimDecomposer
from modules.reasoner import EvidenceReasoner


class StackingEnsemble:
    """
    Meta-classifier that combines BERT and RAG outputs.

    Input features:
      1. bert_prob          â€” BERT probability of REAL (0-1)
      2. rag_confidence     â€” RAG verdict confidence (0-1)
      3. max_similarity     â€” best evidence similarity score
      4. source_weight_avg  â€” average source credibility
      5. hallucination_flag â€” 1 if hallucinated, 0 if not
      6. sub_claim_count    â€” how many sub-claims were found

    Why Logistic Regression as meta-classifier?
    Simple, interpretable, works well with small feature sets.
    SHAP values are meaningful on LR â€” directly shows which
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
            direction = "â†’ REAL" if coef > 0 else "â†’ FAKE"
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
            path = os.path.join(config.MODELS_DIR, "ensemble.pkl")

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
            path = os.path.join(config.MODELS_DIR, "ensemble.pkl")

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.classifier = data["classifier"]
        self.scaler = data["scaler"]
        self.feature_names = data["features"]
        self.is_trained = True
        print(f"Ensemble loaded from {path}")


def _resolve_liar_validation_assets():
    """
    Returns the LIAR validation file and checkpoint paths used for
    ensemble training.
    """

    data_path = os.path.join(config.DATA_PROCESSED, "val.csv")
    model_path = os.path.join(config.MODELS_DIR, "roberta_liar")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing LIAR validation data: {data_path}")
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Missing LIAR RoBERTa checkpoint: {model_path}")

    return data_path, model_path


def _get_bert_probability(text, model, tokenizer, device):
    """Returns RoBERTa probability of REAL for one claim."""

    clean = preprocess_text(text)
    encoding = tokenizer(
        clean,
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


def _run_rag_branch(claim, decomposer, retriever, reasoner):
    """
    Runs the claim decomposition + retrieval + reasoning branch for one
    LIAR claim and returns an aggregated RAG result.
    """

    clean_claim = preprocess_text(claim)
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


def train_ensemble_on_validation(sample_size=None):
    """
    Generates real validation features from the LIAR validation set,
    trains the ensemble, and saves both the trained model and feature
    table for inspection.

    Why LIAR only?
    The current RAG pipeline is built for short, claim-style inputs.
    Using the same decomposition and reasoning flow on full ISOT
    articles would produce weak and misleading ensemble features.
    """

    data_path, model_path = _resolve_liar_validation_assets()

    print("=" * 65)
    print("  TRAINING STACKING ENSEMBLE ON LIAR VALIDATION FEATURES")
    print("=" * 65)
    print(f"Validation data : {data_path}")
    print(f"RoBERTa path    : {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device          : {device}")

    val_df = pd.read_csv(data_path)
    val_df = val_df.dropna(subset=["binary_label", "statement"]).reset_index(drop=True)

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
        claim = str(row["statement"])
        label = int(row["binary_label"])

        try:
            rag_result = _run_rag_branch(
                claim, decomposer, retriever, reasoner
            )
            bert_prob = _get_bert_probability(
                claim, bert_model, tokenizer, device
            )
            feature_vector = ensemble.build_feature_vector(
                bert_prob, rag_result
            )[0]

            feature_rows.append({
                "claim"              : claim,
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

            if (idx + 1) % 25 == 0:
                print(f"  Processed {idx + 1}/{len(val_df)}")

        except Exception as exc:
            print(f"  Warning: failed on validation row {idx} - {exc}")

    if not feature_rows:
        raise RuntimeError("No validation features were generated.")

    features_df = pd.DataFrame(feature_rows)
    X = features_df[ensemble.feature_names].to_numpy(dtype=float)
    y = features_df["label"].to_numpy(dtype=int)

    ensemble.train(X, y)
    preds, _ = ensemble.predict(X)

    metrics = {
        "dataset"  : "liar",
        "samples"  : int(len(features_df)),
        "accuracy" : round(accuracy_score(y, preds), 4),
        "f1"       : round(f1_score(y, preds, average="weighted"), 4)
    }

    print("\nValidation metrics after ensemble training:")
    for key, value in metrics.items():
        print(f"  {key:<10}: {value}")

    os.makedirs(config.TABLES_DIR, exist_ok=True)
    features_path = os.path.join(config.TABLES_DIR, "ensemble_features_liar.csv")
    metrics_path = os.path.join(config.TABLES_DIR, "ensemble_results_liar.csv")

    features_df.to_csv(features_path, index=False)
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    ensemble.save()

    print(f"\nSaved features to {features_path}")
    print(f"Saved metrics  to {metrics_path}")

    return features_df, metrics


if __name__ == "__main__":
    train_ensemble_on_validation()
