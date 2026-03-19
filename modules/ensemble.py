# modules/ensemble.py
# Stacking ensemble meta-classifier.
# Combines BERT and RAG outputs into final verdict.
#
# Why stacking?
# Simple averaging: final = 0.5*BERT + 0.5*RAG
# This assumes both are equally reliable always — wrong.
# Stacking trains a meta-classifier to learn optimal weights
# from validation data — data-driven, not hand-tuned.

import os
import sys
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class StackingEnsemble:
    """
    Meta-classifier that combines BERT and RAG outputs.
    
    Input features:
      1. bert_prob          — BERT probability of REAL (0-1)
      2. rag_confidence     — RAG verdict confidence (0-1)
      3. max_similarity     — best evidence similarity score
      4. source_weight_avg  — average source credibility
      5. hallucination_flag — 1 if hallucinated, 0 if not
      6. sub_claim_count    — how many sub-claims were found
    
    Why Logistic Regression as meta-classifier?
    Simple, interpretable, works well with small feature sets.
    SHAP values are meaningful on LR — directly shows which
    feature drove the final decision.
    XGBoost is included as alternative — compared in ablation.
    """

    def __init__(self):
        # Why StandardScaler?
        # Features have different scales:
        # bert_prob is 0-1, sub_claim_count could be 1-5
        # Scaling ensures no feature dominates due to magnitude
        self.scaler     = StandardScaler()
        self.classifier = LogisticRegression(
            random_state = config.RANDOM_SEED,
            max_iter     = 1000,    # ensure convergence
            C            = 1.0      # regularization strength
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
        
        Args:
            bert_prob  : float — BERT probability of REAL class
            rag_result : dict  — output from reasoner.aggregate_sub_claim_verdicts()
            
        Returns:
            numpy array of shape (1, 6)
        """

        # Convert hallucination flag to numeric
        hallucination = 1.0 if rag_result.get("hallucination_flag") else 0.0

        # Convert RAG verdict to confidence direction
        # If verdict is FALSE, confidence points toward FAKE
        # If verdict is TRUE, confidence points toward REAL
        rag_conf = rag_result.get("final_confidence", 0.5)
        if rag_result.get("final_verdict") == "FALSE":
            rag_conf = 1.0 - rag_conf  # flip toward FAKE

        features = np.array([[
            bert_prob,
            rag_conf,
            rag_result.get("max_similarity", 0.0),
            rag_result.get("source_weight_avg", 0.5),
            hallucination,
            float(rag_result.get("sub_claim_count", 1))
        ]])

        return features

    def train(self, feature_matrix, labels):
        """
        Trains the meta-classifier on validation set features.
        
        Why train on validation set not training set?
        BERT and RAG were trained/tuned on training set.
        Using training set for ensemble = data leakage.
        Validation set is unseen by both models — clean signal.
        
        Args:
            feature_matrix : np.array of shape (n_samples, 6)
            labels         : list of 0/1 labels
        """

        print("Training stacking ensemble...")
        print(f"  Samples  : {len(labels)}")
        print(f"  Features : {self.feature_names}")

        # Scale features
        X_scaled = self.scaler.fit_transform(feature_matrix)

        # Train meta-classifier
        self.classifier.fit(X_scaled, labels)
        self.is_trained = True

        # Training accuracy
        train_preds = self.classifier.predict(X_scaled)
        train_acc   = accuracy_score(labels, train_preds)
        train_f1    = f1_score(labels, train_preds, average="weighted")

        print(f"  Train Acc : {train_acc:.4f}")
        print(f"  Train F1  : {train_f1:.4f}")

        # Feature importance via coefficients
        # Positive coefficient = feature pushes toward REAL
        # Negative coefficient = feature pushes toward FAKE
        print("\n  Feature Coefficients:")
        for name, coef in zip(
            self.feature_names, self.classifier.coef_[0]
        ):
            direction = "→ REAL" if coef > 0 else "→ FAKE"
            print(f"    {name:<22} : {coef:+.4f}  {direction}")

    def predict(self, feature_matrix):
        """
        Predicts final verdict for given features.
        
        Returns:
            predictions : np.array of 0/1 labels
            probabilities: np.array of REAL probabilities
        """

        if not self.is_trained:
            raise ValueError("Ensemble not trained yet. Call train() first.")

        X_scaled      = self.scaler.transform(feature_matrix)
        predictions   = self.classifier.predict(X_scaled)
        probabilities = self.classifier.predict_proba(X_scaled)[:, 1]

        return predictions, probabilities

    def predict_single(self, bert_prob, rag_result):
        """
        Predicts verdict for a single claim.
        Convenience wrapper for pipeline use.
        
        Returns:
            dict with final_verdict, confidence
        """

        features      = self.build_feature_vector(bert_prob, rag_result)
        preds, probs  = self.predict(features)

        verdict    = "REAL" if preds[0] == 1 else "FAKE"
        confidence = float(probs[0]) if preds[0] == 1 else 1.0 - float(probs[0])

        return {
            "final_verdict"    : verdict,
            "confidence"       : round(confidence, 4),
            "bert_prob"        : round(bert_prob, 4),
            "rag_verdict"      : rag_result.get("final_verdict"),
            "rag_confidence"   : rag_result.get("final_confidence"),
            "hallucination"    : rag_result.get("hallucination_flag")
        }

    def save(self, path=None):
        """Saves trained ensemble to disk."""
        if path is None:
            path = os.path.join(config.MODELS_DIR, "ensemble.pkl")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "classifier" : self.classifier,
                "scaler"     : self.scaler,
                "features"   : self.feature_names
            }, f)
        print(f"Ensemble saved to {path}")

    def load(self, path=None):
        """Loads trained ensemble from disk."""
        if path is None:
            path = os.path.join(config.MODELS_DIR, "ensemble.pkl")

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.classifier  = data["classifier"]
        self.scaler      = data["scaler"]
        self.feature_names = data["features"]
        self.is_trained  = True
        print(f"Ensemble loaded from {path}")


if __name__ == "__main__":

    # Simulate what the full pipeline produces
    # In production these come from BERT + RAG on real data
    # Here we test the ensemble logic with synthetic data

    print("Testing ensemble with synthetic data...")
    print("="*55)

    np.random.seed(config.RANDOM_SEED)
    n_samples = 200

    # Simulate feature matrix
    # In real pipeline: built from BERT + RAG on val set
    synthetic_features = np.column_stack([
        np.random.uniform(0.3, 0.9, n_samples),  # bert_prob
        np.random.uniform(0.4, 0.95, n_samples), # rag_confidence
        np.random.uniform(0.3, 0.9, n_samples),  # max_similarity
        np.random.uniform(0.6, 0.95, n_samples), # source_weight_avg
        np.random.randint(0, 2, n_samples),       # hallucination_flag
        np.random.randint(1, 4, n_samples)        # sub_claim_count
    ]).astype(float)

    # Synthetic labels — in real pipeline: from val set binary_label
    synthetic_labels = np.random.randint(0, 2, n_samples)

    # Train ensemble
    ensemble = StackingEnsemble()
    ensemble.train(synthetic_features, synthetic_labels)

    # Test prediction on single claim
    print("\nSingle claim prediction test:")
    test_rag_result = {
        "final_verdict"      : "FALSE",
        "final_confidence"   : 0.87,
        "hallucination_flag" : False,
        "sub_claim_count"    : 2,
        "max_similarity"     : 0.82,
        "source_weight_avg"  : 0.90
    }

    result = ensemble.predict_single(
        bert_prob  = 0.34,   # BERT thinks 34% chance REAL = leans FAKE
        rag_result = test_rag_result
    )

    print(f"  BERT probability : {result['bert_prob']}")
    print(f"  RAG verdict      : {result['rag_verdict']}")
    print(f"  RAG confidence   : {result['rag_confidence']}")
    print(f"  Final verdict    : {result['final_verdict']}")
    print(f"  Confidence       : {result['confidence']}")

    # Save
    ensemble.save()