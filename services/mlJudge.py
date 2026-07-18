"""Regression-based judging logic for Debate Chamber."""

import os
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


class DebateRegressionJudge:
    FEATURE_COLUMNS = ["word_count", "complexity_score"]
    DATASET_PATH = "historical_debates.csv"

    def __init__(self):
        self.model = None
        self._metrics = {"mse": None, "r2_score": None}

    def extract_NLP_features(self, text):
        """Turn raw debate text into the same numeric features the model was trained on."""
        if not text or not str(text).strip():
            return {"word_count": 0, "complexity_score": 1.0, "sentiment": 0.0}

        text = str(text).strip()
        words = re.findall(r"[A-Za-z']+", text)
        word_count = len(words)

        if word_count == 0:
            return {"word_count": 0, "complexity_score": 1.0, "sentiment": 0.0}

        # These values give the judge a simple view of argument depth and language variety.
        unique_words = len(set(w.lower() for w in words))
        vocabulary_richness = unique_words / word_count
        avg_word_length = sum(len(w) for w in words) / word_count

        sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
        sentence_count = max(len(sentences), 1)
        avg_sentence_length = word_count / sentence_count

        complexity_raw = (
            vocabulary_richness * 4.0
            + (avg_word_length / 8.0) * 3.0
            + min(avg_sentence_length / 6.0, 1.0) * 3.0
        )
        complexity_score = float(np.clip(complexity_raw, 1.0, 10.0))

        # Sentiment is tracked as supporting context, but it is not used by the regression model.
        positive = len(
            re.findall(
                r"\b(good|great|benefit|proven|success|strong|effective|advantage|support|evidence)\b",
                text,
                re.IGNORECASE,
            )
        )
        negative = len(
            re.findall(
                r"\b(bad|fail|danger|risk|weak|flaw|wrong|harm|crisis|problem|destroy)\b",
                text,
                re.IGNORECASE,
            )
        )
        sentiment = (positive - negative) / max(word_count, 1)

        return {
            "word_count": word_count,
            "complexity_score": round(complexity_score, 2),
            "sentiment": round(sentiment, 4),
        }

    def train_model(self, dataset_path=None):
        """Load the CSV data, fit the regression model, and return basic metrics."""
        path = dataset_path or self.DATASET_PATH
        if not os.path.isabs(path):
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(base_dir, path)

        print(f"Loading dataset from {path}...")
        df = pd.read_csv(path)

        required = self.FEATURE_COLUMNS + ["human_persuasiveness_score"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}")

        X = df[self.FEATURE_COLUMNS].values
        y = df["human_persuasiveness_score"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # A small RandomForest model is enough for this demo and handles non-linear patterns well.
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        mse = float(mean_squared_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        self._metrics = {"mse": round(mse, 4), "r2_score": round(r2, 4)}
        print(
            f"Model trained — MSE: {self._metrics['mse']}, R²: {self._metrics['r2_score']}"
        )
        return self._metrics

    def predict_score(self, text):
        """Return a 1–10 score for a debate argument based on extracted features."""
        if self.model is None:
            raise RuntimeError("Model is not trained yet. Call train_model() first.")

        features = self.extract_NLP_features(text)
        feature_vector = np.array(
            [[features["word_count"], features["complexity_score"]]]
        )
        raw_score = float(self.model.predict(feature_vector)[0])
        return round(float(np.clip(raw_score, 1.0, 10.0)), 1)

    def get_metrics(self):
        """Return the most recent training metrics."""
        return self._metrics
