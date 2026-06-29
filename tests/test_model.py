"""
Modelin temel davranışlarını doğrulayan testler.
CSV'ye İHTİYAÇ DUYMAZ; commit'lenmiş model dosyalarını (.joblib) kullanır,
böylece GitHub Actions CI'da (veri seti olmadan) da çalışır.

Çalıştırmak için:  pytest
"""

import os
import joblib
import pandas as pd

# Proje kök dizini (bu dosya tests/ altında)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    model = joblib.load(os.path.join(ROOT, "churn_model.joblib"))
    cols = joblib.load(os.path.join(ROOT, "model_columns.joblib"))
    return model, cols


def _sample_row(cols):
    """Modelin beklediği tüm sütunları içeren tek satırlık örnek girdi üretir."""
    row = {f: cols["categories"][f][0] for f in cols["categorical"]}
    if "SeniorCitizen" in cols["numeric"]:
        row["SeniorCitizen"] = 0
    row["tenure"] = 12
    row["MonthlyCharges"] = 70.0
    row["TotalCharges"] = 800.0
    return pd.DataFrame([row])


def test_artifacts_exist():
    """Eğitilmiş model dosyaları repoda mevcut olmalı."""
    for fname in ["churn_model.joblib", "model_columns.joblib"]:
        assert os.path.exists(os.path.join(ROOT, fname)), f"{fname} bulunamadı"


def test_model_columns_structure():
    """model_columns beklenen anahtarları içermeli."""
    _, cols = _load()
    for key in ["numeric", "categorical", "categories"]:
        assert key in cols, f"'{key}' anahtarı eksik"
    assert len(cols["categorical"]) > 0


def test_prediction_probability_in_range():
    """predict_proba 0 ile 1 arasında bir olasılık döndürmeli."""
    model, cols = _load()
    proba = model.predict_proba(_sample_row(cols))[0][1]
    assert 0.0 <= proba <= 1.0


def test_prediction_is_binary():
    """predict 0 veya 1 döndürmeli (kaldı / ayrıldı)."""
    model, cols = _load()
    pred = model.predict(_sample_row(cols))[0]
    assert int(pred) in (0, 1)
