"""
Müşteri Kaybı (Churn) Tahmin Modeli - Eğitim Scripti
=====================================================
Bu script şunları yapar:
1. Veriyi yükler
2. Temizler ve sayısala çevirir (ön işleme)
3. Birkaç grafik üretir (görselleştirme)
4. Logistic Regression + Random Forest modeli eğitir
5. Başarıyı ölçer (accuracy, precision, recall, ROC-AUC)
6. Eğitilmiş modeli 'churn_model.joblib' olarak kaydeder (Streamlit kullansın diye)

Kullanım:
    python train_model.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.inspection import permutation_importance
# Not: 'shap' kütüphanesini en üstte import ETMİYORUZ. Kurulu değilse veya
# Windows'ta sorun çıkarırsa tüm script çökmesin diye, ileride bir try/except
# bloğunun İÇİNDE import edeceğiz.

# ---------------------------------------------------------------------------
# 1. VERİYİ YÜKLE
# ---------------------------------------------------------------------------
# Kaggle Telco Customer Churn veri setini indirip aynı klasöre koy:
# https://www.kaggle.com/datasets/blastchar/telco-customer-churn
DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

print("1) Veri yükleniyor...")
df = pd.read_csv(DATA_PATH)
print(f"   Satır sayısı: {df.shape[0]}, Sütun sayısı: {df.shape[1]}")

# ---------------------------------------------------------------------------
# 2. VERİ TEMİZLEME / ÖN İŞLEME
# ---------------------------------------------------------------------------
print("2) Veri temizleniyor...")

# 'customerID' tahmin için işe yaramaz (her satırda farklı), atıyoruz.
df = df.drop(columns=["customerID"])

# 'TotalCharges' sütunu CSV'de metin olarak gelir ve bazı boş ('  ') değerler içerir.
# Sayısala çeviriyoruz; çevrilemeyenler NaN olur, sonra dolduruyoruz.
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# Hedef değişken: Churn (Yes/No) -> 1/0
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Özellikler (X) ve hedef (y)
X = df.drop(columns=["Churn"])
y = df["Churn"]

# Sayısal ve kategorik sütunları ayır.
# Önce tüm sayısal sütunları ("number" hepsini kapsar) seçiyoruz; geri kalan
# her şeyi kategorik kabul ediyoruz. Bu yaklaşım pandas 2 ve 3'te de aynı
# çalışır (object/str dtype karışıklığına takılmaz).
numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = [c for c in X.columns if c not in numeric_features]

print(f"   Sayısal sütunlar: {numeric_features}")
print(f"   Kategorik sütunlar: {len(categorical_features)} adet")

# ---------------------------------------------------------------------------
# 3. GÖRSELLEŞTİRME (veriyi anlamak için)
# ---------------------------------------------------------------------------
print("3) Grafikler oluşturuluyor (grafikler/ klasörüne)...")
import os
os.makedirs("grafikler", exist_ok=True)

# 3a. Churn dağılımı: müşterilerin yüzde kaçı ayrılmış?
plt.figure(figsize=(5, 4))
sns.countplot(x="Churn", data=df)
plt.title("Müşteri Kaybı Dağılımı (0=Kaldı, 1=Ayrıldı)")
plt.tight_layout()
plt.savefig("grafikler/churn_dagilimi.png", dpi=120)
plt.close()

# 3b. Sözleşme tipine göre churn (iş açısından en önemli grafiklerden biri)
plt.figure(figsize=(6, 4))
sns.countplot(x="Contract", hue="Churn", data=df)
plt.title("Sözleşme Tipine Göre Müşteri Kaybı")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("grafikler/sozlesme_churn.png", dpi=120)
plt.close()

# 3c. Aylık ücret ile churn ilişkisi
plt.figure(figsize=(6, 4))
sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
plt.title("Aylık Ücret ve Müşteri Kaybı")
plt.tight_layout()
plt.savefig("grafikler/aylik_ucret_churn.png", dpi=120)
plt.close()

# ---------------------------------------------------------------------------
# 4. EĞİTİM / TEST AYIRMA
# ---------------------------------------------------------------------------
# Veriyi %80 eğitim, %20 test olarak ayırıyoruz.
# stratify=y: ayrılan/kalan oranını her iki sette de aynı tutar (dengesiz veri için önemli).
print("4) Eğitim/test ayrılıyor...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------------------------
# 5. ÖN İŞLEME + MODEL PIPELINE
# ---------------------------------------------------------------------------
# ColumnTransformer: sayısalları ölçekle (StandardScaler), kategorikleri OneHot yap.
# Pipeline kullanmak "veri sızıntısını" (data leakage) önler: ölçekleme sadece
# eğitim verisinden öğrenilir, teste uygulanır.
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# İki model deniyoruz ve karşılaştırıyoruz.
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced"
    ),
}

best_model = None
best_auc = 0
best_name = ""
best_proba = None  # En iyi modelin test seti olasılıkları (threshold/ROI için saklanır)

print("5) Modeller eğitiliyor ve değerlendiriliyor...\n")
for name, clf in models.items():
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])

    # 5-katlı çapraz doğrulama (cross-validation): veriyi 5 parçaya böler, her
    # turda 4'üyle eğitip 1'iyle test eder. Sonucun TEK bir test setine bağlı
    # şans olmadığını, modelin tutarlı olduğunu gösterir. Sadece eğitim verisinde
    # yapıyoruz ki test seti tamamen "görülmemiş" kalsın.
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc")

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"--- {name} ---")
    print(f"   Accuracy : {acc:.3f}")
    print(f"   Precision: {prec:.3f}")
    print(f"   Recall   : {rec:.3f}  (ayrılacak müşterilerin ne kadarını yakaladık)")
    print(f"   F1-score : {f1:.3f}")
    print(f"   ROC-AUC  : {auc:.3f}")
    print(f"   5-katlı CV ROC-AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})\n")

    if auc > best_auc:
        best_auc = auc
        best_model = pipe
        best_name = name
        best_proba = y_proba

# En iyi modelin karışıklık matrisi (confusion matrix)
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Kaldı", "Ayrıldı"], yticklabels=["Kaldı", "Ayrıldı"])
plt.title(f"Karışıklık Matrisi - {best_name}")
plt.ylabel("Gerçek")
plt.xlabel("Tahmin")
plt.tight_layout()
plt.savefig("grafikler/karisiklik_matrisi.png", dpi=120)
plt.close()

print(f">> En iyi model: {best_name} (ROC-AUC = {best_auc:.3f})")
print("\nDetaylı rapor:")
print(classification_report(y_test, y_pred_best, target_names=["Kaldı", "Ayrıldı"]))

# ---------------------------------------------------------------------------
# 6. AÇIKLANABİLİRLİK - 1: PERMUTATION IMPORTANCE (global özellik önemi)
# ---------------------------------------------------------------------------
# "Model neye göre karar veriyor?" sorusuna cevap. Permutation importance,
# bir özelliğin değerlerini rastgele karıştırınca modelin başarısı ne kadar
# DÜŞÜYOR ona bakar: çok düşüyorsa o özellik önemlidir.
# best_model bir Pipeline olduğu için ham X_test ile çalışır (ön işlemeyi
# içeride yapar), yani orijinal sütun adlarını kullanabiliriz - kolaylık.
print("\n6) Permutation importance hesaplanıyor (özellik önemi)...")
perm = permutation_importance(
    best_model, X_test, y_test, n_repeats=10, random_state=42, scoring="roc_auc"
)
importance_df = pd.DataFrame(
    {"feature": X_test.columns, "importance": perm.importances_mean}
).sort_values("importance", ascending=False)

# En etkili 15 özelliği yatay bar grafiği olarak çiz
top15 = importance_df.head(15).iloc[::-1]  # iloc[::-1]: en önemli en üstte görünsün
plt.figure(figsize=(7, 6))
plt.barh(top15["feature"], top15["importance"], color="#4C72B0")
plt.title("Özellik Önemi (Permutation Importance)")
plt.xlabel("Önem (ROC-AUC'ye katkı)")
plt.tight_layout()
plt.savefig("grafikler/feature_importance.png", dpi=120)
plt.close()

# Streamlit'in göstermesi için ilk 15'i kaydet
joblib.dump(importance_df.head(15), "feature_importance.joblib")
print("   En etkili 5 özellik:")
print(importance_df.head(5).to_string(index=False))

# ---------------------------------------------------------------------------
# 7. AÇIKLANABİLİRLİK - 2: SHAP (her tahmin için katkıları gösterir)
# ---------------------------------------------------------------------------
# SHAP, permutation importance'tan farklı olarak her bir özelliğin tahmini
# hangi yönde (artı/eksi) ittiğini gösterir. Daha modern ve gösterişli.
# Pipeline'ın ham hali yerine: önce ön işleme uygula, sonra final modele ver.
# Tüm blok try/except içinde: SHAP kurulu değilse veya hata verirse proje
# çökmesin, sadece bu grafik üretilmesin.
print("\n7) SHAP analizi yapılıyor...")
try:
    import shap

    preprocessor_fitted = best_model.named_steps["preprocessor"]
    classifier_fitted = best_model.named_steps["classifier"]
    feature_names = preprocessor_fitted.get_feature_names_out()

    # Hız için test setinden en fazla 200 satır örnekle (SHAP yavaş olabilir)
    sample = X_test.sample(n=min(200, len(X_test)), random_state=42)
    sample_transformed = preprocessor_fitted.transform(sample)
    # OneHotEncoder seyrek (sparse) matris döndürebilir; SHAP yoğun dizi bekler.
    if hasattr(sample_transformed, "toarray"):
        sample_transformed = sample_transformed.toarray()

    # Ağaç tabanlı model (Random Forest) ise TreeExplainer, değilse LinearExplainer
    if best_name == "Random Forest":
        explainer = shap.TreeExplainer(classifier_fitted)
        shap_values = explainer.shap_values(sample_transformed)
        # Yeni shap sürümlerinde ikili sınıflandırmada 3 boyutlu dizi gelebilir;
        # pozitif sınıfın (ayrıldı=1) katkılarını alıyoruz.
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]
    else:
        explainer = shap.LinearExplainer(classifier_fitted, sample_transformed)
        shap_values = explainer.shap_values(sample_transformed)

    plt.figure()
    shap.summary_plot(
        shap_values, sample_transformed, feature_names=feature_names, show=False
    )
    plt.tight_layout()
    plt.savefig("grafikler/shap_summary.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("   SHAP özet grafiği 'grafikler/shap_summary.png' olarak kaydedildi.")
except Exception as e:
    print(f"   ! SHAP atlandı (proje yine de çalışır): {e}")

# ---------------------------------------------------------------------------
# 8. İŞ ETKİSİ (ROI) HESABI - modelin para değeri
# ---------------------------------------------------------------------------
# Confusion matrix'ten:
#   TN = doğru "kaldı"   FP = yanlış "ayrıldı"
#   FN = kaçırılan ayrılan  TP = doğru yakalanan ayrılan
tn, fp, fn, tp = cm.ravel()

# Varsayımlar (gerçek bir şirkette bu rakamlar finans ekibinden gelir)
MUSTERI_DEGERI = 1000      # Kaybedilen bir müşterinin yıllık geliri (TL)
KAMPANYA_MALIYETI = 100    # Riskli müşteriye yapılan elde tutma kampanyası (TL)
KAMPANYA_BASARI_ORANI = 0.30  # Kampanyanın müşteriyi tutma oranı

# Model "ayrılacak" dediği herkese kampanya yapılır: TP + FP kişi
hedeflenen_musteri = tp + fp
# Sadece gerçekten ayrılacak olanların (TP) bir kısmı kampanyayla kurtarılır
kurtarilan_gelir = tp * KAMPANYA_BASARI_ORANI * MUSTERI_DEGERI
toplam_kampanya_maliyeti = hedeflenen_musteri * KAMPANYA_MALIYETI
net_kazanc = kurtarilan_gelir - toplam_kampanya_maliyeti

print("\n8) İş Etkisi (ROI) - test seti üzerinden:")
print(f"   Hedeflenen müşteri (TP+FP) : {hedeflenen_musteri}")
print(f"   Kurtarılan gelir           : {kurtarilan_gelir:,.0f} TL")
print(f"   Toplam kampanya maliyeti   : {toplam_kampanya_maliyeti:,.0f} TL")
print(f"   >> NET KAZANÇ              : {net_kazanc:,.0f} TL")

# Streamlit'in canlı ROI hesabı yapabilmesi için metrikleri kaydet
joblib.dump(
    {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
     "recall": float(recall_score(y_test, y_pred_best)),
     "precision": float(precision_score(y_test, y_pred_best)),
     "model_name": best_name, "roc_auc": float(best_auc)},
    "business_metrics.joblib"
)

# Test seti olasılıkları + gerçek etiketler: Streamlit'te farklı karar eşiklerinde
# (threshold) ROI'yi yeniden hesaplayabilmek için. Böylece "hangi eşik en çok kâr
# getirir?" sorusunu canlı yanıtlayabiliriz.
joblib.dump(
    {"y_test": y_test.to_numpy(), "y_proba": best_proba},
    "test_predictions.joblib"
)

# ---------------------------------------------------------------------------
# 9. MODELİ KAYDET (Streamlit arayüzü kullanacak)
# ---------------------------------------------------------------------------
joblib.dump(best_model, "churn_model.joblib")
# Streamlit formunda seçenekleri göstermek için sütun bilgisini de kaydediyoruz.
joblib.dump(
    {"numeric": numeric_features, "categorical": categorical_features,
     "categories": {c: sorted(df[c].dropna().unique().tolist()) for c in categorical_features}},
    "model_columns.joblib"
)
print("\n9) Model 'churn_model.joblib' olarak kaydedildi. Artık 'streamlit run app.py' çalıştırabilirsin.")
