"""
Müşteri Kaybı (Churn) Tahmin Arayüzü - Streamlit
=================================================
Önce 'python train_model.py' çalıştırıp modeli ürettiğinden emin ol.
Sonra:  streamlit run app.py

Arayüz 4 sekmeden oluşur:
  1) 🔮 Tahmin        - tek bir müşteri için churn olasılığı
  2) 📊 Model Analizi - hangi özellikler önemli (feature importance + SHAP)
  3) 💰 İş Etkisi     - modelin para değeri + karar eşiği (threshold) optimizasyonu
  4) 📁 Toplu Tahmin  - CSV yükle, tüm müşterileri skorla, sonucu indir
"""

import os
import numpy as np
import streamlit as st
import pandas as pd
import joblib

# Sayfa ayarları
st.set_page_config(page_title="Müşteri Kaybı Tahmini", page_icon="📉", layout="centered")


# Eğitilmiş modeli ve yardımcı dosyaları yükle (cache ile bir kez yüklenir).
# Bazı dosyalar henüz üretilmemiş olabilir (örn. SHAP atlanmışsa) -> güvenli yükleme.
@st.cache_resource
def load_artifacts():
    model = joblib.load("churn_model.joblib")
    cols = joblib.load("model_columns.joblib")
    importance = joblib.load("feature_importance.joblib") if os.path.exists("feature_importance.joblib") else None
    business = joblib.load("business_metrics.joblib") if os.path.exists("business_metrics.joblib") else None
    preds = joblib.load("test_predictions.joblib") if os.path.exists("test_predictions.joblib") else None
    return model, cols, importance, business, preds


model, cols, importance_df, business, preds = load_artifacts()

st.title("📉 Müşteri Kaybı (Churn) Tahmini")

tab_tahmin, tab_analiz, tab_roi, tab_batch = st.tabs(
    ["🔮 Tahmin", "📊 Model Analizi", "💰 İş Etkisi", "📁 Toplu Tahmin"]
)

# ===========================================================================
# SEKME 1: TAHMİN
# ===========================================================================
with tab_tahmin:
    st.write(
        "Bir telekom müşterisinin bilgilerini gir; model, bu müşterinin "
        "**ayrılma (churn) olasılığını** tahmin etsin."
    )
    st.subheader("Müşteri Bilgileri")

    user_input = {}
    col1, col2 = st.columns(2)

    # Kategorik alanlar: açılır menü (selectbox)
    cat_features = cols["categorical"]
    for i, feature in enumerate(cat_features):
        options = cols["categories"][feature]
        target_col = col1 if i % 2 == 0 else col2
        user_input[feature] = target_col.selectbox(feature, options)

    # Sayısal alanlar: kayar çubuk / sayı girişi
    st.subheader("Sayısal Değerler")
    # SeniorCitizen veri setinde 0/1 (sayısal) ama anlamca evet/hayır -> menü göster,
    # modele 0/1 olarak gönder.
    if "SeniorCitizen" in cols["numeric"]:
        senior = st.selectbox("Yaşlı vatandaş mı? (SeniorCitizen)", ["Hayır", "Evet"])
        user_input["SeniorCitizen"] = 1 if senior == "Evet" else 0
    user_input["tenure"] = st.slider("Müşteri süresi (ay) - tenure", 0, 72, 12)
    user_input["MonthlyCharges"] = st.slider("Aylık ücret (MonthlyCharges)", 18.0, 120.0, 70.0)
    user_input["TotalCharges"] = st.number_input("Toplam ücret (TotalCharges)", 0.0, 9000.0, 800.0)

    if st.button("Tahmin Et", type="primary", use_container_width=True):
        # Girdileri modelin beklediği tek satırlık DataFrame'e çevir.
        # Güvenlik ağı: modelin beklediği TÜM sütunlar var mı? Eksikse uyar.
        input_df = pd.DataFrame([user_input])
        beklenen = cols["numeric"] + cols["categorical"]
        eksik = [c for c in beklenen if c not in input_df.columns]
        if eksik:
            st.error(f"Form eksik sütun(lar) içeriyor: {eksik}. Lütfen train_model.py'yi tekrar çalıştırın.")
            st.stop()

        proba = model.predict_proba(input_df)[0][1]  # ayrılma olasılığı
        prediction = model.predict(input_df)[0]

        st.subheader("Sonuç")
        st.metric("Ayrılma Olasılığı", f"%{proba * 100:.1f}")
        st.progress(float(proba))

        if prediction == 1:
            st.error("⚠️ Bu müşterinin **ayrılma riski yüksek**. Elde tutma kampanyası önerilir.")
        else:
            st.success("✅ Bu müşterinin **kalma olasılığı yüksek**.")

# ===========================================================================
# SEKME 2: MODEL ANALİZİ (açıklanabilirlik)
# ===========================================================================
with tab_analiz:
    st.subheader("Model Neye Göre Karar Veriyor?")
    st.write(
        "Aşağıda hangi müşteri özelliklerinin ayrılma (churn) tahmininde **en etkili** "
        "olduğunu görüyorsun. Bu, modelin bir 'kara kutu' olmadığını gösterir."
    )

    # Permutation importance tablosu / bar grafiği
    if importance_df is not None:
        st.markdown("**En Etkili Özellikler (Permutation Importance)**")
        chart_df = importance_df.set_index("feature")["importance"]
        st.bar_chart(chart_df)
        with st.expander("Tabloyu göster"):
            st.dataframe(importance_df.reset_index(drop=True))
    else:
        st.info("Önce `python train_model.py` çalıştır; özellik önemi dosyası oluşacak.")

    # train_model.py'nin ürettiği grafikleri göster (varsa)
    st.markdown("**Üretilen Grafikler**")
    if os.path.exists("grafikler/feature_importance.png"):
        st.image("grafikler/feature_importance.png", caption="Özellik Önemi (Permutation Importance)")
    if os.path.exists("grafikler/shap_summary.png"):
        st.image("grafikler/shap_summary.png", caption="SHAP Özeti - her özelliğin tahmine katkısı")
    else:
        st.caption("SHAP grafiği bulunamadı (SHAP atlanmış olabilir; proje yine de çalışır).")

    st.info(
        "💡 Tipik bulgu: **düşük müşteri süresi (tenure)**, **aya-dayalı (month-to-month) "
        "sözleşme** ve **yüksek aylık ücret** ayrılma riskini en çok artıran faktörlerdir."
    )

# ===========================================================================
# SEKME 3: İŞ ETKİSİ (ROI) + KARAR EŞİĞİ (THRESHOLD) OPTİMİZASYONU
# ===========================================================================
with tab_roi:
    st.subheader("Bu Model Şirkete Ne Kadar Kazandırır?")
    st.write(
        "Model her müşteriye bir risk skoru verir. Bir **karar eşiği** seçeriz: skoru "
        "bu eşiğin üstünde olanlara kampanya yapılır. Aşağıdaki değerleri değiştirerek "
        "**net kazancı** ve **en kârlı eşiği** canlı gör."
    )

    if preds is None or business is None:
        st.info("Önce `python train_model.py` çalıştır; gerekli dosyalar oluşacak.")
    else:
        y_test = np.asarray(preds["y_test"])
        y_proba = np.asarray(preds["y_proba"])

        # İş parametreleri (gerçek bir şirkette finans ekibinden gelir)
        c1, c2, c3 = st.columns(3)
        musteri_degeri = c1.number_input("Müşteri yıllık geliri (TL)", 100, 100000, 1000, step=100)
        kampanya_maliyeti = c2.number_input("Kampanya maliyeti / kişi (TL)", 0, 5000, 100, step=10)
        basari_orani = c3.slider("Kampanya başarı oranı", 0.0, 1.0, 0.30, step=0.05)

        def roi_at(t):
            """Verilen eşik t için (hedeflenen, kurtarılan_gelir, net_kazanç) döndürür."""
            pred = (y_proba >= t).astype(int)
            tp = int(np.sum((pred == 1) & (y_test == 1)))  # doğru yakalanan ayrılan
            fp = int(np.sum((pred == 1) & (y_test == 0)))  # boşuna kampanya yapılan
            hedef = tp + fp
            kurtarilan = tp * basari_orani * musteri_degeri
            net = kurtarilan - hedef * kampanya_maliyeti
            return hedef, kurtarilan, net

        threshold = st.slider(
            "Karar eşiği (bu olasılığın üstündekiler 'ayrılacak' sayılır)",
            0.05, 0.95, 0.50, step=0.05
        )
        hedef, kurtarilan, net = roi_at(threshold)

        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Hedeflenen müşteri", f"{hedef}")
        m2.metric("Kurtarılan gelir", f"{kurtarilan:,.0f} TL")
        m3.metric("Net Kazanç", f"{net:,.0f} TL")

        # Eşik taraması: hangi eşik en çok kâr getiriyor?
        thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
        profits = [roi_at(t)[2] for t in thresholds]
        chart_df = pd.DataFrame({"Karar eşiği": thresholds, "Net kazanç (TL)": profits}).set_index("Karar eşiği")
        st.markdown("**Net kazanç, karar eşiğine göre nasıl değişiyor?**")
        st.line_chart(chart_df)

        best_i = int(np.argmax(profits))
        st.success(
            f"💡 En kârlı karar eşiği: **{thresholds[best_i]:.2f}** "
            f"→ net kazanç **{profits[best_i]:,.0f} TL**. "
            f"(Varsayılan 0.50 eşik her zaman en kârlı değildir — bu, ML'i doğrudan kâra bağlar.)"
        )
        st.caption(
            f"Model: {business['model_name']} | ROC-AUC: {business['roc_auc']:.3f} "
            f"(test seti üzerinden)"
        )

# ===========================================================================
# SEKME 4: TOPLU TAHMİN (CSV)
# ===========================================================================
with tab_batch:
    st.subheader("Birden Çok Müşteriyi Aynı Anda Skorla")
    st.write(
        "Müşteri bilgilerini içeren bir **CSV** yükle; model her satır için ayrılma "
        "olasılığı hesaplasın. Sonucu indirebilirsin."
    )
    st.caption("İpucu: Kaggle'daki orijinal Telco CSV'sini de yükleyebilirsin — `customerID` ve `Churn` sütunları otomatik atılır.")

    uploaded = st.file_uploader("CSV dosyası seç", type=["csv"])
    if uploaded is not None:
        try:
            data = pd.read_csv(uploaded)

            # Tahmin için gereksiz sütunları at (varsa)
            for col in ["customerID", "Churn"]:
                if col in data.columns:
                    data = data.drop(columns=[col])

            # TotalCharges metin/boş gelebilir -> sayısala çevir, boşları doldur
            if "TotalCharges" in data.columns:
                data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
                data["TotalCharges"] = data["TotalCharges"].fillna(data["TotalCharges"].median())

            beklenen = cols["numeric"] + cols["categorical"]
            eksik = [c for c in beklenen if c not in data.columns]
            if eksik:
                st.error(f"CSV'de eksik sütunlar var: {eksik}")
            else:
                proba = model.predict_proba(data[beklenen])[:, 1]
                sonuc = data.copy()
                sonuc["churn_olasiligi_%"] = (proba * 100).round(1)
                sonuc["tahmin"] = np.where(proba >= 0.5, "Ayrılır", "Kalır")

                st.success(f"{len(sonuc)} müşteri skorlandı.")
                st.markdown("**En riskli 10 müşteri:**")
                st.dataframe(
                    sonuc.sort_values("churn_olasiligi_%", ascending=False).head(10),
                    use_container_width=True,
                )

                csv_out = sonuc.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Tüm sonuçları indir (CSV)",
                    csv_out,
                    file_name="churn_tahminleri.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
        except Exception as e:
            st.error(f"Dosya işlenirken hata oluştu: {e}")

st.divider()
st.caption("Model: scikit-learn | Veri: Telco Customer Churn (Kaggle)")
