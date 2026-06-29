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
st.set_page_config(
    page_title="Müşteri Kaybı Tahmini",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# ÖZEL CSS — uygulamaya modern bir "dashboard" görünümü verir
# ---------------------------------------------------------------------------
CSS = """
<style>
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1150px; }
footer { visibility: hidden; }

/* Başlık (hero) bandı */
.hero {
  background: linear-gradient(135deg, #6C5CE7 0%, #4338CA 55%, #2563EB 100%);
  padding: 2.1rem 2.2rem; border-radius: 18px; margin-bottom: 1.4rem;
  box-shadow: 0 12px 34px rgba(99,91,255,0.28);
}
.hero h1 { color: #fff; font-size: 2rem; margin: 0 0 .45rem 0; font-weight: 800; letter-spacing: -.5px; }
.hero p  { color: rgba(255,255,255,0.9); font-size: 1.03rem; margin: 0; }

/* Metric kartları */
[data-testid="stMetric"] {
  background: #161B26; border: 1px solid #262C3A; border-radius: 14px;
  padding: 1rem 1.1rem; box-shadow: 0 2px 10px rgba(0,0,0,0.25);
}
[data-testid="stMetricValue"] { font-size: 1.55rem; font-weight: 700; }

/* Butonlar */
.stButton > button, .stDownloadButton > button {
  border-radius: 10px; font-weight: 600; height: 3rem; border: none;
  transition: transform .06s ease, box-shadow .2s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  transform: translateY(-1px); box-shadow: 0 6px 18px rgba(99,91,255,0.35);
}

/* Sekmeler */
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
  border-radius: 10px 10px 0 0; padding: 9px 18px; font-weight: 600;
}

/* Tahmin sonuç kartı */
.result-card {
  text-align: center; padding: 1.7rem 1.5rem; border-radius: 18px;
  background: #161B26; border: 2px solid #333; margin: .4rem 0 1rem 0;
}
.result-pct   { font-size: 3.4rem; font-weight: 800; line-height: 1; }
.result-label { font-size: 1.15rem; margin-top: .55rem; font-weight: 700; }
.result-sub   { opacity: .7; margin-top: .3rem; font-size: .9rem; }

/* Bölüm başlığı */
.section-title { font-size: 1.15rem; font-weight: 700; margin: .2rem 0 .6rem 0; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# Eğitilmiş modeli ve yardımcı dosyaları yükle (cache ile bir kez yüklenir).
@st.cache_resource
def load_artifacts():
    model = joblib.load("churn_model.joblib")
    cols = joblib.load("model_columns.joblib")
    importance = joblib.load("feature_importance.joblib") if os.path.exists("feature_importance.joblib") else None
    business = joblib.load("business_metrics.joblib") if os.path.exists("business_metrics.joblib") else None
    preds = joblib.load("test_predictions.joblib") if os.path.exists("test_predictions.joblib") else None
    return model, cols, importance, business, preds


model, cols, importance_df, business, preds = load_artifacts()

# ---------------------------------------------------------------------------
# YAN PANEL (sidebar)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📉 Churn Predictor")
    st.markdown(
        "Telekom müşterilerinin **ayrılma (churn)** riskini tahmin eden uçtan uca "
        "bir makine öğrenmesi projesi."
    )
    st.divider()
    if business is not None:
        st.markdown("**🧠 Model**")
        st.write(f"`{business['model_name']}`")
        st.write(f"ROC-AUC: **{business['roc_auc']:.3f}**")
    st.divider()
    st.markdown("**🔗 Bağlantılar**")
    st.markdown("[GitHub Deposu](https://github.com/Cumali0/musteri-kaybi-tahmini)")
    st.caption("Veri: Telco Customer Churn (Kaggle)")
    st.caption("scikit-learn · XGBoost · SHAP · Streamlit")

# ---------------------------------------------------------------------------
# BAŞLIK (hero)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <h1>📉 Müşteri Kaybı (Churn) Tahmini</h1>
      <p>Hangi müşterilerin ayrılma riski taşıdığını önceden tahmin edip, kaybetmeden önce harekete geç.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# ÜST KPI ŞERİDİ — model performansını bir bakışta göster
# ---------------------------------------------------------------------------
if business is not None:
    toplam = business["tn"] + business["fp"] + business["fn"] + business["tp"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("En İyi Model", business["model_name"])
    k2.metric("ROC-AUC", f"{business['roc_auc']:.3f}")
    k3.metric("Recall (yakalama)", f"%{business['recall'] * 100:.0f}")
    k4.metric("Test Müşterisi", f"{toplam:,}")

tab_tahmin, tab_analiz, tab_roi, tab_batch = st.tabs(
    ["🔮 Tahmin", "📊 Model Analizi", "💰 İş Etkisi", "📁 Toplu Tahmin"]
)

# ===========================================================================
# SEKME 1: TAHMİN
# ===========================================================================
with tab_tahmin:
    st.markdown('<div class="section-title">🔮 Tek Müşteri Tahmini</div>', unsafe_allow_html=True)
    st.caption("Müşteri bilgilerini gir, model ayrılma olasılığını hesaplasın.")

    with st.container(border=True):
        user_input = {}
        col1, col2 = st.columns(2)

        # Kategorik alanlar: açılır menü (selectbox)
        cat_features = cols["categorical"]
        for i, feature in enumerate(cat_features):
            options = cols["categories"][feature]
            target_col = col1 if i % 2 == 0 else col2
            user_input[feature] = target_col.selectbox(feature, options)

        st.markdown("###### Sayısal Değerler")
        s1, s2 = st.columns(2)
        if "SeniorCitizen" in cols["numeric"]:
            senior = s1.selectbox("Yaşlı vatandaş mı? (SeniorCitizen)", ["Hayır", "Evet"])
            user_input["SeniorCitizen"] = 1 if senior == "Evet" else 0
        user_input["tenure"] = s2.slider("Müşteri süresi (ay) - tenure", 0, 72, 12)
        user_input["MonthlyCharges"] = s1.slider("Aylık ücret (MonthlyCharges)", 18.0, 120.0, 70.0)
        user_input["TotalCharges"] = s2.number_input("Toplam ücret (TotalCharges)", 0.0, 9000.0, 800.0)

        tahmin_butonu = st.button("Tahmin Et", type="primary", use_container_width=True)

    if tahmin_butonu:
        input_df = pd.DataFrame([user_input])
        beklenen = cols["numeric"] + cols["categorical"]
        eksik = [c for c in beklenen if c not in input_df.columns]
        if eksik:
            st.error(f"Form eksik sütun(lar) içeriyor: {eksik}. Lütfen train_model.py'yi tekrar çalıştırın.")
            st.stop()

        proba = model.predict_proba(input_df)[0][1]
        prediction = model.predict(input_df)[0]

        if prediction == 1:
            renk, etiket, alt = "#FF5A65", "⚠️ Yüksek Ayrılma Riski", "Elde tutma kampanyası önerilir."
        else:
            renk, etiket, alt = "#22C55E", "✅ Düşük Risk — Sadık Müşteri", "Bu müşterinin kalma olasılığı yüksek."

        st.markdown(
            f"""
            <div class="result-card" style="border-color:{renk}">
              <div class="result-pct" style="color:{renk}">%{proba * 100:.1f}</div>
              <div class="result-label">{etiket}</div>
              <div class="result-sub">{alt}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(float(proba))

# ===========================================================================
# SEKME 2: MODEL ANALİZİ (açıklanabilirlik)
# ===========================================================================
with tab_analiz:
    st.markdown('<div class="section-title">📊 Model Neye Göre Karar Veriyor?</div>', unsafe_allow_html=True)
    st.caption(
        "Hangi müşteri özelliklerinin ayrılma tahmininde en etkili olduğunu görüyorsun — "
        "yani model bir 'kara kutu' değil."
    )

    if importance_df is not None:
        st.markdown("**En Etkili Özellikler (Permutation Importance)**")
        chart_df = importance_df.set_index("feature")["importance"]
        st.bar_chart(chart_df, color="#6C5CE7")
        with st.expander("📋 Tabloyu göster"):
            st.dataframe(importance_df.reset_index(drop=True), use_container_width=True)
    else:
        st.info("Önce `python train_model.py` çalıştır; özellik önemi dosyası oluşacak.")

    st.markdown("**Üretilen Grafikler**")
    g1, g2 = st.columns(2)
    if os.path.exists("grafikler/feature_importance.png"):
        g1.image("grafikler/feature_importance.png", caption="Özellik Önemi (Permutation Importance)")
    if os.path.exists("grafikler/shap_summary.png"):
        g2.image("grafikler/shap_summary.png", caption="SHAP Özeti — her özelliğin tahmine katkısı")
    else:
        g2.caption("SHAP grafiği bulunamadı (SHAP atlanmış olabilir; proje yine de çalışır).")

    st.info(
        "💡 Tipik bulgu: **düşük müşteri süresi (tenure)**, **aya-dayalı (month-to-month) "
        "sözleşme** ve **yüksek aylık ücret** ayrılma riskini en çok artıran faktörlerdir."
    )

# ===========================================================================
# SEKME 3: İŞ ETKİSİ (ROI) + KARAR EŞİĞİ (THRESHOLD) OPTİMİZASYONU
# ===========================================================================
with tab_roi:
    st.markdown('<div class="section-title">💰 Bu Model Şirkete Ne Kadar Kazandırır?</div>', unsafe_allow_html=True)
    st.caption(
        "Skoru karar eşiğinin üstünde olan müşterilere kampanya yapılır. Değerleri değiştirerek "
        "net kazancı ve en kârlı eşiği canlı gör."
    )

    if preds is None or business is None:
        st.info("Önce `python train_model.py` çalıştır; gerekli dosyalar oluşacak.")
    else:
        y_test = np.asarray(preds["y_test"])
        y_proba = np.asarray(preds["y_proba"])

        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            musteri_degeri = c1.number_input("Müşteri yıllık geliri (TL)", 100, 100000, 1000, step=100)
            kampanya_maliyeti = c2.number_input("Kampanya maliyeti / kişi (TL)", 0, 5000, 100, step=10)
            basari_orani = c3.slider("Kampanya başarı oranı", 0.0, 1.0, 0.30, step=0.05)

            def roi_at(t):
                """Verilen eşik t için (hedeflenen, kurtarılan_gelir, net_kazanç) döndürür."""
                pred = (y_proba >= t).astype(int)
                tp = int(np.sum((pred == 1) & (y_test == 1)))
                fp = int(np.sum((pred == 1) & (y_test == 0)))
                hedef = tp + fp
                kurtarilan = tp * basari_orani * musteri_degeri
                net = kurtarilan - hedef * kampanya_maliyeti
                return hedef, kurtarilan, net

            threshold = st.slider(
                "Karar eşiği (bu olasılığın üstündekiler 'ayrılacak' sayılır)",
                0.05, 0.95, 0.50, step=0.05
            )

        hedef, kurtarilan, net = roi_at(threshold)
        m1, m2, m3 = st.columns(3)
        m1.metric("Hedeflenen müşteri", f"{hedef}")
        m2.metric("Kurtarılan gelir", f"{kurtarilan:,.0f} TL")
        m3.metric("Net Kazanç", f"{net:,.0f} TL")

        thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
        profits = [roi_at(t)[2] for t in thresholds]
        chart_df = pd.DataFrame({"Karar eşiği": thresholds, "Net kazanç (TL)": profits}).set_index("Karar eşiği")
        st.markdown("**Net kazanç, karar eşiğine göre nasıl değişiyor?**")
        st.line_chart(chart_df, color="#22C55E")

        best_i = int(np.argmax(profits))
        st.success(
            f"💡 En kârlı karar eşiği: **{thresholds[best_i]:.2f}** "
            f"→ net kazanç **{profits[best_i]:,.0f} TL**. "
            f"(Varsayılan 0.50 eşik her zaman en kârlı değildir — bu, ML'i doğrudan kâra bağlar.)"
        )

# ===========================================================================
# SEKME 4: TOPLU TAHMİN (CSV)
# ===========================================================================
with tab_batch:
    st.markdown('<div class="section-title">📁 Birden Çok Müşteriyi Aynı Anda Skorla</div>', unsafe_allow_html=True)
    st.caption(
        "Müşteri bilgilerini içeren bir CSV yükle; model her satır için ayrılma olasılığı hesaplasın. "
        "İpucu: Kaggle'daki orijinal Telco CSV'sini de yükleyebilirsin — customerID ve Churn sütunları otomatik atılır."
    )

    uploaded = st.file_uploader("CSV dosyası seç", type=["csv"])
    if uploaded is not None:
        try:
            data = pd.read_csv(uploaded)

            for col in ["customerID", "Churn"]:
                if col in data.columns:
                    data = data.drop(columns=[col])

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

                riskli = int((proba >= 0.5).sum())
                b1, b2, b3 = st.columns(3)
                b1.metric("Skorlanan müşteri", f"{len(sonuc):,}")
                b2.metric("Riskli (ayrılır)", f"{riskli:,}")
                b3.metric("Ortalama risk", f"%{proba.mean() * 100:.1f}")

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
