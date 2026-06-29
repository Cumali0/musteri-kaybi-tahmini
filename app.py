"""
Müşteri Kaybı (Churn) Tahmin Arayüzü - Streamlit  /  Customer Churn Prediction UI
=================================================================================
Önce 'python train_model.py' çalıştırıp modeli ürettiğinden emin ol.
Sonra:  streamlit run app.py

Arayüz çift dillidir (TR/EN, sidebar'dan seçilir) ve 4 sekmeden oluşur:
  1) Tahmin        - tek bir müşteri için churn olasılığı
  2) Model Analizi - hangi özellikler önemli (feature importance + SHAP)
  3) İş Etkisi     - modelin para değeri + karar eşiği (threshold) optimizasyonu
  4) Toplu Tahmin  - CSV yükle, tüm müşterileri skorla, sonucu indir
"""

import os
import numpy as np
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Churn Prediction",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# DİL / LANGUAGE — tüm metinler bu sözlükten gelir (i18n)
# ---------------------------------------------------------------------------
STR = {
    "side_desc": {
        "tr": "Telekom müşterilerinin **ayrılma (churn)** riskini tahmin eden uçtan uca bir makine öğrenmesi projesi.",
        "en": "An end-to-end machine learning project that predicts telecom **customer churn** risk.",
    },
    "side_model": {"tr": "🧠 Model", "en": "🧠 Model"},
    "side_links": {"tr": "🔗 Bağlantılar", "en": "🔗 Links"},
    "side_repo": {"tr": "GitHub Deposu", "en": "GitHub Repository"},
    "side_data": {"tr": "Veri: Telco Customer Churn (Kaggle)", "en": "Data: Telco Customer Churn (Kaggle)"},
    "hero_title": {"tr": "📉 Müşteri Kaybı (Churn) Tahmini", "en": "📉 Customer Churn Prediction"},
    "hero_sub": {
        "tr": "Hangi müşterilerin ayrılma riski taşıdığını önceden tahmin et, kaybetmeden önce harekete geç.",
        "en": "Predict which customers are at risk of leaving — and act before you lose them.",
    },
    "kpi_model": {"tr": "En İyi Model", "en": "Best Model"},
    "kpi_recall": {"tr": "Recall (yakalama)", "en": "Recall"},
    "kpi_customers": {"tr": "Test Müşterisi", "en": "Test Customers"},
    "tab_predict": {"tr": "🔮 Tahmin", "en": "🔮 Predict"},
    "tab_analysis": {"tr": "📊 Model Analizi", "en": "📊 Model Analysis"},
    "tab_roi": {"tr": "💰 İş Etkisi", "en": "💰 Business Impact"},
    "tab_batch": {"tr": "📁 Toplu Tahmin", "en": "📁 Batch Prediction"},
    # --- Tab 1
    "t1_title": {"tr": "🔮 Tek Müşteri Tahmini", "en": "🔮 Single Customer Prediction"},
    "t1_caption": {
        "tr": "Müşteri bilgilerini gir, model ayrılma olasılığını hesaplasın.",
        "en": "Enter customer details and let the model estimate the churn probability.",
    },
    "t1_numeric": {"tr": "Sayısal Değerler", "en": "Numeric Values"},
    "f_senior": {"tr": "Yaşlı vatandaş mı?", "en": "Senior citizen?"},
    "f_tenure": {"tr": "Müşteri süresi (ay)", "en": "Tenure (months)"},
    "f_monthly": {"tr": "Aylık ücret", "en": "Monthly charges"},
    "f_total": {"tr": "Toplam ücret", "en": "Total charges"},
    "t1_button": {"tr": "Tahmin Et", "en": "Predict"},
    "t1_missing": {
        "tr": "Form eksik sütun(lar) içeriyor: {x}. Lütfen train_model.py'yi tekrar çalıştırın.",
        "en": "The form is missing column(s): {x}. Please re-run train_model.py.",
    },
    "t1_high": {"tr": "⚠️ Yüksek Ayrılma Riski", "en": "⚠️ High Churn Risk"},
    "t1_high_sub": {"tr": "Elde tutma kampanyası önerilir.", "en": "A retention campaign is recommended."},
    "t1_low": {"tr": "✅ Düşük Risk — Sadık Müşteri", "en": "✅ Low Risk — Loyal Customer"},
    "t1_low_sub": {"tr": "Bu müşterinin kalma olasılığı yüksek.", "en": "This customer is likely to stay."},
    # --- Tab 2
    "t2_title": {"tr": "📊 Model Neye Göre Karar Veriyor?", "en": "📊 What Drives the Model's Decisions?"},
    "t2_caption": {
        "tr": "Hangi müşteri özelliklerinin ayrılma tahmininde en etkili olduğunu görüyorsun — yani model bir 'kara kutu' değil.",
        "en": "See which features matter most for the churn prediction — the model is not a 'black box'.",
    },
    "t2_imp": {"tr": "**En Etkili Özellikler (Permutation Importance)**", "en": "**Most Important Features (Permutation Importance)**"},
    "t2_table": {"tr": "📋 Tabloyu göster", "en": "📋 Show table"},
    "t2_noimp": {"tr": "Önce `python train_model.py` çalıştır; özellik önemi dosyası oluşacak.", "en": "Run `python train_model.py` first; the feature-importance file will be created."},
    "t2_charts": {"tr": "**Üretilen Grafikler**", "en": "**Generated Charts**"},
    "t2_fi_cap": {"tr": "Özellik Önemi (Permutation Importance)", "en": "Feature Importance (Permutation Importance)"},
    "t2_shap_cap": {"tr": "SHAP Özeti — her özelliğin tahmine katkısı", "en": "SHAP Summary — each feature's contribution"},
    "t2_noshap": {"tr": "SHAP grafiği bulunamadı (SHAP atlanmış olabilir; proje yine de çalışır).", "en": "SHAP chart not found (SHAP may have been skipped; the project still works)."},
    "t2_tip": {
        "tr": "💡 Tipik bulgu: **düşük müşteri süresi (tenure)**, **aya-dayalı (month-to-month) sözleşme** ve **yüksek aylık ücret** ayrılma riskini en çok artıran faktörlerdir.",
        "en": "💡 Typical finding: **low tenure**, **month-to-month contracts** and **high monthly charges** are the strongest churn drivers.",
    },
    # --- Tab 3
    "t3_title": {"tr": "💰 Bu Model Şirkete Ne Kadar Kazandırır?", "en": "💰 How Much Does This Model Save?"},
    "t3_caption": {
        "tr": "Skoru karar eşiğinin üstünde olan müşterilere kampanya yapılır. Değerleri değiştirerek net kazancı ve en kârlı eşiği canlı gör.",
        "en": "Customers scoring above the decision threshold get a campaign. Adjust the values to see net profit and the most profitable threshold live.",
    },
    "t3_nodata": {"tr": "Önce `python train_model.py` çalıştır; gerekli dosyalar oluşacak.", "en": "Run `python train_model.py` first; the required files will be created."},
    "t3_value": {"tr": "Müşteri yıllık geliri (TL)", "en": "Customer annual value (TL)"},
    "t3_cost": {"tr": "Kampanya maliyeti / kişi (TL)", "en": "Campaign cost / person (TL)"},
    "t3_rate": {"tr": "Kampanya başarı oranı", "en": "Campaign success rate"},
    "t3_thr": {
        "tr": "Karar eşiği (bu olasılığın üstündekiler 'ayrılacak' sayılır)",
        "en": "Decision threshold (above this probability = 'will churn')",
    },
    "t3_targeted": {"tr": "Hedeflenen müşteri", "en": "Targeted customers"},
    "t3_saved": {"tr": "Kurtarılan gelir", "en": "Saved revenue"},
    "t3_net": {"tr": "Net Kazanç", "en": "Net Profit"},
    "t3_chart": {"tr": "**Net kazanç, karar eşiğine göre nasıl değişiyor?**", "en": "**How does net profit change with the decision threshold?**"},
    "t3_best": {
        "tr": "💡 En kârlı karar eşiği: **{thr:.2f}** → net kazanç **{profit:,.0f} TL**. (Varsayılan 0.50 eşik her zaman en kârlı değildir — bu, ML'i doğrudan kâra bağlar.)",
        "en": "💡 Most profitable threshold: **{thr:.2f}** → net profit **{profit:,.0f} TL**. (The default 0.50 is not always optimal — this ties ML directly to profit.)",
    },
    # --- Tab 4
    "t4_title": {"tr": "📁 Birden Çok Müşteriyi Aynı Anda Skorla", "en": "📁 Score Many Customers at Once"},
    "t4_caption": {
        "tr": "Müşteri bilgilerini içeren bir CSV yükle; model her satır için ayrılma olasılığı hesaplasın. İpucu: Kaggle'daki orijinal Telco CSV'sini de yükleyebilirsin — customerID ve Churn sütunları otomatik atılır.",
        "en": "Upload a CSV of customers; the model scores each row. Tip: you can upload the original Kaggle Telco CSV — customerID and Churn columns are dropped automatically.",
    },
    "t4_upload": {"tr": "CSV dosyası seç", "en": "Choose a CSV file"},
    "t4_missing": {"tr": "CSV'de eksik sütunlar var: {x}", "en": "Missing columns in CSV: {x}"},
    "t4_scored": {"tr": "Skorlanan müşteri", "en": "Customers scored"},
    "t4_risky": {"tr": "Riskli (ayrılır)", "en": "At risk (will churn)"},
    "t4_avg": {"tr": "Ortalama risk", "en": "Average risk"},
    "t4_top": {"tr": "**En riskli 10 müşteri:**", "en": "**Top 10 riskiest customers:**"},
    "t4_download": {"tr": "📥 Tüm sonuçları indir (CSV)", "en": "📥 Download all results (CSV)"},
    "t4_error": {"tr": "Dosya işlenirken hata oluştu: {e}", "en": "Error while processing the file: {e}"},
}

# Veri setindeki İngilizce özellik adlarının Türkçe karşılıkları (ekranda gösterim için)
FEATURE_TR = {
    "gender": "Cinsiyet", "Partner": "Eşi var mı?", "Dependents": "Bakmakla yükümlü kişi?",
    "PhoneService": "Telefon hizmeti", "MultipleLines": "Çoklu hat", "InternetService": "İnternet hizmeti",
    "OnlineSecurity": "Çevrimiçi güvenlik", "OnlineBackup": "Çevrimiçi yedekleme",
    "DeviceProtection": "Cihaz koruma", "TechSupport": "Teknik destek",
    "StreamingTV": "TV yayını", "StreamingMovies": "Film yayını", "Contract": "Sözleşme tipi",
    "PaperlessBilling": "Kağıtsız fatura", "PaymentMethod": "Ödeme yöntemi",
}
# Değerlerin Türkçe karşılıkları (modele yine İngilizce orijinal değer gider)
VALUE_TR = {
    "Yes": "Evet", "No": "Hayır", "Male": "Erkek", "Female": "Kadın",
    "No phone service": "Telefon hizmeti yok", "No internet service": "İnternet hizmeti yok",
    "DSL": "DSL", "Fiber optic": "Fiber optik",
    "Month-to-month": "Aylık", "One year": "1 Yıllık", "Two year": "2 Yıllık",
    "Electronic check": "Elektronik çek", "Mailed check": "Posta çeki",
    "Bank transfer (automatic)": "Banka havalesi (otomatik)", "Credit card (automatic)": "Kredi kartı (otomatik)",
}

# ---------------------------------------------------------------------------
# ÖZEL CSS — modern "dashboard" görünümü
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1150px; }
footer { visibility: hidden; }
.hero { background: linear-gradient(135deg, #6C5CE7 0%, #4338CA 55%, #2563EB 100%);
  padding: 2.1rem 2.2rem; border-radius: 18px; margin-bottom: 1.4rem;
  box-shadow: 0 12px 34px rgba(99,91,255,0.28); }
.hero h1 { color: #fff; font-size: 2rem; margin: 0 0 .45rem 0; font-weight: 800; letter-spacing: -.5px; }
.hero p  { color: rgba(255,255,255,0.9); font-size: 1.03rem; margin: 0; }
[data-testid="stMetric"] { background: #161B26; border: 1px solid #262C3A; border-radius: 14px;
  padding: 1rem 1.1rem; box-shadow: 0 2px 10px rgba(0,0,0,0.25); }
[data-testid="stMetricValue"] { font-size: 1.55rem; font-weight: 700; }
.stButton > button, .stDownloadButton > button { border-radius: 10px; font-weight: 600; height: 3rem;
  border: none; transition: transform .06s ease, box-shadow .2s ease; }
.stButton > button:hover, .stDownloadButton > button:hover { transform: translateY(-1px);
  box-shadow: 0 6px 18px rgba(99,91,255,0.35); }
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] { border-radius: 10px 10px 0 0; padding: 9px 18px; font-weight: 600; }
.result-card { text-align: center; padding: 1.7rem 1.5rem; border-radius: 18px;
  background: #161B26; border: 2px solid #333; margin: .4rem 0 1rem 0; }
.result-pct   { font-size: 3.4rem; font-weight: 800; line-height: 1; }
.result-label { font-size: 1.15rem; margin-top: .55rem; font-weight: 700; }
.result-sub   { opacity: .7; margin-top: .3rem; font-size: .9rem; }
.section-title { font-size: 1.15rem; font-weight: 700; margin: .2rem 0 .6rem 0; }
</style>
""",
    unsafe_allow_html=True,
)


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
# YAN PANEL (sidebar) — önce dil seçilir, sonra geri kalan içerik
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📉 Churn Predictor")
    secim = st.radio("🌐 Dil / Language", ["Türkçe", "English"], horizontal=True)
    lang = "tr" if secim == "Türkçe" else "en"

    # Dil seçildikten sonra kısayol fonksiyonları
    def t(key, **kw):
        s = STR[key][lang]
        return s.format(**kw) if kw else s

    def flabel(col):
        return FEATURE_TR.get(col, col) if lang == "tr" else col

    def vlabel(val):
        return VALUE_TR.get(val, val) if lang == "tr" else val

    st.markdown(t("side_desc"))
    st.divider()
    if business is not None:
        st.markdown(f"**{t('side_model')}**")
        st.write(f"`{business['model_name']}`")
        st.write(f"ROC-AUC: **{business['roc_auc']:.3f}**")
    st.divider()
    st.markdown(f"**{t('side_links')}**")
    st.markdown(f"[{t('side_repo')}](https://github.com/Cumali0/musteri-kaybi-tahmini)")
    st.caption(t("side_data"))
    st.caption("scikit-learn · XGBoost · SHAP · Streamlit")

# ---------------------------------------------------------------------------
# BAŞLIK (hero) + KPI şeridi
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero">
      <h1>{t('hero_title')}</h1>
      <p>{t('hero_sub')}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if business is not None:
    toplam = business["tn"] + business["fp"] + business["fn"] + business["tp"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(t("kpi_model"), business["model_name"])
    k2.metric("ROC-AUC", f"{business['roc_auc']:.3f}")
    k3.metric(t("kpi_recall"), f"%{business['recall'] * 100:.0f}")
    k4.metric(t("kpi_customers"), f"{toplam:,}")

tab_tahmin, tab_analiz, tab_roi, tab_batch = st.tabs(
    [t("tab_predict"), t("tab_analysis"), t("tab_roi"), t("tab_batch")]
)

# ===========================================================================
# SEKME 1: TAHMİN
# ===========================================================================
with tab_tahmin:
    st.markdown(f'<div class="section-title">{t("t1_title")}</div>', unsafe_allow_html=True)
    st.caption(t("t1_caption"))

    with st.container(border=True):
        user_input = {}
        col1, col2 = st.columns(2)

        cat_features = cols["categorical"]
        for i, feature in enumerate(cat_features):
            options = cols["categories"][feature]
            target_col = col1 if i % 2 == 0 else col2
            # format_func: ekranda çevrilmiş etiket; dönen değer yine orijinal İngilizce
            user_input[feature] = target_col.selectbox(flabel(feature), options, format_func=vlabel)

        st.markdown(f"###### {t('t1_numeric')}")
        s1, s2 = st.columns(2)
        if "SeniorCitizen" in cols["numeric"]:
            evet_hayir = ["Evet", "Hayır"] if lang == "tr" else ["Yes", "No"]
            sec = s1.selectbox(t("f_senior"), evet_hayir)
            user_input["SeniorCitizen"] = 1 if sec in ("Evet", "Yes") else 0
        user_input["tenure"] = s2.slider(t("f_tenure"), 0, 72, 12)
        user_input["MonthlyCharges"] = s1.slider(t("f_monthly"), 18.0, 120.0, 70.0)
        user_input["TotalCharges"] = s2.number_input(t("f_total"), 0.0, 9000.0, 800.0)

        tahmin_butonu = st.button(t("t1_button"), type="primary", width="stretch")

    if tahmin_butonu:
        input_df = pd.DataFrame([user_input])
        beklenen = cols["numeric"] + cols["categorical"]
        eksik = [c for c in beklenen if c not in input_df.columns]
        if eksik:
            st.error(t("t1_missing", x=eksik))
            st.stop()

        proba = model.predict_proba(input_df)[0][1]
        prediction = model.predict(input_df)[0]

        if prediction == 1:
            renk, etiket, alt = "#FF5A65", t("t1_high"), t("t1_high_sub")
        else:
            renk, etiket, alt = "#22C55E", t("t1_low"), t("t1_low_sub")

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
# SEKME 2: MODEL ANALİZİ
# ===========================================================================
with tab_analiz:
    st.markdown(f'<div class="section-title">{t("t2_title")}</div>', unsafe_allow_html=True)
    st.caption(t("t2_caption"))

    if importance_df is not None:
        st.markdown(t("t2_imp"))
        chart_df = importance_df.set_index("feature")["importance"]
        st.bar_chart(chart_df, color="#6C5CE7")
        with st.expander(t("t2_table")):
            st.dataframe(importance_df.reset_index(drop=True), width="stretch")
    else:
        st.info(t("t2_noimp"))

    st.markdown(t("t2_charts"))
    g1, g2 = st.columns(2)
    if os.path.exists("grafikler/feature_importance.png"):
        g1.image("grafikler/feature_importance.png", caption=t("t2_fi_cap"))
    if os.path.exists("grafikler/shap_summary.png"):
        g2.image("grafikler/shap_summary.png", caption=t("t2_shap_cap"))
    else:
        g2.caption(t("t2_noshap"))

    st.info(t("t2_tip"))

# ===========================================================================
# SEKME 3: İŞ ETKİSİ (ROI) + THRESHOLD
# ===========================================================================
with tab_roi:
    st.markdown(f'<div class="section-title">{t("t3_title")}</div>', unsafe_allow_html=True)
    st.caption(t("t3_caption"))

    if preds is None or business is None:
        st.info(t("t3_nodata"))
    else:
        y_test = np.asarray(preds["y_test"])
        y_proba = np.asarray(preds["y_proba"])

        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            musteri_degeri = c1.number_input(t("t3_value"), 100, 100000, 1000, step=100)
            kampanya_maliyeti = c2.number_input(t("t3_cost"), 0, 5000, 100, step=10)
            basari_orani = c3.slider(t("t3_rate"), 0.0, 1.0, 0.30, step=0.05)

            def roi_at(thr):
                pred = (y_proba >= thr).astype(int)
                tp = int(np.sum((pred == 1) & (y_test == 1)))
                fp = int(np.sum((pred == 1) & (y_test == 0)))
                hedef = tp + fp
                kurtarilan = tp * basari_orani * musteri_degeri
                net = kurtarilan - hedef * kampanya_maliyeti
                return hedef, kurtarilan, net

            threshold = st.slider(t("t3_thr"), 0.05, 0.95, 0.50, step=0.05)

        hedef, kurtarilan, net = roi_at(threshold)
        m1, m2, m3 = st.columns(3)
        m1.metric(t("t3_targeted"), f"{hedef}")
        m2.metric(t("t3_saved"), f"{kurtarilan:,.0f} TL")
        m3.metric(t("t3_net"), f"{net:,.0f} TL")

        thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
        profits = [roi_at(thr)[2] for thr in thresholds]
        chart_df = pd.DataFrame({"threshold": thresholds, "net": profits}).set_index("threshold")
        st.markdown(t("t3_chart"))
        st.line_chart(chart_df, color="#22C55E")

        best_i = int(np.argmax(profits))
        st.success(t("t3_best", thr=thresholds[best_i], profit=profits[best_i]))

# ===========================================================================
# SEKME 4: TOPLU TAHMİN (CSV)
# ===========================================================================
with tab_batch:
    st.markdown(f'<div class="section-title">{t("t4_title")}</div>', unsafe_allow_html=True)
    st.caption(t("t4_caption"))

    uploaded = st.file_uploader(t("t4_upload"), type=["csv"])
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
                st.error(t("t4_missing", x=eksik))
            else:
                proba = model.predict_proba(data[beklenen])[:, 1]
                sonuc = data.copy()
                sonuc["churn_olasiligi_%"] = (proba * 100).round(1)
                sonuc["tahmin"] = np.where(proba >= 0.5, "Ayrılır" if lang == "tr" else "Churn", "Kalır" if lang == "tr" else "Stay")

                riskli = int((proba >= 0.5).sum())
                b1, b2, b3 = st.columns(3)
                b1.metric(t("t4_scored"), f"{len(sonuc):,}")
                b2.metric(t("t4_risky"), f"{riskli:,}")
                b3.metric(t("t4_avg"), f"%{proba.mean() * 100:.1f}")

                st.markdown(t("t4_top"))
                st.dataframe(
                    sonuc.sort_values("churn_olasiligi_%", ascending=False).head(10),
                    width="stretch",
                )

                csv_out = sonuc.to_csv(index=False).encode("utf-8")
                st.download_button(
                    t("t4_download"), csv_out,
                    file_name="churn_tahminleri.csv", mime="text/csv",
                    width="stretch",
                )
        except Exception as e:
            st.error(t("t4_error", e=e))
