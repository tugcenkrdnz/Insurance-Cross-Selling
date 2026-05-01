import streamlit as st
import pandas as pd
import lightgbm as lgb
import numpy as np

# 1. Sayfa Yapılandırması
st.set_page_config(
    page_title="Sigorta Çapraz Satış Tahmini",
    page_icon="🚗",
    layout="centered"
)

# 2. Model Yükleme (Cache kullanarak hızlandırıyoruz)
@st.cache_resource
def load_my_model():
    # Model dosyasının app.py ile aynı klasörde olduğundan emin ol
    return lgb.Booster(model_file='lgbm_model.txt')

try:
    model = load_my_model()
except Exception as e:
    st.error(f"Model dosyası yüklenemedi! Hata: {e}")
    st.stop()

# 3. Başlık ve Açıklama
st.title("🚗 Araç Sigortası Satış Tahmin Paneli")
st.markdown("""
Bu uygulama, müşterinin demografik ve araç bilgilerini kullanarak, 
ek araç sigortası teklifine **ilgi duyup duymayacağını** tahmin eder.
""")

# 4. Kullanıcı Giriş Alanları (Sidebar)
st.sidebar.header("Müşteri Verilerini Girin")

gender = st.sidebar.selectbox("Cinsiyet", ["Male", "Female"])
age = st.sidebar.slider("Yaş", 18, 90, 35)
driving_license = st.sidebar.selectbox("Ehliyet Var mı?", [1, 0], format_func=lambda x: "Evet" if x == 1 else "Hayır")
region_code = st.sidebar.number_input("Bölge Kodu", 0, 52, 28)
prev_insured = st.sidebar.selectbox("Önceden Sigortalı mı?", [1, 0], format_func=lambda x: "Evet" if x == 1 else "Hayır")
vehicle_age = st.sidebar.selectbox("Araç Yaşı", ["< 1 Year", "1-2 Year", "> 2 Years"])
vehicle_damage = st.sidebar.selectbox("Araç Hasar Geçmişi?", ["Yes", "No"])
premium = st.sidebar.number_input("Yıllık Prim (Annual Premium)", 2000, 150000, 35000)
channel = st.sidebar.number_input("Satış Kanalı Kodu", 1, 163, 124)
vintage = st.sidebar.slider("Müşteri Sadakati (Gün)", 10, 300, 150)

# 5. Tahmin Butonu ve Hesaplama
if st.button("Tahmin Et"):
    # Ham veri sözlüğü
    data_dict = {
        'Gender': 1 if gender == "Male" else 0,
        'Age': age,
        'Driving_License': driving_license,
        'Region_Code': region_code,
        'Previously_Insured': prev_insured,
        'Vehicle_Age': 0 if vehicle_age == "< 1 Year" else (1 if vehicle_age == "1-2 Year" else 2),
        'Vehicle_Damage': 1 if vehicle_damage == "Yes" else 0,
        'Annual_Premium': float(premium),
        'Policy_Sales_Channel': float(channel),
        'Vintage': int(vintage)
    }
    
    # DataFrame oluşturma
    df_input = pd.DataFrame([data_dict])
    
    # Feature Engineering (Eğitimdeki 14 sütuna tamamlıyoruz)
    df_input['Age_Group'] = pd.cut(df_input['Age'], bins=[0, 25, 45, 65, 100], labels=[0, 1, 2, 3]).astype(int)
    df_input['Risk_Score'] = (df_input['Vehicle_Damage'] + (1 - df_input['Previously_Insured'])).astype(int)
    df_input['Insured_Damage_Score'] = (df_input['Previously_Insured'] == 0).astype(int) + (df_input['Vehicle_Damage'] == 1).astype(int) * 2
    
    # Kritik: Region_Density için eğitimdeki ortalama bir değer veya mantıklı bir placeholder
    # Eğer elinde region_counts varsa buraya onu eklemek daha doğrudur.
    df_input['Region_Density'] = 5000.0  # Placeholder değer
    
    # Sütun sırasını eğitimdekiyle eşitlemek (Çok Önemli!)
    expected_columns = [
        'Gender', 'Age', 'Driving_License', 'Region_Code', 'Previously_Insured',
        'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium', 'Policy_Sales_Channel',
        'Vintage', 'Age_Group', 'Risk_Score', 'Insured_Damage_Score', 'Region_Density'
    ]
    
    # Sütunları doğru sıraya diz
    df_input = df_input[expected_columns]
    
    # Tahmin yapma
    prediction_proba = model.predict(df_input)[0]
    
    # Sonuçları Görselleştirme
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("İlgi Olasılığı", f"%{prediction_proba*100:.2f}")
    
    with col2:
        if prediction_proba > 0.5:
            st.success("🎯 SATIŞ POTANSİYELİ YÜKSEK")
        else:
            st.warning("💤 İLGİ DÜŞÜK")

    # Detaylı Progress Bar
    st.progress(prediction_proba)
    
    if prediction_proba > 0.8:
        st.balloons()
        st.info("Bu müşteri için hemen aksiyon alınmalı!")