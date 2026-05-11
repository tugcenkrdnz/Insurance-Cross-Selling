# 🚗 Insurance Cross Selling Prediction (Kaggle S4E7)

Bu proje, Kaggle **Playground Series Season 4, Episode 7** yarışması için geliştirilmiştir. Veri seti kullanılarak, mevcut sağlık sigortası müşterilerinin araç sigortası teklifine ilgi duyup duymayacağı tahmin edilmektedir.

## 📊 Proje Özeti
Bu bir **İkili Sınıflandırma (Binary Classification)** problemidir. Model, müşteri demografisi, araç geçmişi ve poliçe detaylarını analiz ederek `Response` (0 veya 1) tahmini yapar.

## 🛠️ Kullanılan Teknolojiler & Yöntemler
- **Model:** LightGBM (Gradient Boosting)
- **Doğrulama:** Stratified K-Fold Cross-Validation (5-Fold)
- **Veri İşleme:** 
  - Bellek Optimizasyonu (Downcasting)
  - Feature Engineering (Age Grouping, Risk Score, Insured & Damage Interaction)
  - Outlier Capping (Annual Premium için %99 clipping)
- **Arayüz:** Streamlit

## 🚀 Başarı Metriği
Model performansı **ROC-AUC** skoru ile ölçülmüştür.
- **Yerel CV Skoru:** 0.87389

## 📂 Dosya Yapısı
- `app.py`: Streamlit web uygulaması kodu.
- `cleaned_data.py`: Veri ön işleme fonksiyonları.
- `final_lgbm.txt`: Eğitilmiş LightGBM model dosyası.
- `requirements.txt`: Gerekli kütüphaneler listesi.

## 💻 Kurulum ve Çalıştırma
Projeyi yerel makinenizde çalıştırmak için:

1. Gereksinimleri yükleyin:
   ```bash
   pip install -r requirements.txt

2. Streamlit uygulamasını başlatın:
    ```bash
   streamlit run app.py
