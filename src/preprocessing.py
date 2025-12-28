import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import LabelEncoder, RobustScaler

# Dosya Yolları 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'wildfire_data.parquet')
OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'wildfire_data_clean.parquet')
ENCODER_PATH = os.path.join(BASE_DIR, 'src', 'label_encoder.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'src', 'scaler.pkl')

def load_data():
    print(" Veri yükleniyor...")
    return pd.read_parquet(INPUT_PATH)

def clean_data(df):
    initial_len = len(df)
    
    df = df.dropna().copy()
    print(f" Temizlik: {initial_len - len(df)} adet boş satır silindi.")
    return df

def encode_target(df):
    le = LabelEncoder()

    df['FIRE_SIZE_CLASS_ID'] = le.fit_transform(df['FIRE_SIZE_CLASS'])
    joblib.dump(le, ENCODER_PATH)
    print(f" Sınıflar sayısallaştırıldı: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    return df

def feature_engineering(df):
    print(" Özellik Mühendisliği yapılıyor...")
    
    # 1. Tarih Parçalamaca 
    df['DISCOVERY_DATE'] = pd.to_datetime(df['DISCOVERY_DATE'])
    df['MONTH'] = df['DISCOVERY_DATE'].dt.month
    df['DAY_OF_WEEK'] = df['DISCOVERY_DATE'].dt.dayofweek

    # 2. Cyclical Encoding 
    df['MONTH_SIN'] = np.sin(2 * np.pi * df['MONTH'] / 12)
    df['MONTH_COS'] = np.cos(2 * np.pi * df['MONTH'] / 12)

    # 3. Kategorik Sebepleri Sayısallaştırma (One-Hot Encoding)
    df = pd.get_dummies(df, columns=['STAT_CAUSE_DESCR'], prefix='CAUSE', dtype='int8')
    
    # 4. Gereksiz Kolonları Atma
    drop_cols = ['FIRE_SIZE_CLASS', 'DISCOVERY_DATE', 'STATE', 'FIRE_YEAR', 'DATE_READABLE', 'MONTH']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    
    return df

def scaling_and_binning(df):
    
    print(" Ölçeklendirme ve Coğrafi Gruplama yapılıyor...")
    
    # Spatial Binning: Koordinatları gruplayarak modelin bölgesel riskleri anlamasını sağlıyoruz
    df['LAT_BIN'] = (df['LATITUDE'] * 10).astype(int)
    df['LON_BIN'] = (df['LONGITUDE'] * 10).astype(int)
    
    # Robust Scaling: Aykırı değerlere karşı dayanıklı ölçeklendirme
    scale_cols = ['LATITUDE', 'LONGITUDE', 'MONTH_SIN', 'MONTH_COS']
    scaler = RobustScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])
    
    joblib.dump(scaler, SCALER_PATH)
    return df

if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    df = encode_target(df)
    df = feature_engineering(df)
    df = scaling_and_binning(df)
    
    print(f" İşlenmiş veri kaydediliyor: {OUTPUT_PATH}")
    df.to_parquet(OUTPUT_PATH)
    print(" PREPROCESSING TAMAMLANDI! Model eğitimine hazır.")
    print(f" Son Veri Boyutu (Satır, Sütun): {df.shape}")