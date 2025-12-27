import sqlite3
import pandas as pd
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'FPA_FOD_20170508.sqlite')
OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'wildfire_data.parquet')

def load_from_sqlite():
    
    print(f" Veritabanına bağlanılıyor: {DB_PATH}")
    
    if not os.path.exists(DB_PATH):
        print(f" HATA: Dosya bulunamadı! Lütfen dosyayı şuraya taşıdığından emin ol: {DB_PATH}")
        sys.exit(1)

    try:
        conn = sqlite3.connect(DB_PATH)
        
        query = """
        SELECT 
            FIRE_YEAR,
            DISCOVERY_DATE,
            STAT_CAUSE_DESCR,
            LATITUDE,
            LONGITUDE,
            STATE,
            FIRE_SIZE,
            FIRE_SIZE_CLASS
        FROM Fires
        """
        
        print(" Sorgu çalıştırılıyor ve veri çekiliyor (Bu işlem 1-2 dk sürebilir)...")
        df = pd.read_sql_query(query, conn)
        conn.close()
        print(f" Veri başarıyla çekildi. Satır sayısı: {len(df)}")
        return df
        
    except Exception as e:
        print(f" Beklenmedik bir hata oluştu: {e}")
        sys.exit(1)

def transform_data(df):

    print(" Veri dönüştürülüyor (Transformation)...")
    
    df['DISCOVERY_DATE'] = pd.to_datetime(
        df['DISCOVERY_DATE'], 
        unit='D', 
        origin='julian'
    )
    
    df['FIRE_YEAR'] = df['FIRE_YEAR'].astype('int16') 
    
    return df

def save_to_parquet(df):
    
    print(f" Veri Parquet formatında kaydediliyor: {OUTPUT_PATH}")
    
    df.to_parquet(OUTPUT_PATH, index=False)
    print(" İŞLEM TAMAM! ETL Süreci başarıyla bitti.")

if __name__ == "__main__":
    
    raw_df = load_from_sqlite()  # 1. Extract
    clean_df = transform_data(raw_df) # 2. Transform
    save_to_parquet(clean_df)    # 3. Load