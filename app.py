import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime

st.set_page_config(page_title="Sistem Stok Ulang Toko Retail Mingguan", page_icon="🛒", layout="wide")

st.title("🛒 Sistem Stok Ulang Toko Retail Mingguan")
st.divider()

# ==========================================
# 1. MEMUAT MODEL
# ==========================================
@st.cache_resource
def load_models():
    folder = "model"
    return joblib.load(f"{folder}/model_lgbm_toko.pkl")

model_lgbm = load_models()

# ==========================================
# 2. KONDISI FISIK TOKO
# ==========================================
st.subheader("📦 1. Cek Fisik Barang di Toko")
col_A, col_B, col_C = st.columns(3)

with col_A:
    nama_barang = st.text_input("📝 Nama / ID Barang (Opsional):", value="Roti Coklat", help="Hanya untuk penamaan.")

with col_B:
    stok_sekarang = st.number_input("📦 Cek Fisik Stok (pcs)", min_value=0, help="Input sisa stok barang di toko.")

with col_C:
    # MAE LGBM = 3.3255  
    safety_stock = st.number_input("🛡️ Safety Stock (pcs)", min_value=0, value=3, help="Default 3 pcs (Berdasarkan nilai Error/MAE algoritma LightGBM).")

st.divider()

# ==========================================
# 3. PROFIL PERILAKU PENJUALAN
# ==========================================
st.subheader("📝 2. Profil Perilaku Penjualan Mingguan")

hari_ini = datetime.date.today()

col1, col2, col3 = st.columns(3)

with col1:
    target_date = st.date_input("📅 Tanggal Target Prediksi (Minggu Depan):", value=hari_ini + datetime.timedelta(days=7))
    
    # EKSTRAKSI OTOMATIS
    month = target_date.month
    week_of_year = target_date.isocalendar()[1]
    
    is_holiday = st.selectbox("🎉 Apakah Minggu Tersebut Ada Hari Libur Nasional / Event?", [0, 1], format_func=lambda x: "Ya (1)" if x==1 else "Tidak (0)")

with col2:
    st.info(f"📌 **Info:** Sistem akan memprediksi untuk **Bulan ke-{month}**, pada **Minggu ke-{week_of_year}** dalam tahun ini.")
    
    kategori_list = [
        "KEBUTUHAN RUMAH TANGGA", "LAIN-LAIN", "MAKANAN INSTAN", "MINUMAN", 
        "OBAT", "PERAWATAN DIRI DAN KOSMETIK", "PERLENGKAPAN BAYI", 
        "PRODUK KHUSUS", "SEMBAKO", "SNACK"
    ]
    kategori_input = st.selectbox("🗂️ Kategori Produk", kategori_list)
    
    item_id_avg_sales = st.number_input("📊 Rata-rata Penjualan Historis Barang Ini", value=20.0)

with col3:
    qty_lag_1w = st.number_input("📈 Laku Minggu Lalu (Lag 1W)", value=22.0)
    qty_lag_2w = st.number_input("📈 Laku 2 Minggu Lalu (Lag 2W)", value=18.0)
    qty_roll_mean_4w = st.number_input("📊 Rata-rata 4 Minggu Terakhir", value=20.5)

# ==========================================
# 4. TOMBOL EKSEKUSI & KALKULASI SUPPLY CHAIN
# ==========================================
st.divider()
if st.button("🚀 HITUNG REKOMENDASI ORDER!", type="primary", use_container_width=True):
    
    # --- PROSES ONE-HOT ENCODING KATEGORI ---
    kategori_dict = {
        'Kategori_KEBUTUHAN RUMAH TANGGA': 0, 'Kategori_LAIN-LAIN': 0, 
        'Kategori_MAKANAN INSTAN': 0, 'Kategori_MINUMAN': 0, 
        'Kategori_OBAT': 0, 'Kategori_PERAWATAN DIRI DAN KOSMETIK': 0, 
        'Kategori_PERLENGKAPAN BAYI': 0, 'Kategori_PRODUK KHUSUS': 0, 
        'Kategori_SEMBAKO': 0, 'Kategori_SNACK': 0
    }
    
    kategori_terpilih = f"Kategori_{kategori_input}"
    if kategori_terpilih in kategori_dict:
        kategori_dict[kategori_terpilih] = 1
        
    # --- MENYIAPKAN DATAFRAME SESUAI URUTAN FITUR TRAINING ---
    input_dict = {
        'Month': month,
        'WeekOfYear': week_of_year,
        'Is_Holiday': is_holiday,
        'Qty_Lag_1W': qty_lag_1w,
        'Qty_Lag_2W': qty_lag_2w,
        'Qty_Roll_Mean_4W': qty_roll_mean_4w,
        **kategori_dict, # Memasukkan 10 kolom kategori
        'item_id_Avg_Sales': item_id_avg_sales
    }
    
    urutan_kolom = [
        'Month', 'WeekOfYear', 'Is_Holiday', 'Qty_Lag_1W', 'Qty_Lag_2W', 'Qty_Roll_Mean_4W',
        'Kategori_KEBUTUHAN RUMAH TANGGA', 'Kategori_LAIN-LAIN', 'Kategori_MAKANAN INSTAN', 
        'Kategori_MINUMAN', 'Kategori_OBAT', 'Kategori_PERAWATAN DIRI DAN KOSMETIK', 
        'Kategori_PERLENGKAPAN BAYI', 'Kategori_PRODUK KHUSUS', 'Kategori_SEMBAKO', 
        'Kategori_SNACK', 'item_id_Avg_Sales'
    ]
    
    input_data = pd.DataFrame([input_dict], columns=urutan_kolom)

    # --- PREDIKSI & KALKULASI ---
    pred_lgbm = max(0, round(model_lgbm.predict(input_data)[0]))
        
    # Kalkulasi Supply Chain untuk LightGBM
    order_lgbm = max(0, (pred_lgbm + safety_stock) - stok_sekarang)
    
    # ==========================================
    # TAMPILAN HASIL
    # ==========================================
    st.subheader(f"📊 Laporan Keputusan Order: {nama_barang}")
    st.caption(f"Kondisi Saat Ini: **Safety Stock = {safety_stock} pcs** | **Sisa Stok di Toko = {stok_sekarang} pcs**")
    st.write("")
    
    # Membuat 3 kolom
    col_kiri, col_tengah, col_kanan = st.columns([1, 2, 1])
    
    with col_tengah:
        st.success("⚡ **Rekomendasi LightGBM**")
        st.metric(label=f"🧠 Prediksi AI (Estimasi Laku Minggu ke-{week_of_year})", value=f"{pred_lgbm} pcs")
        
        if order_lgbm > 0:
            st.metric(label="🛒 KEPUTUSAN ORDER (Kuantitas Restock)", value=f"{order_lgbm} pcs", delta="Perlu Order Barang Segera!")
        else:
            st.metric(label="🛒 KEPUTUSAN ORDER (Kuantitas Restock)", value=f"{order_lgbm} pcs", delta="Stok Masih Aman", delta_color="off")
                
        st.divider()