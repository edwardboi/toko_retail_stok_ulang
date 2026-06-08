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
    models = {
        "1_minggu": joblib.load(f"{folder}/model_lgbm_toko.pkl"),
        "2_minggu": joblib.load(f"{folder}/model_lgbm_toko_plus2.pkl"),
        "3_minggu": joblib.load(f"{folder}/model_lgbm_toko_plus3.pkl")
    }
    return models

models = load_models()

# ==========================================
# DATA PIPELINE
# ==========================================
@st.cache_data
def process_raw_transaction_data(df):
    """
    Mengubah data transaksi mentah harian menjadi data mingguan 
    dengan zero imputation dan fitur historis secara otomatis.
    """
    
    # 1. Pastikan format tanggal
    df['Tanggal'] = pd.to_datetime(df['Tanggal'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Tanggal'])
    
    # 2. Agregasi Minggu
    weekly_df = df.groupby([
        pd.Grouper(key='Tanggal', freq='W-MON'), 
        'Detail Items', 
        'Kategori'
    ])['Total Item'].sum().reset_index()
    
    # 3. ZERO IMPUTATION
    # Menggunakan pivot table agar setiap barang punya baris di setiap minggu kalender
    pivot_df = weekly_df.pivot_table(
        index='Tanggal', 
        columns=['Detail Items', 'Kategori'], 
        values='Total Item', 
        aggfunc='sum'
    ).fillna(0)
    
    # 4. Kembalikan ke bentuk vertikal
    weekly_df = pivot_df.unstack().reset_index(name='quantity')
    
    # 5. Rename kolom
    weekly_df.rename(columns={'Tanggal': 'Date', 'Detail Items': 'item_id', 'Kategori': 'kategori', 'Total Item': 'quantity'}, inplace=True)
    
    # 6. Urutkan secara kronologis
    weekly_df = weekly_df.sort_values(['item_id', 'Date'])
    
    # 7. PEMBUATAN FITUR HISTORIS
    weekly_df['Qty_Lag_1W'] = weekly_df.groupby('item_id')['quantity'].shift(1).fillna(0)
    weekly_df['Qty_Lag_2W'] = weekly_df.groupby('item_id')['quantity'].shift(2).fillna(0)
    
    weekly_df['Qty_Roll_Mean_4W'] = weekly_df.groupby('item_id')['quantity'].transform(
        lambda x: x.shift(1).rolling(window=4, min_periods=1).mean()
    ).fillna(0)
    
    weekly_df['item_id_Avg_Sales'] = weekly_df.groupby('item_id')['quantity'].transform(
        lambda x: x.shift(1).expanding().mean()
    ).fillna(0)
    
    return weekly_df

# ==========================================
# 2. PILIHAN MODE INPUT
# ==========================================
st.subheader("⚙️ Mode Input Data")
mode_input = st.radio("Pilih cara memasukkan data historis:", ["Manual", "Upload Data Transaksi (CSV)"], horizontal=True)

# Inisialisasi Nilai Default
def_nama_barang = "Roti Coklat"
def_avg_sales = 20
def_lag_1w = 22
def_lag_2w = 18
def_roll_mean_4w = 21
def_kategori_index = 0 

kategori_list = [
    "KEBUTUHAN RUMAH TANGGA", "LAIN-LAIN", "MAKANAN INSTAN", "MINUMAN", 
    "OBAT", "PERAWATAN DIRI DAN KOSMETIK", "PERLENGKAPAN BAYI", 
    "PRODUK KHUSUS", "SEMBAKO", "SNACK"
]

if mode_input == "Upload Data Transaksi (CSV)":
    uploaded_file = st.file_uploader("📂 Upload file Data Transaksi (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Baca data mentah
            df_raw = pd.read_csv(uploaded_file)
            st.success("✅ Dataset transaksi berhasil dimuat! Sedang memproses pipeline data...")
            
            # Jalankan Pipa Pengolah Data Otomatis
            df_processed = process_raw_transaction_data(df_raw)
            
            # Buat dropdown barang dari hasil proses
            list_barang = df_processed['item_id'].unique()
            barang_terpilih = st.selectbox("🔍 Cari & Pilih Barang:", list_barang)
            
            # Ambil riwayat spesifik dari barang yang dipilih
            df_barang = df_processed[df_processed['item_id'] == barang_terpilih]
            
            # Ambil data minggu paling terakhir (baris terbawah)
            baris_terakhir = df_barang.iloc[-1]
            
            # Update Nilai Default berdasarkan perhitungan pipeline otomatis
            def_nama_barang = str(barang_terpilih)
            def_avg_sales = int(round(float(baris_terakhir['item_id_Avg_Sales'])))
            def_lag_1w = int(round(float(baris_terakhir['Qty_Lag_1W'])))
            def_lag_2w = int(round(float(baris_terakhir['Qty_Lag_2W'])))
            def_roll_mean_4w = int(round(float(baris_terakhir['Qty_Roll_Mean_4W'])))
            
            # Coba deteksi kategori otomatis (jika ada di daftar kita)
            kategori_terakhir = str(baris_terakhir['kategori']).upper()
            if kategori_terakhir in kategori_list:
                def_kategori_index = kategori_list.index(kategori_terakhir)
            
            st.info(f"📌 Fitur historis otomatis dihitung dan diisi berdasarkan minggu terakhir untuk **{def_nama_barang}**.")
            
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan pada pipeline data: Pastikan kolom 'Tanggal', 'Detail Items', 'Kategori', dan 'Total Item' ada di file CSV. Error lengkap: {e}")

st.divider()

# ==========================================
# 3. KONDISI FISIK TOKO
# ==========================================
st.subheader("📦 1. Cek Fisik Barang di Toko")
col_A, col_B, col_C = st.columns(3)

with col_A:
    nama_barang = st.text_input("📝 Nama / ID Barang:", value=def_nama_barang, help="Nama barang yang diprediksi.")

with col_B:
    stok_sekarang = st.number_input("📦 Cek Fisik Stok (pcs)", min_value=0, help="Input sisa stok barang di toko.")

with col_C:
    safety_stock = st.number_input("🛡️ Safety Stock (pcs)", min_value=0, value=3, help="Default 3 pcs.")

st.divider()

# ==========================================
# 4. PROFIL PERILAKU PENJUALAN
# ==========================================
st.subheader("📝 2. Profil Perilaku Penjualan Mingguan")

hari_ini = datetime.date.today()
col1, col2, col3 = st.columns(3)

with col1:
    target_date = st.date_input("📅 Tanggal Target Prediksi:", value=hari_ini + datetime.timedelta(days=7))
    month = target_date.month
    week_of_year = target_date.isocalendar()[1]
    
    st.info(f"📌 **Bulan ke-{month}** | **Minggu ke-{week_of_year}**")
    is_holiday = st.selectbox("🎉 Ada Libur Nasional / Event?", [0, 1], format_func=lambda x: "Ya (1)" if x==1 else "Tidak (0)")

with col2:
    kategori_input = st.selectbox("🗂️ Kategori Produk", kategori_list, index=def_kategori_index)
    item_id_avg_sales = st.number_input("📊 Rata-rata Penjualan Historis", min_value=0, value=def_avg_sales)

with col3:
    qty_lag_1w = st.number_input("📈 Laku Minggu Lalu (Lag 1W)", min_value=0, value=def_lag_1w)
    qty_lag_2w = st.number_input("📈 Laku 2 Minggu Lalu (Lag 2W)", min_value=0, value=def_lag_2w)
    qty_roll_mean_4w = st.number_input("📊 Rata-rata 4 Minggu Terakhir", min_value=0, value=def_roll_mean_4w)

# ==========================================
# 5. TOMBOL EKSEKUSI & KALKULASI SUPPLY CHAIN
# ==========================================
st.divider()
if st.button("🚀 HITUNG REKOMENDASI ORDER!", type="primary", use_container_width=True):
    
    if models is None:
        st.error("Model gagal dimuat. Proses dihentikan.")
    else:
        # --- PROSES ONE-HOT ENCODING KATEGORI ---
        kategori_dict = {f"Kategori_{k}": 0 for k in kategori_list}
        kategori_terpilih = f"Kategori_{kategori_input}"
        if kategori_terpilih in kategori_dict:
            kategori_dict[kategori_terpilih] = 1
            
        # --- MENYIAPKAN DATAFRAME ---
        input_dict = {
            'Month': month,
            'WeekOfYear': week_of_year,
            'Is_Holiday': is_holiday,
            'Qty_Lag_1W': qty_lag_1w,
            'Qty_Lag_2W': qty_lag_2w,
            'Qty_Roll_Mean_4W': qty_roll_mean_4w,
            **kategori_dict, 
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
        pred_1w = max(0, round(models["1_minggu"].predict(input_data)[0]))
        pred_2w = max(0, round(models["2_minggu"].predict(input_data)[0]))
        pred_3w = max(0, round(models["3_minggu"].predict(input_data)[0]))
            
        order_1w = max(0, (pred_1w + safety_stock) - stok_sekarang)
        order_2w = max(0, (pred_2w + safety_stock) - stok_sekarang)
        order_3w = max(0, (pred_3w + safety_stock) - stok_sekarang)
        
        # ==========================================
        # TAMPILAN HASIL
        # ==========================================
        st.subheader(f"📊 Laporan Proyeksi Penjualan & Restock: {nama_barang}")
        st.caption(f"Kondisi Saat Ini: **Safety Stock = {safety_stock} pcs** | **Sisa Stok di Toko = {stok_sekarang} pcs**")
        st.write("")
        
        res_col1, res_col2, res_col3 = st.columns(3)
        
        with res_col1:
            st.success("1️⃣ **Minggu Depan**")
            st.metric(label=f"Estimasi Laku (Minggu ke-{week_of_year})", value=f"{pred_1w} pcs")
            if order_1w > 0:
                st.metric(label="🛒 KEPUTUSAN RESTOCK", value=f"{order_1w} pcs", delta="Perlu Order", delta_color="inverse")
            else:
                st.metric(label="🛒 KEPUTUSAN RESTOCK", value=f"{order_1w} pcs", delta="Stok Aman", delta_color="off")
                
        with res_col2:
            st.info("2️⃣ **2 Minggu Depan**")
            st.metric(label="Estimasi Laku", value=f"{pred_2w} pcs")
            if order_2w > 0:
                st.metric(label="🛒 KEPUTUSAN RESTOCK", value=f"{order_2w} pcs", delta="Perlu Order", delta_color="inverse")
            else:
                st.metric(label="🛒 KEPUTUSAN RESTOCK", value=f"{order_2w} pcs", delta="Stok Aman", delta_color="off")

        with res_col3:
            st.warning("3️⃣ **3 Minggu Depan**")
            st.metric(label="Estimasi Laku", value=f"{pred_3w} pcs")
            if order_3w > 0:
                st.metric(label="🛒 KEPUTUSAN RESTOCK", value=f"{order_3w} pcs", delta="Perlu Order", delta_color="inverse")
            else:
                st.metric(label="🛒 KEPUTUSAN RESTOCK", value=f"{order_3w} pcs", delta="Stok Aman", delta_color="off")

        st.divider()