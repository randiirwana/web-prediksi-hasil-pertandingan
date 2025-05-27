import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import tensorflow as tf
import os

# Path ke model dan scaler yang sudah dilatih
MODEL_PATH = os.path.join('model', 'soccer_predictor_model.h5')
SCALER_PATH = os.path.join('model', 'scaler.pkl')

# Muat model dan scaler
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model dan scaler berhasil dimuat!")
except Exception as e:
    print(f"Error memuat model atau scaler: {e}")
    model = None
    scaler = None
    exit() # Keluar jika model/scaler tidak bisa dimuat

# Load dataset
try:
    df = pd.read_csv("soccer21-22.csv")
except FileNotFoundError:
    print("Error: File 'soccer21-22.csv' tidak ditemukan.")
    exit()

# Pisahkan fitur dan label
# Menggunakan FTR (Full Time Result) sebagai label
X = df.drop(['Date', 'HomeTeam', 'AwayTeam', 'FTR', 'Referee', 'HTR'], axis=1)  # Hapus HTR juga
y = df['FTR'].map({'H': 1, 'D': 0, 'A': 2})  # H=Home Win, D=Draw, A=Away Win

# Lakukan scaling pada seluruh data fitur sebelum split (menggunakan scaler yang sudah dilatih)
X_scaled = scaler.transform(X)

# Split data untuk mendapatkan data uji (harus sama dengan split saat training)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Lakukan prediksi pada data uji menggunakan model yang dimuat
if model is not None:
    predictions = model.predict(X_test)
    y_pred = np.argmax(predictions, axis=1)

    # Hitung metrik evaluasi klasifikasi
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Akurasi Model pada data uji: {accuracy:.4f}")

    # Tampilkan classification report
    # Sesuaikan target_names sesuai mapping label Anda (0: Draw, 1: Home Win, 2: Away Win)
    target_names = ['Draw', 'Home Win', 'Away Win']
    print("Classification Report pada data uji:")
    print(classification_report(y_test, y_pred, target_names=target_names))
else:
    print("Tidak dapat melakukan evaluasi karena model tidak termuat.")
