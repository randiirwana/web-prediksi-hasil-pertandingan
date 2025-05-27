import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib # Import joblib
import os # Import os to create directory
import pickle # Import pickle

# Load dataset
df = pd.read_csv("soccer21-22.csv")

# Pisahkan fitur dan label
# Menggunakan FTR (Full Time Result) sebagai label
X = df.drop(['Date', 'HomeTeam', 'AwayTeam', 'FTR', 'Referee', 'HTR'], axis=1)  # Hapus HTR juga
y = df['FTR'].map({'H': 1, 'D': 0, 'A': 2})  # H=Home Win, D=Draw, A=Away Win

# Normalisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model backpropagation
model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(3, activation='softmax')  # 3 kelas: Home Win, Draw, Away Win
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Latih model dan simpan riwayat pelatihan
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, callbacks=[early_stop])

# Evaluasi
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Akurasi Model: {accuracy:.4f}")

# Prediksi
y_pred = np.argmax(model.predict(X_test), axis=1)
print(classification_report(y_test, y_pred, target_names=['Draw', 'Home Win', 'Away Win']))

# Simpan model, scaler, dan riwayat pelatihan
model_dir = "model"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model.save(os.path.join(model_dir, 'soccer_predictor_model.h5'))
joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))

history_path = os.path.join(model_dir, 'training_history.pkl')
with open(history_path, 'wb') as f:
    pickle.dump(history.history, f)

print("Model, scaler, dan riwayat pelatihan berhasil disimpan di folder 'model'")
