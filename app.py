from flask import Flask, render_template, request, url_for, send_from_directory, session, redirect
import numpy as np
import joblib
import tensorflow as tf
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
#from visualisasi import create_loss_plot, create_accuracy_plot # Tidak perlu lagi jika plotting di app.py
#import matplotlib.pyplot as plt # Tidak perlu matplotlib jika pakai plotly
import plotly # Import plotly
import plotly.graph_objs as go # Import objek grafik Plotly
import json # Import json

# Inisialisasi aplikasi Flask
app = Flask(__name__)
# Konfigurasi sesi
app.config['SECRET_KEY'] = os.urandom(24) # Ganti dengan kunci rahasia yang sebenarnya di produksi

# Path ke model dan scaler
MODEL_PATH = os.path.join('model', 'soccer_predictor_model.h5')
SCALER_PATH = os.path.join('model', 'scaler.pkl')
DATASET_PATH = 'soccer21-22.csv'
HISTORY_PATH = os.path.join('model', 'training_history.pkl')

# Variabel global untuk model, scaler, data test, dan history
model = None
scaler = None
X_test_scaled = None
y_test = None
history_data = None

# Muat sumber daya (model, scaler, data test, history) saat aplikasi dimulai
@app.before_request
def load_resources():
    global model, scaler, X_test_scaled, y_test, history_data
    # Hanya muat jika belum dimuat dan ini bukan permintaan untuk static files
    if request.endpoint != 'static' and (model is None or scaler is None or X_test_scaled is None or y_test is None or history_data is None):
        try:
            # Muat model dan scaler
            if os.path.exists(MODEL_PATH):
                 model = tf.keras.models.load_model(MODEL_PATH)
                 print("Model berhasil dimuat!")
            else:
                 print(f"Peringatan: File model tidak ditemukan di {MODEL_PATH}")

            if os.path.exists(SCALER_PATH):
                 scaler = joblib.load(SCALER_PATH)
                 print("Scaler berhasil dimuat!")
            else:
                 print(f"Peringatan: File scaler tidak ditemukan di {SCALER_PATH}")

            # Load dataset dan siapkan data test jika model dan scaler berhasil dimuat
            if model is not None and scaler is not None and os.path.exists(DATASET_PATH):
                try:
                    df = pd.read_csv(DATASET_PATH)
                    X = df.drop(['Date', 'HomeTeam', 'AwayTeam', 'FTR', 'Referee', 'HTR'], axis=1)
                    y = df['FTR'].map({'H': 1, 'D': 0, 'A': 2})
                    # Hindari error jika X kosong setelah drop
                    if not X.empty:
                         X_scaled = scaler.transform(X)
                         _, X_test_scaled, _, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
                         print("Data uji berhasil disiapkan!")
                    else:
                         print("Peringatan: Dataset kosong setelah menghapus kolom.")

                except FileNotFoundError:
                    print(f"Error: File dataset tidak ditemukan di {DATASET_PATH}.")
                except Exception as e:
                    print(f"Error menyiapkan data uji: {e}")
            elif not os.path.exists(DATASET_PATH):
                 print(f"Peringatan: File dataset tidak ditemukan di {DATASET_PATH}")


            # Muat riwayat pelatihan
            if os.path.exists(HISTORY_PATH):
                 try:
                      with open(HISTORY_PATH, 'rb') as f:
                         history_data = pickle.load(f)
                      print("Riwayat pelatihan berhasil dimuat!")
                 except Exception as e:
                      print(f"Error memuat riwayat pelatihan dari {HISTORY_PATH}: {e}")
            else:
                 print(f"Peringatan: File riwayat pelatihan tidak ditemukan di {HISTORY_PATH}")

        except Exception as e:
            print(f"Error memuat sumber daya (model, scaler, dll): {e}")

# Halaman utama (form input)
@app.route('/')
def index():
    # Hapus data prediksi dari sesi saat kembali ke form
    session.pop('prediction_results', None)
    print("Sesi prediksi dihapus.")
    return render_template('index.html')

# Halaman hasil prediksi
@app.route('/predict', methods=['POST', 'GET']) # Terima juga metode GET
def predict():
    if model is None or scaler is None:
        print("Model atau scaler tidak dimuat.")
        return "Error: Model atau scaler tidak dapat dimuat."

    try:
        feature_names = ['FTHG', 'FTAG', 'HTHG', 'HTAG', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']
        
        if request.method == 'POST':
            # Ambil data dari form saat POST
            features = [float(request.form[name]) for name in feature_names]
            print("Data fitur diambil dari form (POST).")
            features_array = np.array(features).reshape(1, -1)
            features_scaled = scaler.transform(features_array)
            prediction_proba = model.predict(features_scaled)[0]
            predicted_class = np.argmax(prediction_proba)
            results_map = {0: 'Draw', 1: 'Home Win', 2: 'Away Win'}
            prediction_text = results_map.get(predicted_class, 'Hasil tidak diketahui')

            proba_data = {
                'Draw': float(prediction_proba[0]),
                'Home Win': float(prediction_proba[1]),
                'Away Win': float(prediction_proba[2])
            }

            # --- Buat Plotly Bar Chart Probabilitas ---
            labels = list(proba_data.keys())
            values = list(proba_data.values())
            
            fig = go.Figure(data=[go.Bar(x=labels, y=values, marker_color=['gray', 'blue', 'red'])])
            fig.update_layout(
                title='Probabilitas Hasil Prediksi',
                yaxis=dict(title='Probabilitas', range=[0, 1])
            )
            proba_plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            # ----------------------------------------

            # Simpan hasil prediksi di sesi
            session['prediction_results'] = {
                'prediction_text': prediction_text,
                'probabilities': proba_data,
                'proba_plot_json': proba_plot_json
            }
            print("Hasil prediksi disimpan di sesi.")

            # Tampilkan halaman hasil
            return render_template('result.html', 
                                   prediction=prediction_text,
                                   probabilities=proba_data,
                                   proba_plot_json=proba_plot_json)
            
        elif request.method == 'GET':
            # Ambil hasil prediksi dari sesi saat GET (dari tombol kembali)
            results = session.get('prediction_results')
            print(f"Mencoba mengambil hasil prediksi dari sesi (GET). Hasil: {'Ada' if results else 'Tidak Ada'}")
            
            if results is None:
                # Jika tidak ada data di sesi, redirect ke halaman form
                print("Tidak ada hasil prediksi di sesi, redirect ke index.")
                return redirect(url_for('index'))
            
            # Tampilkan halaman hasil menggunakan data dari sesi
            return render_template('result.html', 
                                   prediction=results['prediction_text'],
                                   probabilities=results['probabilities'],
                                   proba_plot_json=results['proba_plot_json'])
            
    except Exception as e:
        # Jika ada error, redirect ke form dan bersihkan sesi prediksi
        print(f"Error saat memproses prediksi (GET/POST): {e}")
        session.pop('prediction_results', None) # Hapus data sesi prediksi jika error
        return redirect(url_for('index'))

# Halaman Evaluasi Model
@app.route('/evaluation')
def evaluation():
    global model, X_test_scaled, y_test
    if model is None or X_test_scaled is None or y_test is None:
         print("Model atau data uji tidak dimuat untuk evaluasi.")
         return "Error: Model atau data uji tidak dapat dimuat untuk evaluasi."

    try:
        predictions = model.predict(X_test_scaled)
        y_pred = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Draw', 'Home Win', 'Away Win'])
        
        return render_template('evaluation.html', accuracy=accuracy, report=report)

    except Exception as e:
        print(f"Error saat melakukan evaluasi: {e}")
        return "Error saat melakukan evaluasi."

# Halaman Visualisasi Grafik Pelatihan
@app.route('/visualization')
def visualization():
    global history_data
    if history_data is None:
        print("Riwayat pelatihan tidak dimuat untuk visualisasi.")
        return "Error: Riwayat pelatihan tidak dapat dimuat untuk visualisasi."

    try:
        # --- Buat Plotly Line Charts Loss dan Akurasi ---
        epochs = list(range(1, len(history_data['loss']) + 1))

        # Plot Loss
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(x=epochs, y=history_data['loss'], mode='lines', name='Training Loss', line=dict(color='blue')))
        fig_loss.add_trace(go.Scatter(x=epochs, y=history_data['val_loss'], mode='lines', name='Validation Loss', line=dict(color='orange')))
        fig_loss.update_layout(
            title='📈 Grafik Loss per Epoch',
            xaxis_title='Epoch',
            yaxis_title='Loss'
        )
        loss_plot_json = json.dumps(fig_loss, cls=plotly.utils.PlotlyJSONEncoder)

        # Plot Akurasi
        fig_accuracy = go.Figure()
        fig_accuracy.add_trace(go.Scatter(x=epochs, y=history_data['accuracy'], mode='lines', name='Training Accuracy', line=dict(color='green')))
        fig_accuracy.add_trace(go.Scatter(x=epochs, y=history_data['val_accuracy'], mode='lines', name='Validation Accuracy', line=dict(color='red')))
        fig_accuracy.update_layout(
            title='📊 Grafik Akurasi per Epoch',
            xaxis_title='Epoch',
            yaxis_title='Accuracy'
        )
        accuracy_plot_json = json.dumps(fig_accuracy, cls=plotly.utils.PlotlyJSONEncoder)
        # ------------------------------------------------
        
        return render_template('visualization.html', 
                               loss_plot_json=loss_plot_json,
                               accuracy_plot_json=accuracy_plot_json)

    except Exception as e:
        print(f"Error saat membuat visualisasi: {e}")
        return "Error saat membuat visualisasi."

# Jalankan aplikasi Flask
if __name__ == '__main__':
    # Pastikan Anda memiliki folder 'model' dengan file-file yang diperlukan sebelum menjalankan aplikasi
    # Peringatan ini akan muncul jika file tidak ada saat pertama kali memuat sumber daya
    app.run(debug=True) 