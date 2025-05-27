import matplotlib.pyplot as plt
import pickle
import os

# Path ke file riwayat pelatihan
HISTORY_PATH = os.path.join('model', 'training_history.pkl')

def create_loss_plot(history_data, output_path):
    """Membuat dan menyimpan plot Training/Validation Loss."""
    plt.figure(figsize=(8,5))
    plt.plot(history_data['loss'], label='Training Loss', color='blue')
    plt.plot(history_data['val_loss'], label='Validation Loss', color='orange')
    plt.title('📈 Grafik Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Simpan plot ke file dalam format PNG
    plt.savefig(output_path, format='png')
    plt.close() # Tutup figure untuk menghemat memori

def create_accuracy_plot(history_data, output_path):
    """Membuat dan menyimpan plot Training/Validation Accuracy."""
    plt.figure(figsize=(8,5))
    plt.plot(history_data['accuracy'], label='Training Accuracy', color='green')
    plt.plot(history_data['val_accuracy'], label='Validation Accuracy', color='red')
    plt.title('📊 Grafik Akurasi per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Simpan plot ke file dalam format PNG
    plt.savefig(output_path, format='png')
    plt.close() # Tutup figure

# --- Bagian ini hanya akan berjalan jika visualisasi.py dijalankan langsung ---
if __name__ == "__main__":
    try:
        with open(HISTORY_PATH, 'rb') as f:
            history_data = pickle.load(f)
        print("Riwayat pelatihan berhasil dimuat untuk visualisasi.")

        # Ubah direktori output dari 'static' ke direktori saat ini
        output_dir = '.' # Direktori saat ini
        # Tidak perlu membuat direktori '.' jika sudah ada
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)

        loss_plot_path = os.path.join(output_dir, 'loss_plot.png')
        accuracy_plot_path = os.path.join(output_dir, 'accuracy_plot.png')

        create_loss_plot(history_data, loss_plot_path)
        create_accuracy_plot(history_data, accuracy_plot_path)

        print(f"Plot loss disimpan di: {loss_plot_path}")
        print(f"Plot akurasi disimpan di: {accuracy_plot_path}")

    except FileNotFoundError:
        print(f"Error: File riwayat pelatihan tidak ditemukan di {HISTORY_PATH}.")
        print("Pastikan Anda sudah menjalankan skrip model.py untuk melatih dan menyimpan riwayat.")
    except Exception as e:
        print(f"Error memuat riwayat pelatihan atau membuat plot: {e}")
