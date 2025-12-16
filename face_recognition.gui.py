import cv2
import tensorflow as tf
import numpy as np
import os
import pandas as pd
from datetime import datetime
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from tkinter.simpledialog import askstring
from tkinter import filedialog
from PIL import Image
import shutil
import time


# =====================================================
# 1️⃣ LOAD MODEL DAN LABEL
# =====================================================
print("[INFO] Memuat model CNN...")
model = tf.keras.models.load_model('face_cnn_model.h5')
class_names = sorted(os.listdir('dataset/train'))

# =====================================================
# 2️⃣ FILE ABSENSI OTOMATIS
# =====================================================
absen_file = 'absensi.xlsx'

if not os.path.exists(absen_file):
    df = pd.DataFrame(columns=['Nama', 'Waktu'])
    df.to_excel(absen_file, index=False)

def catat_absensi(nama):
    waktu = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    df = pd.read_excel(absen_file)

    if nama not in df['Nama'].values:
        new_data = pd.DataFrame([[nama, waktu]], columns=['Nama', 'Waktu'])
        df = pd.concat([df, new_data], ignore_index=True)
        df.to_excel(absen_file, index=False)

        print(f"[✔] {nama} tercatat pada {waktu}")

# =====================================================
# 3️⃣ DETEKSI KAMERA OTOMATIS
# =====================================================
def get_available_camera():
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"[INFO] Kamera index {i} digunakan.")
            return cap
    print("[ERROR] Tidak ada kamera terdeteksi.")
    return None

# =====================================================
# 4️⃣ FUNGSI UTAMA: MULAI ABSENSI
# =====================================================
def mulai_absensi():
    cap = get_available_camera()
    if cap is None:
        messagebox.showerror("Error", "Kamera tidak terdeteksi!")
        return

    cap.set(3, 640)
    cap.set(4, 480)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    messagebox.showinfo("Info", "Tekan 'q' untuk menghentikan absensi.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Kamera tidak aktif.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (150, 150))
            face_roi = np.expand_dims(face_roi, axis=0) / 255.0

            predictions = model.predict(face_roi)
            class_index = np.argmax(predictions)
            confidence = np.max(predictions)

            if confidence > 0.8:
                nama = class_names[class_index]
                catat_absensi(nama)
                label = f"{nama} ({confidence*100:.1f}%)"
                color = (0, 255, 0)
            else:
                label = "Tidak Dikenal"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow('Face Recognition - Absensi Otomatis', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# =====================================================
# 5️⃣ LIHAT DATA ABSENSI
# =====================================================
def lihat_absensi():
    if not os.path.exists(absen_file):
        messagebox.showinfo("Info", "Belum ada data absensi.")
        return

    df = pd.read_excel(absen_file)
    if df.empty:
        messagebox.showinfo("Info", "Belum ada nama yang tercatat.")
        return

    top = tk.Toplevel(window)
    top.title("Data Absensi")
    top.geometry("400x300")

    tree = ttk.Treeview(top, columns=("Nama", "Waktu"), show='headings')
    tree.heading("Nama", text="Nama")
    tree.heading("Waktu", text="Waktu")
    tree.pack(fill=tk.BOTH, expand=True)

    for _, row in df.iterrows():
        tree.insert("", tk.END, values=(row["Nama"], row["Waktu"]))

        def tambah_mahasiswa():
            nama = askstring("Tambah Mahasiswa", "Masukkan Nama Mahasiswa:")
            if not nama:
                return

            # Folder dataset
            train_path = f"dataset/train/{nama}"
            test_path = f"dataset/test/{nama}"
            os.makedirs(train_path, exist_ok=True)
            os.makedirs(test_path, exist_ok=True)

            cap = get_available_camera()
            if cap is None:
                messagebox.showerror("Error", "Kamera tidak tersedia!")
                return

            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )

            messagebox.showinfo(
                "Info",
                "Wajah akan otomatis terdeteksi setiap detik\nTekan 'q' untuk selesai"
            )

            count = 0
            last_capture_time = time.time()
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                current_time = time.time()
                if len(faces) > 0 and (current_time - last_capture_time) >= 1.0:
                    # Capture the first detected face
                    (x, y, w, h) = faces[0]
                    face = frame[y:y + h, x:x + w]
                    face = cv2.resize(face, (150, 150))

                    if count < 20:
                        cv2.imwrite(f"{train_path}/{count}.jpg", face)
                    else:
                        cv2.imwrite(f"{test_path}/{count}.jpg", face)

                    count += 1
                    print(f"[INFO] Capture {count}")
                    last_capture_time = current_time

                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                cv2.imshow("Tambah Data Mahasiswa", frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or count >= 30:
                    break

            cap.release()
            cv2.destroyAllWindows()

            messagebox.showinfo(
                "Sukses",
                f"Data wajah {nama} berhasil disimpan!\nSilakan retrain model."
            )

def get_list_mahasiswa():
    base_path = "dataset/train"
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    return sorted([
        d for d in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, d))
    ])

def tambah_mahasiswa():
    nama = askstring("Tambah Mahasiswa", "Masukkan Nama Mahasiswa:")
    if not nama:
        return

    os.makedirs(f"dataset/train/{nama}", exist_ok=True)
    os.makedirs(f"dataset/test/{nama}", exist_ok=True)

    messagebox.showinfo("Sukses", f"Mahasiswa {nama} berhasil ditambahkan")
    refresh_dataset_window()

def tambah_foto(nama, mode):
    files = filedialog.askopenfilenames(
        title="Pilih Foto Wajah",
        filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
    )

    if not files:
        return

    save_path = f"dataset/{mode}/{nama}"

    for file in files:
        img = Image.open(file).convert("RGB")
        img = img.resize((150, 150))

        filename = os.path.basename(file)
        img.save(os.path.join(save_path, filename))

    messagebox.showinfo(
        "Sukses",
        f"{len(files)} foto berhasil ditambahkan ke {mode}/{nama}"
    )

def buka_manajemen_dataset():
    global dataset_window
    dataset_window = tk.Toplevel(window)
    dataset_window.title("Manajemen Dataset Mahasiswa")
    dataset_window.geometry("500x400")

    frame_top = tk.Frame(dataset_window)
    frame_top.pack(pady=10)

    btn_add = tk.Button(
        frame_top,
        text="➕ Tambah Mahasiswa",
        bg="#4CAF50",
        fg="white",
        font=("Arial", 10, "bold"),
        command=tambah_mahasiswa
    )
    btn_add.pack()

    frame_list = tk.Frame(dataset_window)
    frame_list.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    for nama in get_list_mahasiswa():
        card = tk.LabelFrame(frame_list, text=nama, padx=10, pady=5)
        card.pack(fill=tk.X, pady=5)

        btn_train = tk.Button(
            card,
            text="Tambah Foto Train",
            command=lambda n=nama: tambah_foto(n, "train")
        )
        btn_train.pack(side=tk.LEFT, padx=5)

        btn_test = tk.Button(
            card,
            text="Tambah Foto Test",
            command=lambda n=nama: tambah_foto(n, "test")
        )
        btn_test.pack(side=tk.LEFT, padx=5)

def refresh_dataset_window():
    dataset_window.destroy()
    buka_manajemen_dataset()


# =====================================================
# 6️⃣ GUI UTAMA TKINTER
# =====================================================
window = tk.Tk()
window.title("Face Recognition - Absensi Otomatis")
window.geometry("400x350")
window.configure(bg="#1E1E1E")

title_label = tk.Label(window, text="📸 Absensi Wajah Otomatis", fg="white", bg="#1E1E1E", font=("Arial", 16, "bold"))
title_label.pack(pady=20)

btn_mulai = tk.Button(window, text="▶️  Mulai Absensi", bg="#4CAF50", fg="white", font=("Arial", 12, "bold"), width=25, height=2, command=mulai_absensi)
btn_mulai.pack(pady=10)

btn_lihat = tk.Button(window, text="📋  Lihat Data Absensi", bg="#2196F3", fg="white", font=("Arial", 12, "bold"), width=25, height=2, command=lihat_absensi)
btn_lihat.pack(pady=10)

btn_keluar = tk.Button(window, text="❌  Keluar", bg="#F44336", fg="white", font=("Arial", 12, "bold"), width=25, height=2, command=window.destroy)
btn_keluar.pack(pady=10)

btn_dataset = tk.Button(
    window,
    text="🗂️  Manajemen Dataset",
    bg="#FF9800",
    fg="white",
    font=("Arial", 12, "bold"),
    width=25,
    height=2,
    command=buka_manajemen_dataset
)
btn_dataset.pack(pady=10)


credit = tk.Label(window, text="Dibuat oleh: Kelompok 5", bg="#1E1E1E", fg="#AAAAAA", font=("Arial", 9))
credit.pack(side=tk.BOTTOM, pady=10)

window.mainloop()
