import cv2
import tensorflow as tf
import numpy as np
import os
import pandas as pd
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.simpledialog import askstring
from tkinter import filedialog
from PIL import Image
import time

# =====================================================
# 1️⃣ LOAD MODEL & LABEL
# =====================================================
print("[INFO] Memuat model CNN...")
model = tf.keras.models.load_model("face_cnn_model.h5")

class_indices = np.load("class_indices.npy", allow_pickle=True).item()
class_names = {v: k for k, v in class_indices.items()}
print("[INFO] Label mapping:", class_names)

# =====================================================
# 2️⃣ FILE ABSENSI
# =====================================================
absen_file = "absensi.xlsx"
if not os.path.exists(absen_file):
    pd.DataFrame(columns=["Nama", "Waktu"]).to_excel(absen_file, index=False)

last_attendance = {}
ATTENDANCE_COOLDOWN = 30  

def catat_absensi(nama):
    now = time.time()
    if nama in last_attendance and now - last_attendance[nama] < ATTENDANCE_COOLDOWN:
        return

    last_attendance[nama] = now
    waktu = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    df = pd.read_excel(absen_file)
    df = pd.concat(
        [df, pd.DataFrame([[nama, waktu]], columns=["Nama", "Waktu"])],
        ignore_index=True
    )
    df.to_excel(absen_file, index=False)
    print(f"[✔] Absensi: {nama} - {waktu}")

# =====================================================
# 3️⃣ KAMERA
# =====================================================
def get_camera():
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"[INFO] Kamera index {i} digunakan")
            return cap
    return None

# =====================================================
# 4️⃣ MULAI ABSENSI
# =====================================================
def mulai_absensi():
    cap = get_camera()
    if cap is None:
        messagebox.showerror("Error", "Kamera tidak ditemukan")
        return

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    messagebox.showinfo("Info", "Tekan 'q' untuk menghentikan absensi")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]

            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (150, 150))
            face = np.expand_dims(face, axis=0) / 255.0

            pred = model.predict(face, verbose=0)
            idx = np.argmax(pred)
            conf = pred[0][idx]

            if conf >= 0.6:
                nama = class_names[idx]
                catat_absensi(nama)
                label = f"{nama} ({conf*100:.1f}%)"
                color = (0, 255, 0)
            else:
                label = "Tidak Dikenal"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Absensi Wajah Otomatis", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# =====================================================
# 5️⃣ LIHAT ABSENSI
# =====================================================
def lihat_absensi():
    df = pd.read_excel(absen_file)
    if df.empty:
        messagebox.showinfo("Info", "Belum ada data absensi")
        return

    top = tk.Toplevel(window)
    top.title("Data Absensi")
    top.geometry("500x350")

    frame = tk.Frame(top)
    frame.pack(fill=tk.BOTH, expand=True)
    
    scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    tree = ttk.Treeview(
        frame,
        columns=("Nama", "Waktu"),
        show="headings",
        yscrollcommand=scrollbar.set
    )

    scrollbar.config(command=tree.yview)

    tree.heading("Nama", text="Nama")
    tree.heading("Waktu", text="Waktu")
    tree.column("Nama", width=150)
    tree.column("Waktu", width=300)

    tree.pack(fill=tk.BOTH, expand=True)

    for _, row in df.iterrows():
        tree.insert("", tk.END, values=(row["Nama"], row["Waktu"]))


# =====================================================
# 6️⃣ MANAJEMEN DATASET
# =====================================================
def get_students():
    base = "dataset/train"
    os.makedirs(base, exist_ok=True)
    return sorted([
        d for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d))
    ])

def tambah_mahasiswa():
    nama = askstring("Tambah Mahasiswa", "Masukkan nama:")
    if not nama:
        return

    os.makedirs(f"dataset/train/{nama}", exist_ok=True)
    os.makedirs(f"dataset/test/{nama}", exist_ok=True)
    refresh_dataset_window()

def tambah_foto(nama, mode):
    files = filedialog.askopenfilenames(
        title="Pilih Foto Wajah",
        filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
    )
    if not files:
        return

    save_path = f"dataset/{mode}/{nama}"
    for f in files:
        img = Image.open(f).convert("RGB")
        img = img.resize((150, 150))
        img.save(os.path.join(save_path, os.path.basename(f)))

    messagebox.showinfo("Sukses", f"{len(files)} foto ditambahkan ke {mode}/{nama}")

def buka_dataset():
    global dataset_window
    dataset_window = tk.Toplevel(window)
    dataset_window.title("Manajemen Dataset")
    dataset_window.geometry("500x400")

    tk.Button(dataset_window, text="➕ Tambah Mahasiswa",
              command=tambah_mahasiswa).pack(pady=10)

    for nama in get_students():
        frame = tk.LabelFrame(dataset_window, text=nama)
        frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Button(frame, text="Tambah Train",
                  command=lambda n=nama: tambah_foto(n, "train")).pack(side=tk.LEFT, padx=5)
        tk.Button(frame, text="Tambah Test",
                  command=lambda n=nama: tambah_foto(n, "test")).pack(side=tk.LEFT, padx=5)

def refresh_dataset_window():
    dataset_window.destroy()
    buka_dataset()

# =====================================================
# 7️⃣ GUI UTAMA
# =====================================================
window = tk.Tk()
window.title("Absensi Wajah Otomatis")
window.geometry("420x360")
window.configure(bg="#1E1E1E")

tk.Label(
    window,
    text="📸 Absensi Wajah Otomatis",
    fg="white",
    bg="#1E1E1E",
    font=("Arial", 16, "bold")
).pack(pady=20)

tk.Button(
    window,
    text="▶️ Mulai Absensi",
    bg="#4CAF50",
    fg="white",
    width=25,
    height=2,
    command=mulai_absensi
).pack(pady=10)

tk.Button(
    window,
    text="📋 Lihat Data Absensi",
    bg="#2196F3",
    fg="white",
    width=25,
    height=2,
    command=lihat_absensi
).pack(pady=10)

tk.Button(
    window,
    text="🗂️ Manajemen Dataset",
    bg="#FF9800",
    fg="white",
    width=25,
    height=2,
    command=buka_dataset
).pack(pady=10)

tk.Button(
    window,
    text="❌ Keluar",
    bg="#F44336",
    fg="white",
    width=25,
    height=2,
    command=window.destroy
).pack(pady=10)

window.mainloop()
