import cv2

def test_camera():
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("[INFO] Kamera laptop (index 0) terdeteksi dan berfungsi.")
        ret, frame = cap.read()
        if ret:
            print("[INFO] Frame berhasil dibaca dari kamera.")
        else:
            print("[ERROR] Tidak dapat membaca frame dari kamera.")
        cap.release()
    else:
        print("[ERROR] Kamera laptop tidak terdeteksi. Periksa pengaturan kamera di laptop Anda.")
        for i in range(1, 5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"[INFO] Kamera ditemukan di index {i}.")
                cap.release()
                break
        else:
            print("[ERROR] Tidak ada kamera yang terdeteksi di index 0-4.")

if __name__ == "__main__":
    test_camera()
