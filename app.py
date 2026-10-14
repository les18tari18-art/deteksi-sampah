"""
Aplikasi web deteksi sampah Organik vs Anorganik.
Jalankan:  streamlit run app.py
Butuh file waste_model.keras dan class_names.json (hasil train_model.py).
"""
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Deteksi Sampah", page_icon="♻️")

LABEL = {  # sesuaikan dengan nama folder dataset
    "O": ("Sampah ORGANIK", "🍃", "Bisa dijadikan kompos. Buang ke tempat sampah hijau."),
    "R": ("Sampah ANORGANIK", "♻️", "Bisa didaur ulang. Buang ke tempat sampah kuning/biru."),
}

@st.cache_resource
def load():
    model = tf.keras.models.load_model("waste_model.keras")
    with open("class_names.json") as f:
        names = json.load(f)
    return model, names

model, class_names = load()

st.title("♻️ Sistem Pendeteksi Sampah")
st.write("Unggah foto atau gunakan kamera untuk mengetahui jenis sampah: **organik** atau **anorganik**.")

tab1, tab2 = st.tabs(["📁 Upload Foto", "📷 Kamera"])
with tab1:
    up = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png"])
with tab2:
    cam = st.camera_input("Ambil foto")

file = up or cam
if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Gambar input", use_container_width=True)

    arr = np.expand_dims(np.array(img.resize((224, 224)), dtype="float32"), 0)
    p = float(model.predict(arr, verbose=0)[0][0])  # probabilitas kelas index 1
    idx = int(p > 0.5)
    conf = p if idx == 1 else 1 - p
    nama, ikon, saran = LABEL.get(class_names[idx], (class_names[idx], "", ""))

    st.subheader(f"{ikon} {nama}")
    st.progress(conf, text=f"Tingkat keyakinan: {conf*100:.1f}%")
    st.info(saran)
    if conf < 0.7:
        st.warning("Keyakinan rendah. Coba foto dengan pencahayaan lebih baik dan objek lebih jelas.")
