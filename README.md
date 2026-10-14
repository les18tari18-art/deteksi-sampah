# Sistem Pendeteksi Sampah Organik & Anorganik

## Langkah
1. Download dataset: Kaggle "Waste Classification Data" -> ekstrak jadi folder `DATASET/` (berisi TRAIN & TEST, masing-masing berisi O dan R).
2. `pip install -r requirements.txt`
3. `python train_model.py`  (disarankan di Google Colab dengan GPU; hasilnya waste_model.keras + class_names.json)
4. `streamlit run app.py`
