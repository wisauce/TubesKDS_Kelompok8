# Analisis Siklus Sel dengan Metode Hybrid Deep Learning dan Biological Validation

Proyek ini merupakan sistem analisis siklus sel tiga lapis yang mengintegrasikan *deep learning* (CNN) dengan validasi biologis berbasis model biofisika (ODE) dan aturan *checkpoint*. Sistem ini dikembangkan untuk mengklasifikasikan fase sel dari citra mikroskopi fluoresensi dan mendeteksi anomali pada siklus sel yang berpotensi memiliki signifikansi klinis (misalnya, identifikasi awal disregulasi seluler atau kanker).

## 👥 Tim Pengembang (Kelompok 8)
- Wisa Ahmaduta Dinutama (18223003)
- Stanislaus Ardy Bramantyo (18223057)
- Sendi Putra Alicia (18223063)
- Ghazy Achmed M. Urbayani (18223093)

Institut Teknologi Bandung

## 🏗️ Arsitektur Sistem

Sistem ini terdiri dari tiga *layer* utama:
1. **Layer 1 (CNN Phase Classifier):** Menggunakan model ResNet-18 yang di-*finetune* pada dataset BBBC048 (citra fluoresensi sel Jurkat dengan pewarnaan Hoechst 33342) untuk mengklasifikasikan 7 fase siklus sel (G1, S, G2, Prophase, Metaphase, Anaphase, Telophase). Dilengkapi dengan **Grad-CAM** untuk *explainability*.
2. **Layer 2 (Validasi Temporal Biologis):** Menggunakan Ordinary Differential Equation (ODE) Tyson-Novak untuk memodelkan dinamika Cyclin-CDK, yang mana dari solusinya diturunkan sebuah matriks transisi untuk Hidden Markov Model (HMM). HMM ini berfungsi mengoreksi prediksi CNN agar secara temporal (sekuensial) valid dan taat pada aturan biologi.
3. **Layer 3 (Detektor Anomali Checkpoint):** Menerapkan aturan biologi (G1/S Restriction, G2/M DNA Damage, Spindle Assembly) untuk mendeteksi secara otomatis anomali biologis nyata dibandingkan dengan sekadar artefak salah klasifikasi.

## 🚀 Fitur Utama
- **Inferensi Model:** Klasifikasi gambar *single-cell* ke fase siklus sel.
- **Explainable AI:** Visualisasi region fokus model menggunakan Grad-CAM.
- **Koreksi HMM-ODE:** Pembetulan sekuens klasifikasi *time-lapse* agar rasional secara biologis.
- **Analisis Populasi:** Estimasi *Mitotic Index*, *Growth Fraction*, dan distribusi fase.
- **Web Dashboard Interaktif:** Menyatukan keseluruhan alur evaluasi di antarmuka web yang rapi (menggunakan Flask).

## 👨‍🏫 Flow Tes untuk Asisten

buka Terminal

### Cara 1: Lewat Python Virtual Environment

```bash
git clone https://github.com/wisauce/TubesKDS_Kelompok8.git
cd TubesKDS_Kelompok8
python3 -m venv .venv
source .venv/bin/activate  # (atau .venv\Scripts\activate kalau asdos pake Windows)
pip install -r TubesKDS2/requirements.txt
python3 TubesKDS2/api/app.py
```

### Cara 2: lewat Docker

```bash
git clone https://github.com/wisauce/TubesKDS_Kelompok8.git
cd TubesKDS_Kelompok8
docker build -t cell-cycle-app -f TubesKDS2/Dockerfile .
docker run -p 5050:5050 cell-cycle-app
```

setelah asdos jalankan salah 1 dari 2 cara di atas, bisa langsung buka http://localhost:5050 di browser dan BOOM, web dashboard langsung menyala menggunakan model

## 📄 Laporan dan Makalah
Detail implementasi metodologi, metrik evaluasi model CNN, dan hasil analisis lengkap dapat ditemukan pada laporan berformat LaTeX IEEEtran yang terdapat pada direktori `TubesKDS2/laporan/main.pdf`.

---
*Dibuat untuk Tugas Tubes KDS*