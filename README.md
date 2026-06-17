# 📋 Sistem Peringkasan Ekstraktif Notulensi — LPP TVRI

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B.svg)
![NLP](https://img.shields.io/badge/NLP-Sastrawi-green.svg)
![PostgreSQL](https://img.shields.io/badge/Database-PostgreSQL-336791.svg)

Sebuah aplikasi berbasis web (Streamlit) berskala riset untuk melakukan peringkasan teks ekstraktif secara otomatis dari dokumen notulensi rapat Forum Reformasi Birokrasi LPP TVRI. Sistem ini dirancang menggunakan arsitektur **Hybrid Feature Scoring**, mengintegrasikan analisis leksikal (*BM25 TF-IDF*), analisis semantik laten (*Truncated SVD / LSA*), dan algoritma anti-redundansi (*Maximal Marginal Relevance / MMR*).

Sistem ini bersifat **100% White-Box**, yang memungkinkan pengguna melihat transparansi perhitungan matematis di balik setiap skor kalimat secara *real-time*.

## ✨ Fitur Utama

1. **📄 Ekstraksi PDF Langsung (Hybrid OCR)**
   Ekstraksi teks cerdas menggunakan `PyMuPDF` tanpa bergantung pada OCR berat, lengkap dengan penggabungan baris otomatis (*smart merge*) untuk memperbaiki teks tabel yang terpotong.
2. **🧠 Pipeline NLP Bahasa Indonesia**
   Pembersihan teks terintegrasi menggunakan `Sastrawi` (Stopword Removal & Stemming), dipadukan dengan ekspansi singkatan domain spesifik pemerintah (misal: RB → Reformasi Birokrasi, WTP → Wajar Tanpa Pengecualian).
3. **⚖️ Hybrid Feature Scoring (Fusi 4 Metrik)**
   Setiap kalimat dinilai menggunakan kombinasi:
   *   **Konten (BM25 TF-IDF & LSA):** Kombinasi distribusi frekuensi kata dan penangkapan topik laten (semantik) via dekomposisi matriks SVD.
   *   **Posisi:** Memberikan bias pada kalimat pembuka dan penutup.
   *   **Panjang:** Kurva *Gaussian* untuk menghindari kalimat terlalu pendek (fragmen) atau terlalu panjang (enumerasi).
   *   **Keyword Boost:** Deteksi kata kunci keputusan (misal: "sepakat", "arahan", "tindak lanjut").
4. **🎯 Seleksi Kalimat dengan MMR**
   Algoritma *Maximal Marginal Relevance* memastikan ringkasan yang dihasilkan sangat relevan dengan dokumen asli (keberpusatan) namun tetap beragam (meminimalkan pengulangan informasi).
5. **📊 Evaluasi ROUGE Bawaan**
   Modul kalkulasi matriks *ROUGE-1*, *ROUGE-2*, dan *ROUGE-L* (berbasis LCS) yang terintegrasi untuk mengevaluasi kualitas ringkasan sistem terhadap ringkasan referensi buatan pakar.
6. **🗄️ Persistensi Database (PostgreSQL)**
   Seluruh riwayat, parameter *hyper-tuning*, matriks hasil *scoring* per kalimat, dan skor evaluasi disimpan ke dalam relasional PostgreSQL menggunakan `psycopg2`.

## 🏗️ Arsitektur Sistem

Pipeline sistem dieksekusi secara berurutan dalam hitungan detik:

1. `PDFExtractor` ➔ Mengekstrak teks & metadata dari PDF.
2. `Preprocessor` ➔ Tokenisasi, ekspansi singkatan, dan normalisasi `Sastrawi`.
3. `FeatureScorer` ➔ Membangun matriks TF-IDF dan menghitung skor fitur (*Position, Length, Keyword*).
4. `LSASummarizer` ➔ Menjalankan *Truncated SVD* untuk mereduksi matriks dokumen menjadi vektor ruang laten.
5. `FusionScorer` ➔ Menggabungkan seluruh skor (Leksikal + Semantik + Fitur Teks) menggunakan koefisien fusi (W1-W4).
6. `MMROptimizer` ➔ Pemilihan kandidat kalimat akhir berdasarkan ambang batas kesamaan (*Cosine Similarity*).
7. `Evaluator` ➔ Perhitungan skor *Precision, Recall, F1* menggunakan *ROUGE*.


### Prasyarat:
* Python 3.9 atau lebih baru.
* PostgreSQL berjalan di lokal (untuk fitur simpan riwayat).
