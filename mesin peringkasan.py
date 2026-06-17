# ============================================================
# SISTEM PERINGKASAN NOTULENSI FORUM REFORMASI BIROKRASI
# LPP TVRI — Part 1 dari 3
# ============================================================
# Isi Part 1:
#   - Import library
#   - Class PDFExtractor   : ekstraksi teks dari PDF (tanpa OCR)
#   - Class Preprocessor   : preprocessing Sastrawi (stop+stem)
#   - Class FeatureScorer  : BM25 TF-IDF + Posisi + Panjang + Keyword
#
# Part 2: LSASummarizer, FusionScorer, MMROptimizer, Evaluator, run_pipeline
# Part 3: CSS + Streamlit GUI (main)
# ============================================================


# ============================================================
# SECTION 1 — IMPORTS
# ============================================================
import re
import io

# ── Integrasi Database PostgreSQL ─────────────────────────
try:
    from db_helper import (
        simpan_dokumen, simpan_parameter,
        simpan_kalimat_batch, simpan_ringkasan,
        simpan_evaluasi_rouge, ambil_riwayat,
        ambil_daftar_dokumen, ambil_teks_dokumen,
        ambil_kalimat_dokumen,
        test_koneksi,
    )
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
import math
import tempfile
import os
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

import fitz  # PyMuPDF — ekstraksi teks PDF langsung, tanpa OCR
from PIL import Image
import pytesseract
import docx  
import platform


# Pengecekan otomatis sistem operasi
if platform.system() == "Windows":
    # Jalur ini hanya aktif jika program dijalankan langsung di Windows Anda
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else:
    # Jika dijalankan di dalam Docker (Linux), biarkan kosong karena Linux memanggilnya secara global
    pass


# ============================================================
# SECTION 2 — CLASS PDFExtractor
# ============================================================
class PDFExtractor:
    """
    Mengekstrak teks dari file PDF secara langsung menggunakan PyMuPDF.
    TIDAK menggunakan OCR — hanya bekerja pada PDF dengan teks yang
    bisa diseleksi/di-copy (text-based PDF).

    Alur kerja:
      1. Buka setiap halaman PDF dengan fitz.open()
      2. Deteksi halaman gambar (foto rapat, scan) 
      3. Ekstrak teks dengan page.get_text("text")
      4. Bersihkan karakter bullet (▪ • ► dll.) dan separator tabel (|)
      5. Gabung baris pendek akibat word-wrap kolom tabel notulensi
      6. Filter baris header/footer berulang (NO, PEMBAHASAN, ARAHAN, dll.)
      7. Ekstrak metadata otomatis dari 40 baris pertama dokumen

    Batasan:
      Jika PDF sepenuhnya berbasis gambar/scan (seperti foto yang di-PDF-kan),
      tidak ada teks yang bisa diekstrak. Gunakan mode Input Teks Manual
      dengan cara copy-paste isi notulensi.
    """

    # ── Pola baris yang DILEWATI (header/footer tabel, nomor halaman) ──
    _SKIP = re.compile(
        r"^("
        r"NO\b|NOTULEN(SI)?|FORUM\s+REFORMASI|LPP\s+TVRI|TVRI$|"
        r"PEMBAHASAN$|PEMBICARA$|KESIMPULAN|SARAN$|ARAHAN$|"
        r"Notulis|Ketua\s+Tim|Pelaksana|DOKUMENTASI\s+KEGIATAN|"
        r"hal\.\s*\d+|\d{1,2}\s*$|"       # nomor halaman
        r"[*\-]{1,3}\s*$"                  # hanya tanda strip/bintang
        r")",
        re.IGNORECASE,
    )

    # ── Pola ekstraksi metadata ──────────────────────────────
    _META_PATTERNS = {
        "tanggal":   re.compile(r"Tanggal\s*[:/]?\s*(.{5,50})",        re.I),
        "tema":      re.compile(r"(Judul|Tema|Agenda)\s*[:/]?\s*(.{5,120})", re.I),
        "pembicara": re.compile(r"Pembicara\s*[:/]?\s*(.{5,120})",     re.I),
        "moderator": re.compile(r"Moderator\s*[:/]?\s*(.{5,80})",      re.I),
    }

    def __init__(self):
        self.metadata: dict = {}

    # ── FUNGSI OCR ─────────────────────────────────
    def _ocr_page(self, page: fitz.Page, zoom: float = 2.0) -> str:
        """Mengubah halaman PDF menjadi gambar dan menjalankan OCR."""
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")
        try:
            return pytesseract.image_to_string(img, lang="ind", config="--psm 6")
        except Exception as e:
            return f"\n[ERROR OCR: {e}]\n"
    # ─────────────────────────────────────────────────────────

    # ── Helper: bersihkan satu baris ─────────────────────────
    @staticmethod
    def _clean_line(line: str) -> str:
        """
        Hapus karakter bullet unicode, separator tabel, dan spasi berlebih.
        """
        line = re.sub(r"[▪►◄•▶◀●○▷◁|]", " ", line)   # bullet & pipe
        line = re.sub(r"-\s*\n\s*", "", line)            # hyphen word-break
        line = re.sub(r"\s{2,}", " ", line)              # spasi ganda
        return line.strip()

    # ── Helper: deteksi halaman gambar ───────────────────────
    @staticmethod
    def _is_image_page(page: fitz.Page) -> bool:
        """
        Halaman dianggap berbasis gambar jika jumlah karakter alfabet
        yang diekstrak kurang dari 20 (hanya bullet/noise OCR residual).
        """
        raw_text = page.get_text("text")
        alpha_count = sum(1 for ch in raw_text if ch.isalpha())
        return alpha_count < 20

    # ── Helper: gabung baris pendek (smart merge) ────────────
    @staticmethod
    def _merge_lines(lines: list) -> list:
        """
        Notulensi tabel memiliki kolom sempit sehingga satu kalimat
        terpecah menjadi banyak baris pendek (word-wrap).

        Aturan penggabungan:
          - Buffer diakhiri '.' / '!' / '?' / ':' → simpan, mulai baru
          - Buffer < 70 karakter DAN tidak diakhiri tanda baca → sambung
          - Baris baru dimulai huruf kapital + buffer > 100 char → simpan & baru
          - Kata dengan tanda '-' di akhir buffer → gabung tanpa spasi
        """
        out = []
        buf = ""

        for raw in lines:
            line = raw.strip()

            # Baris kosong = pemisah paragraf
            if not line:
                if buf:
                    out.append(buf)
                    buf = ""
                continue

            # Mulai buffer baru jika masih kosong
            if not buf:
                buf = line
                continue

            last_char = buf.rstrip()[-1] if buf.rstrip() else ""

            if last_char in ".!?:":
                # Akhiran kalimat jelas → simpan, mulai baru
                out.append(buf)
                buf = line
            elif buf.endswith("-"):
                # Kata terpotong tanda hubung → gabung tanpa spasi
                buf = buf[:-1] + line
            elif len(buf) < 70:
                # Buffer pendek → sambung dengan spasi
                buf += " " + line
            elif line and line[0].isupper() and len(buf) > 100:
                # Buffer panjang + baris baru mulai kapital → simpan & baru
                out.append(buf)
                buf = line
            else:
                buf += " " + line

        if buf:
            out.append(buf)

        return out

    # ── Helper: ekstrak metadata dari raw lines ──────────────
    def _extract_metadata(self, lines: list) -> None:
        """
        Cari metadata (tanggal, tema, pembicara, moderator) dari
        40 baris pertama dokumen.
        """
        for line in lines[:40]:
            for key, pattern in self._META_PATTERNS.items():
                if key in self.metadata:
                    continue
                m = pattern.search(line)
                if m:
                    # Grup 2 untuk 'tema' (karena regex punya 2 grup)
                    value = m.group(2) if key == "tema" else m.group(1)
                    self.metadata[key] = value.strip()[:100]

    # ── Helper: post-cleaning teks gabungan ──────────────────
    @staticmethod
    def _post_clean(text: str) -> str:
        """
        Normalisasi karakter tipografi dan perbaiki pemisah kalimat
        yang mepet tanpa spasi (mis. "selesai.Berikutnya" → "selesai. Berikutnya").
        """
        text = re.sub(r'["\u201c\u201d]', '"', text)   # curly → straight quote
        text = re.sub(r"['\u2018\u2019`]", "'", text)  # curly apostrophe
        text = re.sub(r"\.([A-Z])", r". \1", text)     # tambah spasi setelah titik
        # Buang baris yang hanya berisi angka/simbol
        lines = [
            ln for ln in text.split("\n")
            if not re.fullmatch(r"[\d\s.,\-/:()]+", ln.strip())
        ]
        return "\n".join(lines)

    # ── MAIN: ekstrak satu PDF ────────────────────────────────
    def extract(self, pdf_bytes: bytes,
                progress_cb=None) -> tuple:
        """
        Ekstrak seluruh teks dari satu file PDF (dalam bentuk bytes).

        Parameters
        ----------
        pdf_bytes   : bytes konten file PDF
        progress_cb : callable(frac: float, msg: str) untuk progress bar

        Returns
        -------
        (text: str, metadata: dict, info: dict)

        info berisi:
          n_total       : total halaman PDF
          n_text_pages  : halaman yang berhasil diekstrak
          n_image_pages : halaman gambar yang dilewati
          n_chars       : jumlah karakter teks hasil ekstraksi
          is_image_based: True jika TIDAK ADA halaman teks sama sekali
        """
        self.metadata = {}
        raw_lines: list = []
        n_text_pages = 0
        n_image_pages = 0
        n_total = 0

        # Tulis bytes ke file temp karena fitz.open() butuh path/stream
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_f:
            tmp_f.write(pdf_bytes)
            tmp_path = tmp_f.name

        try:
            doc = fitz.open(tmp_path)
            n_total = doc.page_count

            for i, page in enumerate(doc):
                if progress_cb:
                    progress_cb(i / n_total,
                                f"Memproses halaman {i+1}/{n_total}...")

                raw_text = page.get_text("text")

                # LOGIKA HYBRID: Jika teks kurang dari 50 karakter, kemungkinan besar ini PDF Scan/Gambar.
                if len(raw_text.strip()) < 50:
                    if progress_cb:
                        progress_cb(i / n_total, f"Scan terdeteksi. Menjalankan OCR halaman {i+1}/{n_total}...")
                    
                    raw_text = self._ocr_page(page)
                    
                    # Jika setelah di-OCR hasilnya tetap sedikit (< 20 huruf alfa), 
                    # maka ini murni foto dokumentasi rapat. Kita lewati.
                    alpha_count = sum(1 for ch in raw_text if ch.isalpha())
                    if alpha_count < 20:
                        n_image_pages += 1
                        continue

                n_text_pages += 1

                for ln in raw_text.split("\n"):
                    c = self._clean_line(ln)
                    # Filter: minimal 8 karakter + minimal 3 huruf alfa
                    if len(c) < 8:
                        continue
                    if sum(1 for ch in c if ch.isalpha()) < 3:
                        continue
                    # Filter header/footer tabel berulang
                    if self._SKIP.match(c):
                        continue
                    raw_lines.append(c)

                raw_lines.append("")  # pemisah antar halaman

        finally:
            doc.close()
            os.unlink(tmp_path)

        # Proses lanjutan
        self._extract_metadata(raw_lines)
        merged = self._merge_lines(raw_lines)
        text   = self._post_clean("\n".join(merged))
        final  = [ln for ln in text.split("\n") if len(ln.strip()) >= 25]
        result_text = "\n".join(final)

        info = {
            "n_total":       n_total,
            "n_text_pages":  n_text_pages,
            "n_image_pages": n_image_pages,
            "n_chars":       len(result_text),
            "is_image_based": n_text_pages == 0,
        }

        return result_text, dict(self.metadata), info

    # ── MAIN: ekstrak multi-file PDF ──────────────────────────
    def extract_multi(self, files: list,
                      progress_cb=None) -> tuple:
        """
        Gabungkan ekstraksi dari beberapa file PDF.
        Tiap dokumen diberi header pemisah sehingga konteks terjaga
        saat diproses sebagai satu teks gabungan.

        Parameters
        ----------
        files       : list of (filename: str, pdf_bytes: bytes)
        progress_cb : callable(frac, msg)

        Returns
        -------
        (combined_text: str, metadata_list: list, info_list: list)
        """
        segments: list = []
        metas:    list = []
        infos:    list = []
        n_files = len(files)

        for idx, (fname, data) in enumerate(files):
            # Closure untuk progress dengan offset per file
            def _cb(frac, msg, _i=idx, _n=n_files):
                if progress_cb:
                    overall = (_i + frac) / _n
                    progress_cb(overall, f"[{_i+1}/{_n}] {fname} — {msg}")

            text, meta, info = self.extract(data, progress_cb=_cb)
            meta["filename"] = fname
            metas.append(meta)
            infos.append({**info, "filename": fname})

            # Header pemisah dokumen
            tgl  = meta.get("tanggal", "?")
            tema = meta.get("tema", "")[:55]
            header = f"[DOKUMEN {idx+1}: {fname} | Tanggal: {tgl} | Tema: {tema}]"

            segments.append(header)
            segments.append(text)
            segments.append("")   # baris kosong pemisah

        combined_text = "\n".join(segments)
        return combined_text, metas, infos


@st.cache_resource(show_spinner="Memuat modul NLP Sastrawi...")
def load_sastrawi():
    stemmer = StemmerFactory().create_stemmer()
    stopwords = set(StopWordRemoverFactory().get_stop_words())
    return stemmer, stopwords


# ============================================================
# SECTION 3 — CLASS Preprocessor
# ============================================================
class Preprocessor:
    """
    Pra-pemrosesan teks Bahasa Indonesia menggunakan Sastrawi.

    Pipeline lengkap per kalimat:
      1. Ekspansi singkatan (opsional)
         RB → "reformasi birokrasi", BPK → "badan pemeriksa keuangan", dll.
         Tujuan: meningkatkan kecocokan term antar kalimat yang menggunakan
         singkatan berbeda untuk konsep yang sama.

      2. Case folding
         Semua huruf menjadi huruf kecil.

      3. Tokenisasi kata
         Split by whitespace setelah strip karakter non-alfanumerik.

      4. Stopword removal (Sastrawi)
         Buang kata-kata umum yang tidak membawa makna diskriminatif:
         "yang", "dan", "atau", "di", "ke", "dari", dll.
         Ditambah custom stopword khusus teks notulensi formal.

      5. Stemming (Sastrawi)
         Ubah kata ke bentuk dasar:
         "membahas" → "bahas", "disampaikan" → "sampai", dll.

      6. Filter token pendek
         Buang token dengan panjang ≤ 1 karakter.

    Pemisah Kalimat (sentence_tokenize):
      - Split pada . ! ? diikuti spasi + huruf kapital
      - Split pada 2+ newline berturutan
      - Kalimat sangat panjang (>200 char) dengan ; → pecah lagi
    """

    # Ekspansi singkatan umum dalam teks notulensi pemerintah
    _ABBREVIATIONS = {
        r"\bRB\b":        "reformasi birokrasi",
        r"\bASN\b":       "aparatur sipil negara",
        r"\bBPK\b":       "badan pemeriksa keuangan",
        r"\bSPI\b":       "satuan pengawas intern",
        r"\bSAKIP\b":     "sistem akuntabilitas kinerja instansi pemerintah",
        r"\bSPIP\b":      "sistem pengendalian intern pemerintah",
        r"\bWTP\b":       "wajar tanpa pengecualian",
        r"\bBMN\b":       "barang milik negara",
        r"\bZI\b":        "zona integritas",
        r"\bWBK\b":       "wilayah bebas korupsi",
        r"\bWBBM\b":      "wilayah birokrasi bersih melayani",
        r"\bTL\b":        "tindak lanjut",
        r"\bMOU\b":       "memorandum of understanding",
        r"\bPNBP\b":      "penerimaan negara bukan pajak",
        r"\bSDM\b":       "sumber daya manusia",
        r"\bAPBN\b":      "anggaran pendapatan belanja negara",
        r"\bSKP\b":       "sasaran kinerja pegawai",
        r"\bLKJIP\b":     "laporan kinerja instansi pemerintah",
        r"\bIKPA\b":      "indikator kinerja pelaksanaan anggaran",
        r"\bLHP\b":       "laporan hasil pemeriksaan",
        r"\bLHA\b":       "laporan hasil audit",
        r"\bDIPA\b":      "daftar isian pelaksanaan anggaran",
        r"\bKPB\b":       "kuasa pengguna barang",
        r"\bIT\b":        "teknologi informasi",
        r"\bAI\b":        "kecerdasan buatan",
    }

    # Custom stopword tambahan (di luar kamus Sastrawi)
    _CUSTOM_STOPWORDS = {
        "bahwa", "terkait", "yakni", "dimana", "sehingga", "namun",
        "namun", "namun", "adapun", "selain", "serta", "termasuk",
        "mengenai", "terkait", "berkaitan", "menyampaikan", "disampaikan",
        "memaparkan", "menyatakan", "menjelaskan", "mengungkapkan",
        "hal", "hal", "ini", "itu", "tersebut", "tersebut",
        "juga", "pula", "sudah", "telah", "akan", "dapat", "bisa",
        "perlu", "harus", "wajib", "kami", "kita", "bagi", "antara",
        "kepada", "terhadap", "dalam", "untuk", "dengan", "dari",
        "pada", "oleh", "agar", "maka", "karena", "apabila", "jika",
        "ketika", "saat", "selama", "setelah", "sebelum", "kemudian",
        "lalu", "lagi", "masih", "sudah", "belum", "tidak", "bukan",
        "sangat", "lebih", "paling", "cukup", "setiap", "seluruh",
        "semua", "masing", "masing", "berbagai", "beberapa", "sebuah",
        "suatu", "satu", "dua", "tiga", "empat", "lima",
    }

    def __init__(self, expand_abbr: bool = True):
        self.stemmer, base_stopwords = load_sastrawi()
        self.stopwords = base_stopwords.copy()
        self.stopwords.update(self._CUSTOM_STOPWORDS)
        self.expand_abbr = expand_abbr

    # ── Normalisasi & ekspansi singkatan ─────────────────────
    def _normalize(self, text: str) -> str:
        """
        Perbaiki tanda hubung antar baris, lalu ekspansi singkatan
        jika expand_abbr = True.
        """
        text = re.sub(r"-\s+", "", text)  # gabung kata terpotong hyphen
        if self.expand_abbr:
            for pattern, replacement in self._ABBREVIATIONS.items():
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    # ── Tokenisasi kalimat ────────────────────────────────────
    def sentence_tokenize(self, text: str) -> list:
        """
        Pisahkan teks menjadi daftar kalimat.

        Aturan split:
          1. [.!?] diikuti spasi dan huruf kapital
          2. [.!?] diikuti newline
          3. Dua newline atau lebih berturutan

        Post-processing:
          - Kalimat < 20 karakter dibuang
          - Kalimat > 200 karakter dengan ';' → dipecah lagi
        """
        text = self._normalize(text)
        raw  = re.split(
            r"(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\n|\n{2,}",
            text.strip()
        )
        sentences: list = []
        for s in raw:
            s = s.strip()
            if len(s) < 20:
                continue
            # Pecah kalimat sangat panjang pada titik koma
            if len(s) > 200 and ";" in s:
                for part in s.split(";"):
                    part = part.strip()
                    if len(part) >= 20:
                        sentences.append(part)
            else:
                sentences.append(s)
        return sentences

    # ── Tokenisasi kata ───────────────────────────────────────
    def word_tokenize(self, text: str) -> list:
        """
        Lowercase + hapus karakter non-alfanumerik + split by whitespace.
        """
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return [t for t in text.split() if len(t) > 1]

    # ── Preprocessing satu kalimat ────────────────────────────
    def preprocess_sentence(self, sentence: str) -> list:
        """
        Pipeline: tokenize → stopword removal → stemming → filter pendek.
        """
        tokens = self.word_tokenize(sentence)
        tokens = [t for t in tokens if t not in self.stopwords]
        tokens = [self.stemmer.stem(t) for t in tokens]
        return [t for t in tokens if len(t) > 1]

    # ── Proses seluruh teks ───────────────────────────────────
    def process(self, text: str) -> dict:
        """
        Proses teks lengkap → kembalikan kalimat asli & versi token.

        Returns
        -------
        {
            "original_sentences"  : list[str],  # kalimat asli
            "processed_sentences" : list[list[str]],  # token bersih
        }
        """
        original  = self.sentence_tokenize(text)
        processed = [self.preprocess_sentence(s) for s in original]
        return {
            "original_sentences":  original,
            "processed_sentences": processed,
        }


# ============================================================
# SECTION 4 — CLASS FeatureScorer
# ============================================================
class FeatureScorer:
    """
    Menghitung 4 fitur skor statistik untuk setiap kalimat:

    ┌─────────────────────────────────────────────────────────┐
    │ 1. BM25 TF-IDF Score                                    │
    │    TF BM25(t,d) = freq*(k1+1) / (freq + k1*(1-b+b*dl/avgdl))  │
    │    IDF(t)       = log((N-df+0.5)/(df+0.5) + 1)         │
    │    Score(t,d)   = TF_BM25 × IDF                         │
    │                                                         │
    │    k1=1.5, b=0.75 (parameter BM25 standar Okapi)        │
    │    Lebih robust dari TF murni untuk teks panjang-pendek │
    ├─────────────────────────────────────────────────────────┤
    │ 2. Position Score                                       │
    │    pos(i) = 1/(i+1)  (kalimat pertama = paling penting) │
    │    + booster 0.4 untuk 5 kalimat terakhir               │
    │    (kesimpulan/penutup sering di akhir paragraf)        │
    ├─────────────────────────────────────────────────────────┤
    │ 3. Length Score (Gaussian bell-curve)                   │
    │    len(i) = exp(-0.5 × ((len_i - μ) / σ)²)             │
    │    Kalimat terlalu pendek (fragmen) atau terlalu panjang │
    │    (enumerasi) cenderung kurang informatif               │
    ├─────────────────────────────────────────────────────────┤
    │ 4. Keyword Score (domain-specific boosting)             │
    │    Deteksi kata kunci penting notulensi pemerintah:     │
    │    keputusan, sepakat, arahan, wtp, bpk, zona integritas│
    │    score = min(jumlah_keyword_hits / 5, 1.0)            │
    └─────────────────────────────────────────────────────────┘

    Semua skor dinormalisasi ke [0, 1] dengan min-max scaling.
    """

    # Kata kunci penting domain notulensi Forum RB
    # Kalimat yang mengandung kata ini kemungkinan adalah kalimat
    # kunci/keputusan yang WAJIB masuk ringkasan
    DOMAIN_KEYWORDS = {
        # Kata aksi/keputusan/arahan
        "keputusan",      "menyepakati",    "sepakat",
        "disepakati",     "diputuskan",     "kesimpulan",
        "rekomendasi",    "arahan",         "instruksi",
        "menginstruksikan","menetapkan",    "ditetapkan",
        "tindak lanjut",  "ditindaklanjuti","percepatan",
        "segera",         "wajib",          "prioritas",
        "target",         "deadline",
        # Topik substantif Forum RB
        "reformasi birokrasi",  "zona integritas",
        "transformasi digital", "piala dunia",
        "wajar tanpa pengecualian",          # WTP (sudah di-expand)
        "badan pemeriksa keuangan",          # BPK
        "barang milik negara",               # BMN
        "tunjangan kinerja",
        "anggaran",       "efisiensi",       "digitalisasi",
        "sakip",          "spip",            "wilayah bebas korupsi",
        # Kata pembicara/peran
        "direktur",       "kepala stasiun",  "pimpinan",
    }

    def __init__(self):
        self.tfidf_matrix  = None
        self.vocabulary    = []
        self.idf_values    = {}
        self._avg_doc_len  = 1.0
        # BM25 hyperparameter
        self.k1 = 1.5
        self.b  = 0.75

    # ── BM25-style TF ─────────────────────────────────────────
    def _compute_bm25_tf(self, tokens: list, avg_len: float) -> dict:
        """
        BM25 TF normalization.
        TF_BM25(t,d) = freq(t,d) * (k1+1) / (freq(t,d) + k1*(1-b+b*|d|/avgdl))

        Dibanding raw TF, BM25 TF lebih stabil karena:
          - Tidak terus naik linear dengan frekuensi (ada saturasi k1)
          - Memperhitungkan panjang dokumen (b*dl/avgdl)
        """
        if not tokens:
            return {}
        freq_map = Counter(tokens)
        dl       = len(tokens)
        result   = {}
        for term, freq in freq_map.items():
            numerator   = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * dl / avg_len)
            result[term] = numerator / denominator
        return result

    # ── BM25-style IDF ────────────────────────────────────────
    def _compute_idf(self, processed: list) -> dict:
        """
        BM25 IDF:
        IDF(t) = log( (N - df(t) + 0.5) / (df(t) + 0.5)  +  1 )

        N    = jumlah kalimat (dokumen)
        df(t) = jumlah kalimat yang mengandung term t
        +1  = smoothing agar IDF tidak negatif bila df(t) > N/2
        """
        N  = len(processed)
        df = Counter()
        for toks in processed:
            df.update(set(toks))   # hitung per kalimat, bukan per token
        return {
            term: math.log((N - freq + 0.5) / (freq + 0.5) + 1)
            for term, freq in df.items()
        }

    # ── Build TF-IDF Matrix ───────────────────────────────────
    def build_tfidf_matrix(self, processed: list) -> np.ndarray:
        """
        Bangun matriks BM25 TF-IDF berukuran (n_kalimat × n_term).
        Sel [i][j] = BM25 TF-IDF dari term j di kalimat i.

        Parameters
        ----------
        processed : list[list[str]] — token bersih per kalimat

        Returns
        -------
        np.ndarray shape (n_sent, n_term)
        """
        # Bangun vocabulary dari semua token
        all_terms = sorted(set(t for toks in processed for t in toks))
        self.vocabulary    = all_terms
        term_to_idx        = {t: i for i, t in enumerate(all_terms)}

        # Hitung avg document length untuk normalisasi BM25
        self._avg_doc_len  = max(np.mean([len(t) for t in processed]), 1.0)
        self.idf_values    = self._compute_idf(processed)

        n_sent = len(processed)
        n_term = len(all_terms)
        matrix = np.zeros((n_sent, n_term))

        for i, toks in enumerate(processed):
            bm25_tf = self._compute_bm25_tf(toks, self._avg_doc_len)
            for term, tf_val in bm25_tf.items():
                if term in term_to_idx:
                    j = term_to_idx[term]
                    matrix[i][j] = tf_val * self.idf_values.get(term, 0.0)

        self.tfidf_matrix = matrix
        return matrix

    # ── Normalisasi Min-Max ───────────────────────────────────
    @staticmethod
    def _minmax(arr: np.ndarray) -> np.ndarray:
        """Normalisasi array ke rentang [0, 1]."""
        r = arr.max() - arr.min()
        if r > 0:
            return (arr - arr.min()) / r
        return np.full_like(arr, 0.5)  # semua nilai sama → 0.5

    # ── Skor 1: TF-IDF ───────────────────────────────────────
    def score_tfidf(self, mat: np.ndarray) -> np.ndarray:
        """
        Skor kalimat = rata-rata BM25 TF-IDF semua term.
        Kalimat dengan term-term bernilai tinggi → skor tinggi.
        """
        return self._minmax(mat.mean(axis=1))

    # ── Skor 2: Position ─────────────────────────────────────
    def score_position(self, n: int) -> np.ndarray:
        """
        Hybrid Position Score:
          - Kalimat awal: 1/(i+1)  → descending
          - Kalimat ~5 terakhir: floor ke min 0.4 (sering berisi kesimpulan)

        Intuisi: dalam notulensi, informasi penting ada di awal
        (pembukaan & topik) dan di akhir (kesimpulan & arahan).
        """
        arr  = np.array([1.0 / (i + 1) for i in range(n)])
        tail = min(5, max(1, n // 4))
        for i in range(n - tail, n):
            arr[i] = max(arr[i], 0.4)
        return self._minmax(arr)

    # ── Skor 3: Length (Gaussian) ────────────────────────────
    def score_length(self, processed: list) -> np.ndarray:
        """
        Gaussian bell-curve berdasarkan panjang kalimat (jumlah token).

        Rumus: score(i) = exp(-0.5 × ((len_i - μ) / σ)²)
          - Nilai maksimum di panjang rata-rata (μ)
          - Kalimat terlalu pendek (<5 token): fragmen, skor rendah
          - Kalimat terlalu panjang (>30 token): enumerasi, skor rendah
        """
        lengths = np.array([max(len(t), 1) for t in processed], dtype=float)
        mu      = lengths.mean()
        sigma   = max(lengths.std(), 1.0)
        scores  = np.exp(-0.5 * ((lengths - mu) / sigma) ** 2)
        return self._minmax(scores)

    # ── Skor 4: Keyword Boost ────────────────────────────────
    def score_keyword(self, original_sentences: list) -> np.ndarray:
        """
        Domain-specific keyword boosting untuk teks notulensi TVRI.

        Deteksi keberadaan kata kunci penting dalam kalimat asli.
        Kalimat berisi kata kunci keputusan/arahan/topik penting
        mendapat skor lebih tinggi.

        Rumus: score = min(jumlah_hit_keyword / 5.0, 1.0)
          - 0 hit = 0.0
          - 1 hit = 0.2
          - 3 hit = 0.6
          - 5+ hit = 1.0 (saturasi)
        """
        kw_lower = {kw.lower() for kw in self.DOMAIN_KEYWORDS}
        scores   = []
        for sent in original_sentences:
            sent_l = sent.lower()
            hits   = sum(1 for kw in kw_lower if kw in sent_l)
            scores.append(min(hits / 5.0, 1.0))
        return self._minmax(np.array(scores, dtype=float))
    
# ============================================================
# SISTEM PERINGKASAN NOTULENSI FORUM REFORMASI BIROKRASI
# LPP TVRI — Part 2 dari 3
# ============================================================
# Isi Part 2:
#   - Class LSASummarizer  : pemodelan semantik via Truncated SVD
#   - Class FusionScorer   : gabungkan 4 skor dengan bobot
#   - Class MMROptimizer   : seleksi kalimat greedy anti-redundansi
#   - Class Evaluator      : hitung ROUGE-1, ROUGE-2, ROUGE-L
#   - Fungsi run_pipeline  : orkestrasi seluruh tahap (end-to-end)
#
# Part 1: Import, PDFExtractor, Preprocessor, FeatureScorer
# Part 3: CSS + Streamlit GUI (main)
#
# CATATAN: File ini harus digabung dengan Part 1 dan Part 3
#          menjadi satu file app.py sebelum dijalankan.
#          Urutan: Part 1 → Part 2 → Part 3
# ============================================================


# ============================================================
# SECTION 5 — CLASS LSASummarizer
# ============================================================
class LSASummarizer:
    """
    Pemodelan semantik menggunakan Latent Semantic Analysis (LSA)
    melalui Truncated Singular Value Decomposition (SVD).

    ─── Latar Belakang ──────────────────────────────────────
    TF-IDF hanya menangkap kecocokan leksikal (kata yang sama).
    LSA menangkap kesamaan SEMANTIK: dua kalimat yang membahas
    topik serupa tetapi menggunakan kata berbeda tetap bisa
    dianggap "mirip" dalam ruang laten.

    ─── Rumus SVD ───────────────────────────────────────────
    Diberikan matriks TF-IDF A berukuran (n_sent × n_term):

        A  =  U  ×  Σ  ×  Vᵀ

    Dimana:
      U   : matriks left singular vectors    (n_sent × k)
            → representasi setiap KALIMAT dalam ruang semantik
      Σ   : matriks diagonal singular values (k × k)
            → "kekuatan" atau bobot setiap konsep laten
      Vᵀ  : matriks right singular vectors   (k × n_term)
            → representasi setiap TERM dalam ruang semantik
      k   : jumlah komponen laten yang dipertahankan

    Truncated SVD hanya menghitung k komponen terbesar,
    sehingga efisien secara komputasi dan membuang noise.

    ─── Skor LSA per kalimat ────────────────────────────────
    Setelah L2-normalisasi pada U:

        Skor_LSA(i) = ‖ U[i] × Σ ‖₂

    Intuisi: kalimat ke-i mendapat skor tinggi jika vektor
    representasinya dalam ruang laten, setelah dibobot dengan
    singular values Σ, memiliki norma besar — artinya kalimat
    tersebut berkontribusi signifikan pada konsep-konsep dominan.

    ─── Mengapa L2-normalisasi sebelum MMR? ─────────────────
    Cosine similarity membutuhkan vektor ternormalisasi agar
    hasilnya setara antara kalimat panjang dan pendek.
    normalize(U, norm="l2") memastikan ‖U[i]‖ = 1 untuk semua i.
    """

    def __init__(self, n_components: int = 5):
        """
        Parameters
        ----------
        n_components : int
            Jumlah komponen laten SVD (k).
            Nilai kecil (2-3) : menangkap topik utama saja
            Nilai besar (8-10): menangkap lebih banyak nuansa
            Rekomendasi       : 3-5 untuk dokumen notulensi
        """
        self.n_components     = n_components
        self.svd_model        = None
        self.sentence_vectors = None   # matriks U ternormalisasi (n_sent × k)
        self.singular_values  = None   # vektor diagonal Σ (k,)
        self.explained_var    = None   # proporsi varians tiap komponen

    def fit_transform(self, tfidf_matrix: np.ndarray) -> np.ndarray:
        """
        Terapkan Truncated SVD pada matriks TF-IDF.

        Parameters
        ----------
        tfidf_matrix : np.ndarray shape (n_sent, n_term)

        Returns
        -------
        sentence_vectors : np.ndarray shape (n_sent, k)
            Representasi kalimat dalam ruang laten, sudah L2-normalized.

        Langkah internal:
          1. Tentukan k = min(n_components, n_sent-1, n_term)
             agar tidak melebihi rank maksimal matriks
          2. Fit TruncatedSVD → dapatkan U, Σ, Vᵀ
          3. L2-normalize baris U → siap untuk Cosine Similarity
        """
        n_sent, n_term = tfidf_matrix.shape

        # Pastikan k tidak melebihi batas rank matriks
        k = max(1, min(self.n_components, n_sent - 1, n_term))

        # Gunakan n_iter=10 untuk konvergensi lebih baik
        self.svd_model = TruncatedSVD(
            n_components=k,
            random_state=42,
            n_iter=10,
            algorithm="randomized",
        )

        # fit_transform langsung menghasilkan U × Σ (bukan U murni)
        # Kita pisahkan dengan mengakses singular_values_ secara terpisah
        U_scaled = self.svd_model.fit_transform(tfidf_matrix)  # (n_sent × k)

        self.singular_values = self.svd_model.singular_values_         # (k,)
        self.explained_var   = self.svd_model.explained_variance_ratio_ # (k,)

        # L2-normalize setiap baris → panjang vektor = 1
        # Diperlukan agar Cosine Similarity di MMR akurat
        self.sentence_vectors = normalize(U_scaled, norm="l2")

        return self.sentence_vectors

    def score_lsa(self) -> np.ndarray:
        """
        Hitung skor LSA untuk setiap kalimat.

        Rumus:
            Skor_LSA(i) = ‖ sentence_vectors[i] × singular_values ‖₂

        Karena sentence_vectors sudah L2-normalized, perkalian dengan
        singular_values memberikan "kepentingan" relatif kalimat
        terhadap setiap konsep laten (komponen SVD).

        Kalimat yang kuat berkontribusi pada konsep-konsep dominan
        (singular value besar) mendapat skor LSA tinggi.

        Returns
        -------
        scores : np.ndarray shape (n_sent,) dinormalisasi ke [0, 1]
        """
        if self.sentence_vectors is None:
            return np.array([])

        # Element-wise multiply: broadcast singular_values ke setiap baris
        weighted = self.sentence_vectors * self.singular_values   # (n_sent × k)
        scores   = np.linalg.norm(weighted, axis=1)               # (n_sent,)

        # Min-max normalisasi ke [0, 1]
        r = scores.max() - scores.min()
        if r > 0:
            return (scores - scores.min()) / r
        return scores


# ============================================================
# SECTION 6 — CLASS FusionScorer
# ============================================================
class FusionScorer:
    """
    Menggabungkan 4 fitur skor menjadi satu skor fusi menggunakan
    Linear Weighted Sum (Rule-of-Sum berbobot).

    ─── Rumus Fusi ──────────────────────────────────────────

        Konten(i) = α × TF-IDF(i) + (1-α) × LSA(i)

        Fusi(i)   = ŵ₁ × Konten(i)
                  + ŵ₂ × Posisi(i)
                  + ŵ₃ × Panjang(i)
                  + ŵ₄ × Keyword(i)

    Dimana:
      ŵⱼ = wⱼ / (w₁ + w₂ + w₃ + w₄)   [bobot dinormalisasi]
      α  = 0.55 (porsi TF-IDF dalam skor Konten)

    ─── Mengapa 4 fitur? ────────────────────────────────────
    • Konten (TF-IDF + LSA): relevansi leksikal & semantik
    • Posisi               : bias informasi di awal teks
    • Panjang              : preferensi kalimat informatif sedang
    • Keyword              : kepentingan domain-specific (keputusan,
                             arahan, sepakat, tindak lanjut, dll.)

    Menggabungkan keempat fitur menghasilkan skor yang lebih
    robust dibanding hanya menggunakan satu fitur.
    """

    @staticmethod
    def fuse(
        sc_tfidf:  np.ndarray,
        sc_lsa:    np.ndarray,
        sc_pos:    np.ndarray,
        sc_len:    np.ndarray,
        sc_kw:     np.ndarray,
        w1: float = 0.40,
        w2: float = 0.20,
        w3: float = 0.15,
        w4: float = 0.25,
        alpha: float = 0.55,
    ) -> np.ndarray:
        """
        Hitung skor fusi untuk setiap kalimat.

        Parameters
        ----------
        sc_tfidf : np.ndarray  — skor BM25 TF-IDF    [0, 1]
        sc_lsa   : np.ndarray  — skor LSA             [0, 1]
        sc_pos   : np.ndarray  — skor posisi          [0, 1]
        sc_len   : np.ndarray  — skor panjang         [0, 1]
        sc_kw    : np.ndarray  — skor keyword boost   [0, 1]
        w1       : float       — bobot fitur konten
        w2       : float       — bobot fitur posisi
        w3       : float       — bobot fitur panjang
        w4       : float       — bobot fitur keyword
        alpha    : float       — porsi TF-IDF vs LSA dalam Konten

        Returns
        -------
        fusion_scores : np.ndarray [0, ~1]
            Skor tidak diclip ke 1.0 agar gradasi antar kalimat terjaga.
        """
        # Normalisasi bobot sehingga ŵ₁ + ŵ₂ + ŵ₃ + ŵ₄ = 1
        total = w1 + w2 + w3 + w4
        if total <= 0:
            # Fallback: semua bobot sama rata
            w1 = w2 = w3 = w4 = 0.25
            total = 1.0

        # Skor konten: kombinasi leksikal (TF-IDF) dan semantik (LSA)
        content_score = alpha * sc_tfidf + (1.0 - alpha) * sc_lsa

        # Fusi linear berbobot
        fusion = (
            (w1 / total) * content_score +
            (w2 / total) * sc_pos        +
            (w3 / total) * sc_len        +
            (w4 / total) * sc_kw
        )

        return fusion


# ============================================================
# SECTION 7 — CLASS MMROptimizer
# ============================================================
class MMROptimizer:
    """
    Maximal Marginal Relevance (MMR) — Seleksi kalimat secara
    greedy iteratif yang memaksimalkan relevansi sekaligus
    meminimalkan redundansi.

    ─── Rumus MMR ───────────────────────────────────────────

        MMR(i) = argmax_{sᵢ ∈ R - S}  [
                    λ  ×  Rel(sᵢ, D)
                 −  (1−λ)  ×  max_{sⱼ ∈ S}  Sim(sᵢ, sⱼ)
                 ]

    Dimana:
      R       = himpunan semua kalimat kandidat
      S       = himpunan kalimat yang SUDAH dipilih
      D       = representasi dokumen (rata-rata semua vektor kalimat)
      λ       = parameter trade-off relevansi vs keberagaman
      Rel(sᵢ,D) = gabungan cosine similarity ke centroid + fusion score
      Sim(A,B)  = cosine similarity antar kalimat

    ─── Cosine Similarity ───────────────────────────────────

        cos(A, B) = (A · B) / (‖A‖ × ‖B‖)

    Karena sentence_vectors sudah L2-normalized (‖v‖=1),
    cosine similarity = dot product sederhana = A · B.

    ─── Algoritma Greedy ────────────────────────────────────
    Iterasi 1:
      Pilih kalimat dengan relevansi tertinggi terhadap D.
      (Tidak ada S dulu, jadi komponen redundansi = 0)

    Iterasi k > 1:
      Untuk setiap kandidat, hitung MMR score.
      Pilih kandidat dengan MMR score tertinggi.
      Tambahkan ke S, hapus dari R.

    ─── Perbaikan dibanding MMR standar ─────────────────────
    1. Similarity threshold (default 0.85):
       Kalimat yang sangat mirip (sim > threshold) dengan kalimat
       terpilih langsung DILEWATI, tidak perlu dihitung MMR-nya.
       Mencegah kalimat parafrase masuk ringkasan.

    2. Output kronologis:
       Hasil MMR diurutkan berdasarkan posisi asli (sorted by index),
       sehingga ringkasan tetap terbaca kronologis seperti teks asli.

    3. Rel(sᵢ, D) = 0.5×cos(sᵢ, centroid) + 0.5×fusion(i):
       Relevansi tidak hanya bergantung cosine similarity ke centroid,
       tetapi juga mempertimbangkan fusion score (yang sudah
       memperhitungkan TF-IDF, LSA, posisi, panjang, keyword).
    """

    @staticmethod
    def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        """
        Hitung cosine similarity antara dua vektor.

        cos(A, B) = (A · B) / (‖A‖ × ‖B‖)

        Mengembalikan 0.0 jika salah satu vektor adalah zero vector
        untuk menghindari division by zero.
        """
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def select(
        self,
        sentence_vectors: np.ndarray,
        fusion_scores:    np.ndarray,
        n_select:         int,
        lambda_param:     float = 0.70,
        sim_threshold:    float = 0.85,
    ) -> list:
        """
        Jalankan algoritma MMR untuk memilih n_select kalimat.

        Parameters
        ----------
        sentence_vectors : np.ndarray (n_sent × k)
            Representasi kalimat dalam ruang laten (L2-normalized).
        fusion_scores    : np.ndarray (n_sent,)
            Skor fusi dari FusionScorer.
        n_select         : int
            Jumlah kalimat yang ingin dipilih.
        lambda_param     : float [0, 1]
            λ → 1 : prioritaskan RELEVANSI (lebih sedikit keberagaman)
            λ → 0 : prioritaskan KEBERAGAMAN (kurangi redundansi)
        sim_threshold    : float [0, 1]
            Kalimat dengan max similarity > threshold langsung dilewati.
            Mencegah kalimat hampir identik masuk ringkasan.

        Returns
        -------
        selected : list[int]
            Indeks kalimat terpilih, diurutkan KRONOLOGIS (ascending).
        """
        n_total    = len(fusion_scores)
        n_pick     = min(max(n_select, 1), n_total)

        # Representasi dokumen = rata-rata semua vektor kalimat
        doc_centroid = sentence_vectors.mean(axis=0)

        candidates: list = list(range(n_total))
        selected  : list = []

        # Sesuaikan threshold: λ rendah → threshold lebih ketat
        # (mode diversity butuh jaminan kalimat benar-benar berbeda)
        effective_threshold = sim_threshold - (1.0 - lambda_param) * 0.15

        for _ in range(n_pick):
            if not candidates:
                break

            mmr_scores: dict = {}

            for idx in candidates:

                # ── Komponen Relevansi ───────────────────────
                # Gabungan cosine ke centroid + fusion score
                cos_to_centroid = self.cosine_sim(
                    sentence_vectors[idx], doc_centroid
                )
                relevance = lambda_param * (
                    0.5 * cos_to_centroid + 0.5 * fusion_scores[idx]
                )

                # ── Komponen Redundansi ──────────────────────
                if selected:
                    # Similarity maksimum terhadap kalimat yang sudah dipilih
                    similarity_to_selected = [
                        self.cosine_sim(
                            sentence_vectors[idx],
                            sentence_vectors[j]
                        )
                        for j in selected
                    ]
                    max_sim = max(similarity_to_selected)

                    # Skip jika terlalu mirip (redundant)
                    if max_sim > effective_threshold:
                        continue

                    penalty = (1.0 - lambda_param) * max_sim
                    mmr_scores[idx] = relevance - penalty

                else:
                    # Iterasi pertama: tidak ada S, tidak ada penalti redundansi
                    mmr_scores[idx] = relevance

            # Tidak ada kandidat yang lolos threshold → berhenti
            if not mmr_scores:
                break

            # Pilih kandidat dengan MMR score tertinggi
            best_idx = max(mmr_scores, key=mmr_scores.get)
            selected.append(best_idx)
            candidates.remove(best_idx)

        # Kembalikan dalam urutan kronologis (posisi asli di teks)
        return sorted(selected)


# ============================================================
# SECTION 8 — CLASS Evaluator (ROUGE)
# ============================================================
class Evaluator:
    """
    Evaluasi kualitas ringkasan menggunakan metrik ROUGE standar.

    ─── ROUGE (Recall-Oriented Understudy for Gisting Evaluation) ───

    ROUGE mengukur seberapa besar overlap antara ringkasan sistem
    (otomatis) dan ringkasan referensi (buatan pakar/manusia).

    ─── ROUGE-N ─────────────────────────────────────────────
    Berdasarkan overlap N-gram antara sistem dan referensi.

        Precision_N = |overlap N-gram| / |N-gram sistem|
        Recall_N    = |overlap N-gram| / |N-gram referensi|
        F1_N        = 2 × P × R / (P + R)

    • ROUGE-1 (N=1): overlap unigram (kata tunggal)
      → mengukur kesesuaian konten/isi secara kasar

    • ROUGE-2 (N=2): overlap bigram (pasangan kata berturutan)
      → mengukur kesesuaian frasa/konteks lebih ketat

    ─── ROUGE-L ─────────────────────────────────────────────
    Berdasarkan Longest Common Subsequence (LCS) antara sistem
    dan referensi. Subsequence tidak harus berurutan (non-contiguous).

    Implementasi menggunakan Dynamic Programming O(m×n):
        dp[i][j] = dp[i-1][j-1] + 1       jika sys[i] == ref[j]
                 = max(dp[i-1][j], dp[i][j-1])  sebaliknya

    Optimasi ruang: hanya menyimpan 2 baris (O(n) space).

        P_L = LCS / |sistem|
        R_L = LCS / |referensi|
        F_L = 2 × P_L × R_L / (P_L + R_L)

    ROUGE-L lebih toleran terhadap perbedaan urutan kata
    dibanding ROUGE-2, cocok untuk bahasa dengan variasi
    struktur kalimat yang tinggi seperti Bahasa Indonesia.

    ─── Interpretasi Skor ───────────────────────────────────
    F1 ≥ 0.50 : Kualitas Baik   — ringkasan sangat representatif
    F1 ≥ 0.30 : Kualitas Cukup  — perlu penyesuaian parameter
    F1 < 0.30 : Perlu Peningkatan — review parameter fusion/MMR
    """

    # ── Helper: generate N-gram Counter ──────────────────────
    @staticmethod
    def _ngrams(tokens: list, n: int) -> Counter:
        """
        Buat Counter dari semua N-gram dalam daftar token.

        Contoh (n=2, tokens=["a","b","c"]):
          → Counter({("a","b"): 1, ("b","c"): 1})
        """
        return Counter(
            tuple(tokens[i : i + n])
            for i in range(len(tokens) - n + 1)
        )

    def rouge_n(self,
                sys_tokens: list,
                ref_tokens: list,
                n:          int) -> dict:
        """
        Hitung ROUGE-N antara ringkasan sistem dan referensi.

        Parameters
        ----------
        sys_tokens : list[str]  — token ringkasan sistem (lowercase)
        ref_tokens : list[str]  — token ringkasan referensi (lowercase)
        n          : int        — ukuran N-gram (1 atau 2)

        Returns
        -------
        dict dengan keys: "precision", "recall", "f1"
        """
        sys_ngrams = self._ngrams(sys_tokens, n)
        ref_ngrams = self._ngrams(ref_tokens, n)

        # Overlap = irisan (min count) antara sistem dan referensi
        overlap_count = sum((sys_ngrams & ref_ngrams).values())

        total_sys = sum(sys_ngrams.values())
        total_ref = sum(ref_ngrams.values())

        # Handle edge case: tidak ada N-gram
        if total_sys == 0 or total_ref == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        precision = overlap_count / total_sys
        recall    = overlap_count / total_ref
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        return {"precision": precision, "recall": recall, "f1": f1}

    @staticmethod
    def _lcs_length(seq_a: list, seq_b: list) -> int:
        """
        Hitung panjang Longest Common Subsequence (LCS)
        antara dua sekuens menggunakan Dynamic Programming.

        Kompleksitas: O(m×n) waktu, O(n) ruang (space-optimized).

        Algoritma:
          dp[j] = panjang LCS dari seq_a[:i] dan seq_b[:j]
          Untuk setiap pasangan (i, j):
            Jika seq_a[i-1] == seq_b[j-1] → dp[j] = prev[j-1] + 1
            Jika tidak                     → dp[j] = max(dp[j-1], prev[j])
        """
        m, n   = len(seq_a), len(seq_b)
        prev   = [0] * (n + 1)  # baris sebelumnya

        for i in range(1, m + 1):
            curr = [0] * (n + 1)  # baris saat ini
            for j in range(1, n + 1):
                if seq_a[i - 1] == seq_b[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(curr[j - 1], prev[j])
            prev = curr  # geser baris

        return prev[n]

    def rouge_l(self,
                sys_tokens: list,
                ref_tokens: list) -> dict:
        """
        Hitung ROUGE-L berdasarkan Longest Common Subsequence.

        Parameters
        ----------
        sys_tokens : list[str]  — token ringkasan sistem
        ref_tokens : list[str]  — token ringkasan referensi

        Returns
        -------
        dict dengan keys: "precision", "recall", "f1"
        """
        if not sys_tokens or not ref_tokens:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        lcs_len = self._lcs_length(sys_tokens, ref_tokens)

        precision = lcs_len / len(sys_tokens)
        recall    = lcs_len / len(ref_tokens)
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        return {"precision": precision, "recall": recall, "f1": f1}

    def evaluate(self, sys_text: str, ref_text: str) -> dict:
        """
        Hitung semua metrik ROUGE sekaligus.

        Preprocessing sederhana: lowercase + split by whitespace.
        (Tidak perlu stemming untuk evaluasi ROUGE — standar umum)

        Parameters
        ----------
        sys_text : str  — teks ringkasan sistem (otomatis)
        ref_text : str  — teks ringkasan referensi (pakar)

        Returns
        -------
        {
            "ROUGE-1": {"precision": ..., "recall": ..., "f1": ...},
            "ROUGE-2": {"precision": ..., "recall": ..., "f1": ...},
            "ROUGE-L": {"precision": ..., "recall": ..., "f1": ...},
        }
        """
        sys_tokens = sys_text.lower().split()
        ref_tokens = ref_text.lower().split()

        return {
            "ROUGE-1": self.rouge_n(sys_tokens, ref_tokens, n=1),
            "ROUGE-2": self.rouge_n(sys_tokens, ref_tokens, n=2),
            "ROUGE-L": self.rouge_l(sys_tokens, ref_tokens),
        }


# ============================================================
# SECTION 9 — FUNGSI run_pipeline
# ============================================================
def run_pipeline(
    text:             str,
    compression_rate: float,
    lambda_mmr:       float,
    w1:               float,
    w2:               float,
    w3:               float,
    w4:               float,
    n_components:     int   = 5,
    expand_abbr:      bool  = True,
) -> dict:
    """
    Orkestrasi seluruh pipeline peringkasan end-to-end.

    ─── Alur Tahap ──────────────────────────────────────────
    INPUT TEXT
        │
        ▼
    [1] Preprocessor
        • Ekspansi singkatan (opsional)
        • Tokenisasi kalimat & kata
        • Stopword removal (Sastrawi)
        • Stemming (Sastrawi)
        │
        ▼ original_sentences, processed_sentences
        │
    [2] FeatureScorer
        • Build BM25 TF-IDF Matrix (n_sent × n_term)
        • Hitung skor TF-IDF per kalimat
        • Hitung skor Posisi
        • Hitung skor Panjang (Gaussian)
        • Hitung skor Keyword (domain notulensi TVRI)
        │
        ▼ tfidf_matrix, sc_tfidf, sc_pos, sc_len, sc_kw
        │
    [3] LSASummarizer
        • Truncated SVD: A = U × Σ × Vᵀ
        • L2-normalize matriks U
        • Hitung skor LSA = ‖U[i] × Σ‖₂
        │
        ▼ sentence_vectors, sc_lsa
        │
    [4] FusionScorer
        • Konten = α×TF-IDF + (1-α)×LSA
        • Fusi = ŵ₁Konten + ŵ₂Posisi + ŵ₃Panjang + ŵ₄Keyword
        │
        ▼ fusion_scores
        │
    [5] MMROptimizer
        • Greedy iteratif dengan cosine similarity
        • Anti-redundansi via similarity threshold
        • Output kronologis (sorted by original position)
        │
        ▼ selected_indices
        │
    OUTPUT RINGKASAN

    ─── Parameters ──────────────────────────────────────────
    text             : str   — teks notulensi input
    compression_rate : float — proporsi kalimat yang dipertahankan [0.1, 0.8]
    lambda_mmr       : float — trade-off relevansi vs keberagaman MMR [0, 1]
    w1, w2, w3, w4   : float — bobot fitur (konten, posisi, panjang, keyword)
    n_components     : int   — jumlah komponen laten SVD
    expand_abbr      : bool  — ekspansi singkatan sebelum preprocessing

    ─── Returns ─────────────────────────────────────────────
    dict berisi semua artefak komputasi untuk ditampilkan di GUI:
      original_sentences  : kalimat asli
      processed_sentences : token bersih per kalimat
      summary_text        : ringkasan akhir (string)
      summary_sentences   : list kalimat ringkasan
      selected_indices    : indeks kalimat terpilih
      scoring_df          : DataFrame semua skor (untuk tab Proses)
      tfidf_df            : DataFrame matriks TF-IDF (untuk heatmap)
      tfidf_matrix        : raw numpy array TF-IDF
      sentence_vectors    : matriks U ternormalisasi
      singular_values     : nilai singular Σ
      explained_var       : varians terjelas per komponen
      vocabulary          : daftar term
      sc_tfidf, sc_lsa, sc_pos, sc_len, sc_kw : skor individu
      n_components_used   : k aktual yang digunakan SVD

    Jika teks terlalu pendek (< 2 kalimat):
      → {"error": "...pesan error..."}
    """

    # ── [1] Preprocessing ────────────────────────────────────
    prep  = Preprocessor(expand_abbr=expand_abbr)
    data  = prep.process(text)
    orig  = data["original_sentences"]
    proc  = data["processed_sentences"]
    n_s   = len(orig)

    # Validasi: butuh minimal 2 kalimat untuk SVD
    if n_s < 2:
        return {
            "error": (
                f"Teks terlalu pendek — hanya {n_s} kalimat terdeteksi. "
                "Minimal diperlukan 2 kalimat. "
                "Periksa apakah teks sudah lengkap dan memiliki tanda titik."
            )
        }

    # ── [2] Feature Scoring ──────────────────────────────────
    scorer    = FeatureScorer()
    tfidf_mat = scorer.build_tfidf_matrix(proc)
    sc_tfidf  = scorer.score_tfidf(tfidf_mat)
    sc_pos    = scorer.score_position(n_s)
    sc_len    = scorer.score_length(proc)
    sc_kw     = scorer.score_keyword(orig)

    # ── [3] LSA via SVD ──────────────────────────────────────
    lsa       = LSASummarizer(n_components=n_components)
    sent_vecs = lsa.fit_transform(tfidf_mat)
    sc_lsa    = lsa.score_lsa()

    # ── [4] Fusion Scoring ───────────────────────────────────
    fusion_sc = FusionScorer.fuse(
        sc_tfidf, sc_lsa, sc_pos, sc_len, sc_kw,
        w1=w1, w2=w2, w3=w3, w4=w4,
    )

    # ── [5] MMR Selection ────────────────────────────────────
    n_target   = max(1, round(n_s * compression_rate))
    sel_sorted = MMROptimizer().select(
        sentence_vectors=sent_vecs,
        fusion_scores=fusion_sc,
        n_select=n_target,
        lambda_param=lambda_mmr,
        sim_threshold=0.85,
    )

    # ── Susun Teks Ringkasan ─────────────────────────────────
    # Gabung kalimat terpilih dengan spasi — urutan sudah kronologis
    summary = " ".join(orig[i] for i in sel_sorted)

    # ── Buat DataFrame Skor (untuk Tab Proses di GUI) ────────
    scoring_df = pd.DataFrame({
        "No":        range(1, n_s + 1),
        "Kalimat":   [
            s[:60] + "..." if len(s) > 60 else s
            for s in orig
        ],
        "TF-IDF":    sc_tfidf.round(4),
        "LSA":       sc_lsa.round(4),
        "Posisi":    sc_pos.round(4),
        "Panjang":   sc_len.round(4),
        "Keyword":   sc_kw.round(4),
        "Fusi":      fusion_sc.round(4),
        "Dipilih":   [
            "✅" if i in sel_sorted else ""
            for i in range(n_s)
        ],
    })

    # ── Buat DataFrame TF-IDF (untuk Heatmap di GUI) ─────────
    tfidf_df = pd.DataFrame(
        tfidf_mat,
        index=[f"K{i+1}" for i in range(n_s)],
        columns=scorer.vocabulary,
    )

    # ── Return Semua Artefak ─────────────────────────────────
    return {
        # Teks
        "original_sentences":  orig,
        "processed_sentences": proc,
        "summary_text":        summary,
        "summary_sentences":   [orig[i] for i in sel_sorted],
        "selected_indices":    sel_sorted,

        # DataFrame untuk GUI
        "scoring_df":          scoring_df,
        "tfidf_df":            tfidf_df,

        # Array numpy mentah
        "tfidf_matrix":        tfidf_mat,
        "sentence_vectors":    sent_vecs,
        "fusion_scores":       fusion_sc,
        "singular_values":     lsa.singular_values,
        "explained_var":       lsa.explained_var,
        "vocabulary":          scorer.vocabulary,

        # Skor individu (untuk visualisasi)
        "sc_tfidf":            sc_tfidf,
        "sc_lsa":              sc_lsa,
        "sc_pos":              sc_pos,
        "sc_len":              sc_len,
        "sc_kw":               sc_kw,

        # Metadata teknis
        "n_components_used":   sent_vecs.shape[1],
    }

# ============================================================
# SISTEM PERINGKASAN NOTULENSI FORUM REFORMASI BIROKRASI
# LPP TVRI — Part 3 dari 3
# ============================================================
# Isi Part 3:
#   - CSS styling lengkap (hero, cards, summary box, dll.)
#   - DEFAULT_TEXT  : contoh teks notulensi untuk demo
#   - Fungsi main() : Streamlit GUI dengan 4 tab:
#       Tab 1 — Input & PDF Upload
#       Tab 2 — Proses & Pemodelan (White-Box Transparency)
#       Tab 3 — Hasil Ringkasan + Download
#       Tab 4 — Evaluasi ROUGE
#
# Part 1: Import, PDFExtractor, Preprocessor, FeatureScorer
# Part 2: LSASummarizer, FusionScorer, MMROptimizer, Evaluator, run_pipeline
#
# CARA MENJALANKAN:
#   1. Gabung Part 1 + Part 2 + Part 3 menjadi satu file app.py
#   2. pip install streamlit pymupdf PySastrawi scikit-learn
#              plotly pandas numpy
#   3. streamlit run app.py
# ============================================================


# ============================================================
# SECTION 10 — CSS STYLING
# ============================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Global ────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}
code, pre, .mono {
    font-family: 'JetBrains Mono', monospace;
}

/* ── Hero Banner ───────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #060c1a 0%, #0a2340 55%, #0d3b6e 100%);
    border-radius: 18px;
    padding: 28px 38px;
    margin-bottom: 22px;
    border: 1px solid rgba(255, 255, 255, 0.07);
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: "";
    position: absolute;
    top: -60px; right: -60px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(30,144,255,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero h1 {
    color: #deeeff;
    margin: 0 0 5px;
    font-size: 1.55rem;
    font-weight: 800;
}
.hero p {
    color: #7aadd4;
    margin: 0;
    font-size: 0.87rem;
    line-height: 1.7;
}
.badge {
    display: inline-block;
    background: rgba(30,144,255,0.14);
    color: #5ba4f5;
    border: 1px solid rgba(30,144,255,0.3);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.70rem;
    font-weight: 700;
    margin: 6px 3px 0 0;
    letter-spacing: .05em;
}

/* ── Info Box ──────────────────────────────────────────── */
.info-box {
    background: #f0f9ff;
    border: 1px solid #7dd3fc;
    border-left: 4px solid #0ea5e9;
    border-radius: 8px;
    padding: 11px 15px;
    font-size: 0.83rem;
    color: #075985;
    margin: 7px 0;
    line-height: 1.6;
}

/* ── Upload Hint ───────────────────────────────────────── */
.upload-hint {
    border: 2px dashed #2b6cb0;
    border-radius: 12px;
    background: linear-gradient(135deg, #ebf8ff, #e0f0ff);
    padding: 18px 22px;
    text-align: center;
    margin: 6px 0 12px;
}
.upload-hint h3 {
    color: #1a4a7a;
    margin: 0 0 4px;
    font-size: 1.0rem;
}
.upload-hint p {
    color: #4a7fa8;
    margin: 0;
    font-size: 0.82rem;
}

/* ── File Cards ────────────────────────────────────────── */
.fcard {
    background: #f0fff4;
    border: 1px solid #9ae6b4;
    border-left: 4px solid #38a169;
    border-radius: 8px;
    padding: 8px 13px;
    margin: 3px 0;
    font-size: 0.82rem;
}
.fcard-warn {
    background: #fffbeb;
    border: 1px solid #f6ad55;
    border-left: 4px solid #ed8936;
    border-radius: 8px;
    padding: 8px 13px;
    margin: 3px 0;
    font-size: 0.82rem;
    color: #744210;
}

/* ── Metadata Card ─────────────────────────────────────── */
.mcard {
    background: #f0f7ff;
    border: 1px solid #bee3f8;
    border-radius: 8px;
    padding: 9px 14px;
    margin: 3px 0;
    font-size: 0.81rem;
    line-height: 1.75;
}
.mcard .k {
    font-weight: 700;
    color: #2b6cb0;
}

/* ── Summary Box ───────────────────────────────────────── */
.summary-box {
    background: linear-gradient(135deg, #f0fdf4, #dcfce7);
    border: 1px solid #86efac;
    border-left: 5px solid #22c55e;
    border-radius: 12px;
    padding: 20px 26px;
    line-height: 1.95;
    font-size: 1.0rem;
    color: #14532d;
}

/* ── Sentence Highlight ────────────────────────────────── */
.sent-sel {
    background: linear-gradient(90deg, #fefce8, #fef3c7);
    border-left: 4px solid #f59e0b;
    border-radius: 0 8px 8px 0;
    padding: 10px 16px;
    margin: 4px 0;
    font-size: 0.91rem;
    color: #78350f;
}
.sent-no {
    padding: 5px 16px;
    margin: 2px 0;
    font-size: 0.87rem;
    color: #94a3b8;
    border-left: 2px solid #e2e8f0;
}
.slabel {
    font-size: .68rem;
    font-weight: 800;
    letter-spacing: .07em;
    color: #a16207;
    margin-right: 5px;
}

/* ── Formula Box ───────────────────────────────────────── */
.fbox {
    background: #06080f;
    border-radius: 8px;
    padding: 10px 15px;
    margin: 7px 0;
    border: 1px solid #162032;
}
.fbox code {
    color: #7dd3fc;
    font-size: 0.81rem;
    font-family: 'JetBrains Mono', monospace;
}

/* ── ROUGE Result Card ─────────────────────────────────── */
.rcard {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-top: 4px solid;
    border-radius: 10px;
    padding: 15px;
    text-align: center;
}

/* ── Sidebar & Tabs ────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background-color: #1e293b !important; /* Biru tua yang elegan */
    border-right: 1px solid #334155 !important;
}
section[data-testid="stSidebar"] * {
    color: #f8fafc !important; /* Memaksa semua teks jadi terang */
}
section[data-testid="stSidebar"] h1, 
section[data-testid="stSidebar"] h2, 
section[data-testid="stSidebar"] h3, 
section[data-testid="stSidebar"] h4, 
section[data-testid="stSidebar"] p, 
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] {
    color: #f8fafc !important; /* Warna teks menjadi putih/terang */
}
.stTabs [data-baseweb="tab"] {
    padding: 7px 18px;
    border-radius: 8px !important;
    font-weight: 600;
    font-size: 0.85rem;
}
</style>
"""

# ============================================================
# SECTION 11 — DEFAULT TEXT (Contoh Demo)
# ============================================================
DEFAULT_TEXT = """Forum Reformasi Birokrasi LPP TVRI membahas desain besar reformasi birokrasi nasional dan transformasi digital lembaga.
Direktur Umum menyampaikan bahwa lembaga harus mentransformasi diri menjadi organisasi berbasis data dan teknologi digital.
Kepala SPI memaparkan dua topik utama terkait Zona Integritas dan persyaratan satuan kerja yang akan diajukan sebagai ZI."""

# ============================================================
# SECTION 12 — STREAMLIT MAIN
# ============================================================
def main():
    # ── Konfigurasi Halaman ───────────────────────────────────
    st.set_page_config(
        page_title="Peringkas Notulensi  – LPP TVRI",
        page_icon="📋",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    # ── Hero Banner ───────────────────────────────────────────
    st.markdown("""
    <div class="hero">
        <h1>📋 Peringkas Notulensi Forum Reformasi Birokrasi — LPP TVRI</h1>
        <p>Upload PDF atau paste teks notulensi ➜ BM25 TF-IDF + Keyword Boost
        + LSA (SVD) + Fusion Scoring ➜ Seleksi MMR ➜ Evaluasi ROUGE</p>
        <span class="badge">PDF Langsung</span>
        <span class="badge">BM25 TF-IDF</span>
        <span class="badge">Keyword Boost</span>
        <span class="badge">LSA SVD</span>
        <span class="badge">MMR Anti-Redundansi</span>
        <span class="badge">Sastrawi NLP</span>
        <span class="badge">ROUGE Eval</span>
        <span class="badge">White-Box 100%</span>
    </div>
    """, unsafe_allow_html=True)

    # ============================================================
    # SIDEBAR — Parameter Sistem
    # ============================================================
    with st.sidebar:
        st.markdown("## ⚙️ Parameter Sistem")
        st.markdown("---")

        # Kompresi
        st.markdown("#### 📏 Kompresi Dokumen")
        compression_rate = st.slider(
            "Compression Rate (%)",
            min_value=1, max_value=80, value=10, step=1,
            help="Persentase kalimat yang dipertahankan dalam ringkasan",
        ) / 100.0

        # MMR
        st.markdown("#### 🎯 MMR — Trade-off Relevansi vs Keberagaman")
        lambda_mmr = st.slider(
            "λ (Lambda)",
            min_value=0.0, max_value=1.0, value=0.80, step=0.05,
            help="λ→1 prioritaskan relevansi | λ→0 prioritaskan keberagaman",
        )
        if lambda_mmr > 0.6:
            st.caption("🟢 Relevansi tinggi")
        elif lambda_mmr >= 0.4:
            st.caption("🟡 Seimbang")
        else:
            st.caption("🔵 Keberagaman tinggi")

        # Bobot Fusi
        st.markdown("#### ⚖️ Bobot Fusi (4 Fitur)")
        w1 = st.slider("W1 — Konten (TF-IDF + LSA)", 0.0, 1.0, 1.00, 0.05)
        w2 = st.slider("W2 — Posisi Kalimat",         0.0, 1.0, 0.30, 0.05)
        w3 = st.slider("W3 — Panjang Kalimat",         0.0, 1.0, 0.30, 0.05)
        w4 = st.slider("W4 — Keyword Notulensi",       0.0, 1.0, 1.00, 0.05)
        tot = w1 + w2 + w3 + w4
        if tot > 0:
            st.info(
                f"Bobot ternormalisasi:\n"
                f"Ŵ1={w1/tot:.2f}  Ŵ2={w2/tot:.2f}  "
                f"Ŵ3={w3/tot:.2f}  Ŵ4={w4/tot:.2f}"
            )

        # LSA
        st.markdown("#### 🧮 LSA — Komponen SVD")
        n_comp = st.slider(
            "k (komponen laten)",
            min_value=1, max_value=10, value=7,
            help="Jumlah topik semantik yang ditangkap SVD",
        )

        # Opsi Lanjutan
        st.markdown("#### ⚙️ Opsi Lanjutan")
        expand_abbr = st.checkbox(
            "Ekspansi singkatan (RB, BPK, BMN, dll.)",
            value=True,
            help="Ubah singkatan ke bentuk panjang sebelum diproses "
                 "agar TF-IDF lebih akurat",
        )

        st.markdown("---")
        st.markdown("#### 🗄️ Database PostgreSQL")
        if DB_AVAILABLE:
            auto_save = st.checkbox(
                "💾 Simpan otomatis setelah peringkasan",
                value=False,
                help="Setiap kali pipeline selesai, data langsung disimpan ke database",
            )
            st.session_state["auto_save_db"] = auto_save
            if st.button("🔌 Test Koneksi DB", use_container_width=True):
                if test_koneksi():
                    st.success("✅ Terhubung!")
                else:
                    st.error("❌ Gagal. Cek db_helper.py")
        else:
            st.caption("⚠️ db_helper.py tidak ditemukan")

        st.markdown("---")
        st.markdown("""
        <div style="font-size:.71rem; color:#94a3b8; line-height:1.9">
        <b>Alur Pipeline:</b><br>
        PDF → Ekstrak Teks Langsung<br>
        ↓ Normalisasi & Expand Abbr<br>
        ↓ Sastrawi Stop + Stem<br>
        ↓ BM25 TF-IDF Matrix<br>
        ↓ Keyword Boost Score<br>
        ↓ SVD Truncated (LSA)<br>
        ↓ Weighted Fusion (4 fitur)<br>
        ↓ MMR + Sim Threshold<br>
        ↓ Output Kronologis<br>
        ↓ ROUGE Evaluation<br>
        <br><b>Tanpa OCR · Tanpa DL ✓</b>
        </div>
        """, unsafe_allow_html=True)

    # ============================================================
    # TABS
    # ============================================================
    tab_input, tab_proses, tab_output, tab_eval, tab_db = st.tabs([
        "📥  Input & PDF Upload",
        "🔬  Proses & Pemodelan",
        "📝  Hasil Ringkasan",
        "📊  Evaluasi ROUGE",
        "🗄️  Riwayat & Database",
    ])

    # ============================================================
    # TAB 1 — INPUT & PDF UPLOAD
    # ============================================================
    with tab_input:
        st.subheader("📥 Input Dokumen Notulensi")

        # Info cara penggunaan
        st.markdown("""
        <div class="info-box">
        ℹ️ <b>Panduan Input:</b><br>
        &nbsp;&nbsp;• <b>Mode PDF:</b> Upload file PDF — teks diekstrak langsung
        (tanpa OCR). Halaman foto dokumentasi rapat otomatis dilewati.<br>
        &nbsp;&nbsp;• <b>Mode Teks Manual:</b> Untuk PDF scan/gambar, buka di
        Adobe Reader → Ctrl+A → Ctrl+C → paste di area teks di bawah.
        </div>
        """, unsafe_allow_html=True)

        # Pilih mode input
        # key="mode_input_radio" memungkinkan tombol "Muat Teks ke Tab
        # Input" di Tab Riwayat memaksa radio ini ke mode "Upload File
        # PDF" via session_state, agar teks yang dimuat dari Pustaka
        # Database langsung terlihat tanpa pengguna ganti mode manual.
        mode = st.radio(
            "Mode input:",
            ["📄  Upload File PDF", "✍️  Teks Manual", "🗄️  Dari Database"],
            horizontal=True,
            key="mode_input_radio",
        )
        st.markdown("---")

        # Teks manual tidak pernah terkait dengan dokumen tersimpan —
        # putuskan tautan id_dokumen lama (jika ada) agar penyimpanan
        # berikutnya membuat baris tb_dokumen baru, bukan menimpa
        # dokumen lain secara tidak sengaja.
        if "Teks Manual" in mode:
            st.session_state.pop("pdf_doc_id", None)

        input_text   = ""
        doc_metadata = []

        # ── MODE PDF ─────────────────────────────────────────
        if "PDF" in mode:
            st.markdown("""
            <div class="upload-hint">
                <h3>📎 Upload File PDF Notulensi LPP TVRI</h3>
                <p>Mendukung satu atau beberapa file sekaligus untuk
                ringkasan lintas sesi forum.<br>
                Ekstraksi teks langsung — cepat, tanpa proses OCR.</p>
            </div>
            """, unsafe_allow_html=True)

            uploaded = st.file_uploader(
                "Pilih file PDF",
                type=["pdf"],
                accept_multiple_files=True,
                label_visibility="collapsed",
            )

            if uploaded:
                st.markdown(f"**{len(uploaded)} file dipilih:**")
                total_pages = 0
                for uf in uploaded:
                    kb = len(uf.getvalue()) / 1024
                    try:
                        doc_tmp = fitz.open(
                            stream=uf.getvalue(), filetype="pdf"
                        )
                        pg = doc_tmp.page_count
                        doc_tmp.close()
                    except Exception:
                        pg = "?"
                    if isinstance(pg, int):
                        total_pages += pg
                    st.markdown(
                        f'<div class="fcard">📄 <b>{uf.name}</b>'
                        f'<span style="color:#555; font-size:.77rem; margin-left:10px">'
                        f'{kb:.1f} KB · {pg} halaman</span></div>',
                        unsafe_allow_html=True,
                    )

                col_btn, col_inf = st.columns([1, 2])
                with col_btn:
                    do_extract = st.button(
                        "📄  Ekstrak Teks PDF",
                        type="primary",
                        use_container_width=True,
                    )
                with col_inf:
                    st.info(
                        f"Total **{total_pages} halaman** dari "
                        f"{len(uploaded)} file — ekstraksi instan tanpa OCR"
                    )

                if do_extract:
                    # Ekstraksi baru -> putuskan tautan ke id_dokumen lama
                    # (jika ada) supaya hasil ekstraksi ini disimpan
                    # sebagai baris tb_dokumen yang baru.
                    st.session_state.pop("pdf_doc_id", None)
                    extractor = PDFExtractor()
                    files     = [(uf.name, uf.getvalue()) for uf in uploaded]
                    prog_bar  = st.progress(0.0, text="Memulai ekstraksi...")
                    prog_text = st.empty()

                    def _progress_cb(frac: float, msg: str):
                        prog_bar.progress(min(frac, 0.97), text=msg)
                        prog_text.caption(msg)

                    # Proses satu atau beberapa file
                    if len(files) == 1:
                        fname, data    = files[0]
                        text, meta, info = extractor.extract(
                            data, progress_cb=_progress_cb
                        )
                        meta["filename"] = fname
                        doc_metadata     = [meta]
                        all_infos        = [{**info, "filename": fname}]
                    else:
                        text, doc_metadata, all_infos = extractor.extract_multi(
                            files, progress_cb=_progress_cb
                        )

                    prog_bar.progress(1.0, text="Selesai!")
                    prog_text.empty()

                    # Laporan hasil per file
                    st.markdown("**Hasil ekstraksi per file:**")
                    for info_item in all_infos:
                        fn     = info_item.get("filename", "?")
                        n_txt  = info_item.get("n_text_pages", 0)
                        n_img  = info_item.get("n_image_pages", 0)
                        n_ch   = info_item.get("n_chars", 0)
                        is_img = info_item.get("is_image_based", False)

                        if is_img:
                            st.markdown(
                                f'<div class="fcard-warn">⚠️ <b>{fn}</b>: '
                                f'Seluruh halaman berbasis gambar/scan — '
                                f'gunakan mode Teks Manual (copy-paste).</div>',
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f'<div class="fcard">✅ <b>{fn}</b>: '
                                f'{n_txt} hal. teks ({n_ch:,} karakter) · '
                                f'{n_img} hal. gambar dilewati</div>',
                                unsafe_allow_html=True,
                            )

                    # Simpan ke session_state
                    if len(text.strip()) < 100:
                        st.error(
                            "❌ Tidak ada teks yang berhasil diekstrak dari semua file. "
                            "PDF Anda kemungkinan sepenuhnya berbasis gambar/scan. "
                            "Silakan gunakan mode **Teks Manual**."
                        )
                    else:
                        st.session_state["pdf_text"]     = text
                        st.session_state["doc_metadata"] = doc_metadata
                        n_words = len(text.split())
                        st.success(
                            f"✅ Berhasil diekstrak: ~{n_words:,} kata "
                            f"dari {len(files)} file."
                        )

                        # ── Simpan hasil ekstraksi ke database SEGERA ────
                        # Tidak menunggu "Jalankan Peringkasan" — teks ini
                        # langsung tersedia di Tab 🗄️ Riwayat & Database
                        # → Pustaka Dokumen, sehingga dapat diringkas
                        # kapan saja (sekarang atau di sesi berikutnya)
                        # tanpa perlu mengunggah ulang PDF.
                        if DB_AVAILABLE:
                            meta_save = doc_metadata[0] if doc_metadata else {}
                            n_kal = len(
                                Preprocessor(expand_abbr=expand_abbr)
                                .sentence_tokenize(text)
                            )
                            id_dok_ext = simpan_dokumen(
                                nama_file=meta_save.get("filename", files[0][0]),
                                tanggal_notulensi=meta_save.get("tanggal"),
                                tema=meta_save.get("tema"),
                                pembicara=meta_save.get("pembicara"),
                                moderator=meta_save.get("moderator"),
                                jumlah_kalimat=n_kal,
                                jumlah_kata=n_words,
                                teks_lengkap=text,
                            )
                            if id_dok_ext > 0:
                                st.session_state["pdf_doc_id"] = id_dok_ext
                                st.info(
                                    f"💾 Teks hasil ekstraksi tersimpan ke "
                                    f"database (ID Dokumen: {id_dok_ext}). "
                                    f"Tersedia di Tab 🗄️ Riwayat & Database → "
                                    f"Pustaka Dokumen meski peringkasan belum "
                                    f"dijalankan."
                                )

            # ── Tampilkan hasil ekstraksi / dokumen dari Pustaka DB ────
            # Diletakkan DI LUAR blok `if uploaded:` agar dokumen yang
            # dimuat dari Pustaka Database (Tab Riwayat) juga tampil
            # tanpa perlu mengunggah file PDF baru.
            if "pdf_text" in st.session_state:
                input_text   = st.session_state["pdf_text"]
                doc_metadata = st.session_state.get("doc_metadata", [])

                if st.session_state.get("loaded_from_db"):
                    col_src, col_clear = st.columns([5, 1])
                    with col_src:
                        src_name = doc_metadata[0].get("filename", "—") if doc_metadata else "—"
                        st.success(
                            f"📚 Teks dimuat dari **Pustaka Database**: {src_name} — "
                            "atur parameter di sidebar lalu jalankan peringkasan ulang."
                        )
                    with col_clear:
                        if st.button("🗑️ Hapus", use_container_width=True,
                                     help="Hapus teks yang dimuat dari database"):
                            st.session_state.pop("pdf_text", None)
                            st.session_state.pop("doc_metadata", None)
                            st.session_state.pop("loaded_from_db", None)
                            st.session_state.pop("pdf_doc_id", None)
                            st.rerun()

                # Tampilkan kartu metadata
                if doc_metadata:
                    st.markdown("##### 📋 Metadata Dokumen Terdeteksi")
                    n_cols = min(len(doc_metadata), 3)
                    meta_cols = st.columns(n_cols)
                    for idx_m, meta in enumerate(doc_metadata):
                        with meta_cols[idx_m % n_cols]:
                            tgl  = meta.get("tanggal", "—")
                            tema = (meta.get("tema") or "—")[:68]
                            pb   = (meta.get("pembicara") or "—")[:55]
                            fn   = meta.get("filename", "—")
                            st.markdown(
                                f'<div class="mcard">'
                                f'<span class="k">📅 Tanggal:</span> {tgl}<br>'
                                f'<span class="k">📌 Tema:</span> {tema}<br>'
                                f'<span class="k">👤 Pembicara:</span> {pb}<br>'
                                f'<span class="k">📎 File:</span> {fn}'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                # Preview teks hasil ekstraksi
                with st.expander("📖 Preview teks hasil ekstraksi PDF", expanded=False):
                    st.text_area(
                        "Preview teks hasil ekstraksi PDF",
                        value=input_text,
                        height=250,
                        disabled=True,
                        label_visibility="collapsed",
                    )
                    st.caption(
                        f"~{len(input_text.split()):,} kata · "
                        f"{len(input_text):,} karakter"
                    )

        # ── MODE TEKS MANUAL ─────────────────────────────────
        elif "Teks Manual" in mode:
            st.markdown("""
            <div class="info-box">
            💡 <b>Tip copy-paste dari PDF scan:</b> Buka PDF di browser
            (Chrome/Edge) atau Adobe Reader → Ctrl+A (pilih semua) →
            Ctrl+C (copy) → Ctrl+V di area teks bawah ini.
            </div>
            """, unsafe_allow_html=True)

            input_text = st.text_area(
                "Teks Notulensi Bahasa Indonesia:",
                value=DEFAULT_TEXT,
                height=380,
                help=(
                    "Paste isi notulensi di sini. "
                    "Satu kalimat per baris atau paragraf biasa."
                ),
            )

        # ── MODE DARI DATABASE ────────────────────────────────
        else:
            st.markdown("""
            <div class="info-box">
            🗄️ <b>Mode Database:</b> Pilih salah satu dokumen yang
            teks lengkapnya sudah pernah diekstrak &amp; tersimpan
            sebelumnya — tanpa perlu mengunggah ulang PDF. Hasil
            peringkasan akan tertaut ke dokumen yang sama (tidak
            membuat data dokumen ganda).
            </div>
            """, unsafe_allow_html=True)

            if not DB_AVAILABLE:
                st.warning(
                    "⚠️ Modul db_helper.py tidak ditemukan, sehingga "
                    "Pustaka Dokumen tidak dapat diakses. Gunakan mode "
                    "**Upload File PDF** atau **Teks Manual**."
                )
            else:
                daftar_dok = ambil_daftar_dokumen()
                if not daftar_dok:
                    st.info(
                        "Belum ada dokumen tersimpan. Ekstrak PDF pada "
                        "mode **📄 Upload File PDF** terlebih dahulu — "
                        "teksnya akan otomatis tersimpan dan muncul "
                        "sebagai pilihan di sini."
                    )
                else:
                    opsi_map = {}
                    for d in daftar_dok:
                        tgl  = d["tanggal_notulensi"] or "—"
                        tema = (d["tema"] or "(tanpa tema)")[:60]
                        label = f"[{d['id_dokumen']}] {d['nama_file']} — {tema} ({tgl})"
                        opsi_map[label] = d["id_dokumen"]

                    pilihan = st.selectbox(
                        "Pilih dokumen tersimpan:",
                        list(opsi_map.keys()),
                        key="pilih_dok_tab1",
                    )
                    id_pilih = opsi_map[pilihan]
                    dok = ambil_teks_dokumen(id_pilih)

                    if dok and dok.get("teks_lengkap"):
                        input_text   = dok["teks_lengkap"]
                        doc_metadata = [{
                            "filename":  dok.get("nama_file"),
                            "tanggal":   str(dok["tanggal_notulensi"])
                                         if dok.get("tanggal_notulensi") else None,
                            "tema":      dok.get("tema"),
                            "pembicara": dok.get("pembicara"),
                            "moderator": dok.get("moderator"),
                        }]
                        st.session_state["doc_metadata"] = doc_metadata
                        # Tautkan ke id_dokumen ini — peringkasan yang
                        # disimpan nanti akan memakai baris tb_dokumen
                        # yang SAMA (tidak duplikat teks_lengkap).
                        st.session_state["pdf_doc_id"] = id_pilih

                        st.markdown("##### 📋 Metadata Dokumen")
                        st.markdown(
                            f'<div class="mcard">'
                            f'<span class="k">📅 Tanggal:</span> '
                            f'{dok.get("tanggal_notulensi") or "—"}<br>'
                            f'<span class="k">📌 Tema:</span> '
                            f'{(dok.get("tema") or "—")[:90]}<br>'
                            f'<span class="k">👤 Pembicara:</span> '
                            f'{(dok.get("pembicara") or "—")[:60]}<br>'
                            f'<span class="k">📎 File:</span> '
                            f'{dok.get("nama_file","—")}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                        with st.expander("📖 Preview teks dokumen", expanded=False):
                            st.text_area(
                                "Preview teks dokumen tersimpan",
                                value=input_text,
                                height=250,
                                disabled=True,
                                label_visibility="collapsed",
                                key=f"preview_db_tab1_{id_pilih}",
                            )
                            st.caption(
                                f"~{len(input_text.split()):,} kata · "
                                f"{len(input_text):,} karakter"
                            )
                    else:
                        st.error("❌ Teks dokumen tidak ditemukan atau kosong.")

        # ── Info Statistik Dokumen ────────────────────────────
        if input_text.strip():
            # Preview jumlah kalimat & target (tanpa jalankan pipeline penuh)
            preview_sents = [
                s.strip()
                for s in re.split(r"(?<=[.!?])\s+|\n{2,}", input_text.strip())
                if len(s.strip()) >= 20
            ]
            n_target_preview = max(1, round(len(preview_sents) * compression_rate))
            st.markdown("---")
            ci1, ci2, ci3, ci4 = st.columns(4)
            ci1.metric("Kalimat Terdeteksi", len(preview_sents))
            ci2.metric("Jumlah Kata",        f"{len(input_text.split()):,}")
            ci3.metric("Target Ringkasan",   f"{n_target_preview} kalimat")
            ci4.metric("Compression Rate",   f"{compression_rate*100:.0f}%")

        # ── Tombol Jalankan ───────────────────────────────────
        st.markdown("")
        _, col_center, _ = st.columns([1, 2, 1])
        with col_center:
            run_btn = st.button(
                "🚀  Jalankan Peringkasan",
                type="primary",
                use_container_width=True,
            )

        if run_btn:
            if not input_text.strip():
                st.warning(
                    "⚠️ Belum ada teks. "
                    "Upload PDF atau isi teks manual terlebih dahulu."
                )
            else:
                with st.spinner(
                    "⚙️ Memproses... (BM25 + Sastrawi + SVD + Keyword + MMR)"
                ):
                    result = run_pipeline(
                        text             = input_text,
                        compression_rate = compression_rate,
                        lambda_mmr       = lambda_mmr,
                        w1=w1, w2=w2, w3=w3, w4=w4,
                        n_components     = n_comp,
                        expand_abbr      = expand_abbr,
                    )

                if "error" in result:
                    st.error(result["error"])
                else:
                    st.session_state["result"]     = result
                    st.session_state["input_text"] = input_text
                    # Reset hasil ROUGE & ID database lama —
                    # ringkasan baru perlu dievaluasi/disimpan ulang
                    st.session_state.pop("rouge_scores", None)
                    st.session_state.pop("rouge_ref_text", None)
                    st.session_state.pop("db_ids", None)
                    n_sel = len(result["selected_indices"])
                    n_tot = len(result["original_sentences"])
                    st.success(
                        f"✅ Selesai! Dipilih **{n_sel}** dari {n_tot} kalimat. "
                        "Lihat tab **Proses**, **Ringkasan**, dan **Evaluasi** →"
                    )

                    # ── Simpan otomatis ke database ──────────
                    if DB_AVAILABLE and st.session_state.get("auto_save_db", False):
                        meta  = st.session_state.get("doc_metadata", [{}])[0]
                        fname = meta.get("filename", "input_manual.txt")
                        # Pakai id_dokumen dari hasil ekstraksi PDF (atau
                        # dokumen yang dimuat dari Pustaka) jika sudah
                        # tersedia — hindari baris tb_dokumen duplikat
                        # untuk teks yang sama.
                        id_dok = st.session_state.get("pdf_doc_id")
                        if not id_dok or id_dok <= 0:
                            id_dok = simpan_dokumen(
                                nama_file=fname,
                                tanggal_notulensi=meta.get("tanggal"),
                                tema=meta.get("tema"),
                                pembicara=meta.get("pembicara"),
                                moderator=meta.get("moderator"),
                                jumlah_kalimat=n_tot,
                                jumlah_kata=len(input_text.split()),
                                teks_lengkap=input_text,
                            )
                            st.session_state["pdf_doc_id"] = id_dok
                        id_par = simpan_parameter(
                            compression_rate=compression_rate,
                            lambda_mmr=lambda_mmr,
                            w1=w1, w2=w2, w3=w3, w4=w4,
                            k_svd=n_comp,
                        )
                        simpan_kalimat_batch(
                            id_dok, result["scoring_df"],
                            original_sentences=result["original_sentences"],
                            processed_sentences=result["processed_sentences"],
                        )
                        id_ring = simpan_ringkasan(id_dok, id_par, result, input_text)
                        st.session_state["db_ids"] = {
                            "id_dokumen": id_dok,
                            "id_param": id_par,
                            "id_ringkasan": id_ring,
                        }
                        if id_ring > 0:
                            st.info(f"💾 Tersimpan otomatis ke database (ID Ringkasan: {id_ring})")

    # ============================================================
    # TAB 2 — PROSES & PEMODELAN (White-Box Transparency)
    # ============================================================
    with tab_proses:
        st.subheader("🔬 Transparansi White-Box — Alur Komputasi Lengkap")

        if "result" not in st.session_state:
            st.info("ℹ️ Jalankan pipeline di tab **Input** terlebih dahulu.")
            st.stop()

        R   = st.session_state["result"]
        n_s = len(R["original_sentences"])
        n_t = len(R["vocabulary"])

        # ── Tahap 1: Preprocessing ────────────────────────────
        with st.expander("📋 Tahap 1 — Preprocessing Sastrawi", expanded=True):
            st.markdown(
                '<div class="fbox"><code>'
                'Input → Expand Abbr (RB→"reformasi birokrasi", dll.) '
                '→ Case Folding → Stopword Removal (Sastrawi) '
                '→ Stemming (Sastrawi) → Token Bersih'
                '</code></div>',
                unsafe_allow_html=True,
            )
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Kalimat Asli**")
                for i, s in enumerate(R["original_sentences"]):
                    preview = s[:85] + ("..." if len(s) > 85 else "")
                    st.markdown(f"`K{i+1}` {preview}")
            with col_b:
                st.markdown("**Token Setelah Preprocessing**")
                for i, toks in enumerate(R["processed_sentences"]):
                    pv = " | ".join(toks[:7])
                    suffix = "..." if len(toks) > 7 else ""
                    st.markdown(f"`K{i+1}` `{pv}{suffix}`")
            st.caption(f"📊 {n_s} kalimat · {n_t} term unik dalam vocabulary")

        # ── Tahap 2: BM25 TF-IDF ─────────────────────────────
        with st.expander("📊 Tahap 2 — BM25 TF-IDF Matrix", expanded=True):
            st.markdown(
                '<div class="fbox"><code>'
                'TF_BM25(t,d) = freq*(k1+1) / (freq + k1*(1−b+b*dl/avgdl)) '
                '&nbsp;|&nbsp; k1=1.5, b=0.75<br>'
                'IDF(t) = log((N−df+0.5)/(df+0.5)+1) '
                '&nbsp;|&nbsp; Score(t,d) = TF_BM25 × IDF'
                '</code></div>',
                unsafe_allow_html=True,
            )
            m1, m2, m3 = st.columns(3)
            m1.metric("Baris (Kalimat)", n_s)
            m2.metric("Kolom (Term)",    n_t)
            m3.metric("Densitas Non-nol",
                      f"{(R['tfidf_matrix'] > 0).mean() * 100:.1f}%")

            # Tampilkan top-12 term berdasarkan total TF-IDF
            top_terms = (
                R["tfidf_df"].sum()
                .nlargest(min(12, n_t))
                .index.tolist()
            )
            st.dataframe(
                R["tfidf_df"][top_terms]
                  .style.background_gradient(cmap="YlOrRd", axis=None)
                  .format("{:.4f}"),
                use_container_width=True,
                height=min(55 + 36 * n_s, 360),
            )
            # Heatmap
            fig_heat = px.imshow(
                R["tfidf_df"][top_terms].values,
                x=top_terms,
                y=[f"K{i+1}" for i in range(n_s)],
                color_continuous_scale="YlOrRd",
                title="Heatmap BM25 TF-IDF (top-12 term)",
                labels={"x": "Term", "y": "Kalimat", "color": "Skor"},
            )
            fig_heat.update_layout(height=255, margin=dict(t=42, b=10))
            st.plotly_chart(fig_heat, use_container_width=True)

        # ── Tahap 3: LSA via SVD ──────────────────────────────
        with st.expander("🧮 Tahap 3 — LSA via Truncated SVD", expanded=True):
            ev_pct = R["explained_var"].sum() * 100 if R["explained_var"] is not None else 0
            st.markdown(
                '<div class="fbox"><code>'
                f'A = U × Σ × Vᵀ &nbsp;|&nbsp; '
                f'Skor_LSA(i) = ‖U[i]×Σ‖₂ &nbsp;|&nbsp; '
                f'Varians dijelaskan: {ev_pct:.1f}%'
                '</code></div>',
                unsafe_allow_html=True,
            )
            sc1, sc2 = st.columns(2)
            with sc1:
                st.metric("Dimensi Asli TF-IDF", f"{n_s} × {n_t}")
                st.metric(
                    "Dimensi Laten (U)",
                    f"{R['sentence_vectors'].shape[0]} × {R['sentence_vectors'].shape[1]}",
                )
                st.metric(
                    "Kompresi Dimensi",
                    f"{n_t} term → {R['n_components_used']} konsep",
                )
                if R["explained_var"] is not None:
                    st.metric("Total Varians Dijelaskan", f"{ev_pct:.1f}%")
            with sc2:
                sv = R["singular_values"]
                if sv is not None and len(sv) > 0:
                    fig_sv = go.Figure(go.Bar(
                        x=[f"Σ{i+1}" for i in range(len(sv))],
                        y=sv,
                        marker=dict(color=sv, colorscale="Blues", showscale=False),
                        text=[f"{v:.3f}" for v in sv],
                        textposition="outside",
                    ))
                    fig_sv.update_layout(
                        title="Singular Values — Kekuatan Konsep Laten",
                        xaxis_title="Komponen SVD",
                        yaxis_title="Nilai",
                        height=255,
                        margin=dict(t=42, b=10),
                    )
                    st.plotly_chart(fig_sv, use_container_width=True)

            # Matriks U ternormalisasi
            st.markdown("**Matriks U — Representasi Kalimat dalam Ruang Laten (L2-normalized)**")
            sv_df = pd.DataFrame(
                R["sentence_vectors"],
                index=[f"K{i+1}" for i in range(n_s)],
                columns=[f"Konsep_{j+1}" for j in range(R["sentence_vectors"].shape[1])],
            )
            st.dataframe(
                sv_df.style.background_gradient(cmap="RdBu", axis=None).format("{:.4f}"),
                use_container_width=True,
                height=min(55 + 36 * n_s, 300),
            )

        # ── Tahap 4: Keyword Score ────────────────────────────
        with st.expander("🔑 Tahap 4 — Keyword Boost Score (Domain Notulensi TVRI)", expanded=False):
            st.markdown(
                '<div class="fbox"><code>'
                'Keyword Score = min(jumlah_hits / 5.0 ,  1.0)  '
                '&nbsp;|&nbsp; Keyword: keputusan, sepakat, arahan, '
                'tindak lanjut, reformasi birokrasi, zona integritas, '
                'anggaran, digitalisasi, dll.'
                '</code></div>',
                unsafe_allow_html=True,
            )
            kw_df = pd.DataFrame({
                "No":            range(1, n_s + 1),
                "Kalimat":       [s[:75] + "..." if len(s) > 75 else s
                                  for s in R["original_sentences"]],
                "Keyword Score": R["sc_kw"].round(4),
                "Flag":          ["🔑 Penting" if v > 0.3 else ""
                                  for v in R["sc_kw"]],
            })
            st.dataframe(kw_df, use_container_width=True,
                         height=min(55 + 36 * n_s, 400))

        # ── Tahap 5 & 6: Fusion + MMR ─────────────────────────
        with st.expander("⚖️ Tahap 5 & 6 — Fusion Scoring + Seleksi MMR", expanded=True):
            st.markdown(
                '<div class="fbox"><code>'
                'Konten = α×TF-IDF + (1−α)×LSA  '
                '&nbsp;|&nbsp;  '
                'Fusi = Ŵ1·Konten + Ŵ2·Posisi + Ŵ3·Panjang + Ŵ4·Keyword<br>'
                'MMR(i) = λ·Rel(sᵢ,D) − (1−λ)·max Sim(sᵢ,sⱼ)  '
                '&nbsp;|&nbsp;  Sim = Cosine  '
                '&nbsp;|&nbsp;  Threshold redundansi = 0.85'
                '</code></div>',
                unsafe_allow_html=True,
            )

            # Style baris terpilih dengan warna hijau
            def _highlight_selected(row):
                if row["Dipilih"] == "✅":
                    return ["background-color: #dcfce7; font-weight: 600"] * len(row)
                return [""] * len(row)

            st.dataframe(
                R["scoring_df"]
                  .style.apply(_highlight_selected, axis=1)
                  .format({c: "{:.4f}" for c in
                           ["TF-IDF", "LSA", "Posisi", "Panjang", "Keyword", "Fusi"]}),
                use_container_width=True,
                height=min(65 + 38 * n_s, 500),
            )

            # Grouped bar chart semua skor
            df_s    = R["scoring_df"]
            fig_bar = go.Figure()
            bar_config = [
                ("TF-IDF",  "#3b82f6", 0.75),
                ("LSA",     "#8b5cf6", 0.75),
                ("Keyword", "#ec4899", 0.75),
                ("Fusi",    "#22c55e", 0.95),
            ]
            for col_name, color, opacity in bar_config:
                fig_bar.add_trace(go.Bar(
                    name=col_name,
                    x=df_s["No"].apply(lambda x: f"K{x}"),
                    y=df_s[col_name],
                    marker_color=color,
                    opacity=opacity,
                ))
            # Highlight background kalimat terpilih
            for idx_k, dipilih in enumerate(df_s["Dipilih"]):
                if dipilih == "✅":
                    fig_bar.add_vline(
                        x=idx_k,
                        line_color="rgba(34,197,94,0.18)",
                        line_width=22,
                    )
            fig_bar.update_layout(
                title="Skor Setiap Kalimat (latar hijau = dipilih oleh MMR)",
                barmode="group",
                height=370,
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="right", x=1),
                margin=dict(t=55, b=15),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # Radar chart profil rata-rata: Dipilih vs Tidak Dipilih
            sel_mask = df_s["Dipilih"] == "✅"
            radar_cats = ["TF-IDF", "LSA", "Posisi", "Panjang", "Keyword", "Fusi"]
            if sel_mask.sum() > 0 and (~sel_mask).sum() > 0:
                avg_sel   = df_s.loc[sel_mask,  radar_cats].mean().tolist()
                avg_nosel = df_s.loc[~sel_mask, radar_cats].mean().tolist()
                fig_radar = go.Figure()
                for vals, name, color in [
                    (avg_sel,   "Dipilih MMR",    "#22c55e"),
                    (avg_nosel, "Tidak Dipilih", "#94a3b8"),
                ]:
                    fig_radar.add_trace(go.Scatterpolar(
                        r=vals + [vals[0]],
                        theta=radar_cats + [radar_cats[0]],
                        name=name,
                        fill="toself",
                        line_color=color,
                    ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    title="Profil Rata-rata Skor: Dipilih vs Tidak Dipilih",
                    showlegend=True,
                    height=340,
                )
                st.plotly_chart(fig_radar, use_container_width=True)

    # ============================================================
    # TAB 3 — HASIL RINGKASAN
    # ============================================================
    with tab_output:
        st.subheader("📝 Hasil Ringkasan Ekstraktif")

        if "result" not in st.session_state:
            st.info("ℹ️ Jalankan pipeline di tab **Input** terlebih dahulu.")
            st.stop()

        R   = st.session_state["result"]
        txt = st.session_state["input_text"]

        n_orig = len(R["original_sentences"])
        n_summ = len(R["summary_sentences"])
        w_orig = len(txt.split())
        w_summ = len(R["summary_text"].split())
        pct_red = (1 - w_summ / w_orig) * 100 if w_orig > 0 else 0

        # Statistik ringkasan
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Kalimat Asli",      n_orig)
        c2.metric("Kalimat Ringkasan", n_summ,
                  delta=f"-{n_orig - n_summ} kalimat", delta_color="inverse")
        c3.metric("Kata Asli",         f"{w_orig:,}")
        c4.metric("Kata Ringkasan",    f"{w_summ:,}",
                  delta=f"-{pct_red:.0f}%", delta_color="inverse")

        st.markdown("---")

        # Teks Ringkasan Final
        st.markdown("#### 📄 Ringkasan Final")
        st.markdown(
            f'<div class="summary-box">{R["summary_text"]}</div>',
            unsafe_allow_html=True,
        )

        # Tombol Download
        st.markdown("")
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.download_button(
                label="⬇️ Unduh (.txt)",
                data=R["summary_text"],
                file_name="ringkasan_notulensi.txt",
                mime="text/plain",
            )
        with d2:
            md_content = (
                f"# Ringkasan Notulensi Forum RB — LPP TVRI\n\n"
                f"{R['summary_text']}\n\n"
                f"---\n"
                f"*Diringkas otomatis menggunakan BM25 TF-IDF + LSA + MMR*"
            )
            st.download_button(
                label="⬇️ Unduh (.md)",
                data=md_content,
                file_name="ringkasan_notulensi.md",
                mime="text/markdown",
            )
        with d3:
            numbered_text = "\n".join(
                f"{i+1}. {sent}"
                for i, sent in enumerate(R["summary_sentences"])
            )
            st.download_button(
                label="⬇️ Unduh Bernomor (.txt)",
                data=numbered_text,
                file_name="ringkasan_bernomor.txt",
                mime="text/plain",
            )
        with d4:
            if DB_AVAILABLE:
                if st.button("💾 Simpan ke DB", use_container_width=True):
                    meta  = st.session_state.get("doc_metadata", [{}])[0]
                    fname = meta.get("filename", "input_manual.txt")
                    teks_lengkap = st.session_state.get("input_text", "")
                    # Pakai id_dokumen dari hasil ekstraksi PDF (atau
                    # dokumen yang dimuat dari Pustaka) jika sudah
                    # tersedia — hindari baris tb_dokumen duplikat
                    # untuk teks yang sama.
                    id_dok = st.session_state.get("pdf_doc_id")
                    if not id_dok or id_dok <= 0:
                        id_dok = simpan_dokumen(
                            nama_file=fname,
                            tanggal_notulensi=meta.get("tanggal"),
                            tema=meta.get("tema"),
                            pembicara=meta.get("pembicara"),
                            moderator=meta.get("moderator"),
                            jumlah_kalimat=len(R["original_sentences"]),
                            jumlah_kata=len(teks_lengkap.split()),
                            teks_lengkap=teks_lengkap,
                        )
                        st.session_state["pdf_doc_id"] = id_dok
                    id_par = simpan_parameter(
                        compression_rate=compression_rate, lambda_mmr=lambda_mmr,
                        w1=w1, w2=w2, w3=w3, w4=w4, k_svd=n_comp,
                    )
                    simpan_kalimat_batch(
                        id_dok, R["scoring_df"],
                        original_sentences=R["original_sentences"],
                        processed_sentences=R["processed_sentences"],
                    )
                    id_ring = simpan_ringkasan(id_dok, id_par, R, teks_lengkap)
                    st.session_state["db_ids"] = {
                        "id_dokumen": id_dok,
                        "id_param": id_par,
                        "id_ringkasan": id_ring,
                    }
                    if id_ring > 0:
                        st.success(f"✅ Tersimpan! ID Ringkasan: {id_ring}")
                    else:
                        st.error("❌ Gagal simpan. Cek db_helper.py")
            else:
                st.caption("db_helper.py tidak ditemukan")

        st.markdown("---")

        # Highlight kalimat terpilih dalam teks asli
        st.markdown("#### 🔍 Visualisasi Kalimat Terpilih dalam Teks Asli")
        st.caption(
            "🟡 Kuning = kalimat yang diekstrak sebagai ringkasan  "
            "·  Abu-abu = tidak dipilih"
        )
        for i, sent in enumerate(R["original_sentences"]):
            if i in R["selected_indices"]:
                st.markdown(
                    f'<div class="sent-sel">'
                    f'<span class="slabel">K{i+1} ✅</span>'
                    f'{sent}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="sent-no">'
                    f'<span style="font-size:.67rem; font-weight:800; '
                    f'color:#cbd5e1; margin-right:5px">K{i+1}</span>'
                    f'{sent}</div>',
                    unsafe_allow_html=True,
                )

        # Pie chart proporsi kalimat
        fig_pie = go.Figure(go.Pie(
            labels=["Dipilih", "Tidak Dipilih"],
            values=[n_summ, n_orig - n_summ],
            hole=0.48,
            marker_colors=["#22c55e", "#e2e8f0"],
            textinfo="percent+label",
        ))
        fig_pie.update_layout(
            title="Proporsi Kalimat: Dipilih vs Dihilangkan",
            height=265,
            margin=dict(t=42, b=8),
            showlegend=False,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # ============================================================
    # TAB 4 — EVALUASI ROUGE
    # ============================================================
    with tab_eval:
        st.subheader("📊 Evaluasi Kualitas Ringkasan — Metrik ROUGE")

        if "result" not in st.session_state:
            st.info("ℹ️ Jalankan pipeline di tab **Input** terlebih dahulu.")
            st.stop()

        R = st.session_state["result"]

        # Tampilkan ringkasan sistem
        st.markdown("**Ringkasan Sistem (Hasil Otomatis):**")
        st.info(R["summary_text"])

        st.markdown("---")
        st.markdown("#### ✍️ Ringkasan Referensi / Pakar (Ground Truth)")
        st.caption(
            "Masukkan ringkasan buatan manusia/pakar sebagai acuan evaluasi. "
            "Semakin mirip dengan ringkasan sistem, semakin tinggi skor ROUGE."
        )

        ref_text = st.text_area(
            "Ringkasan Referensi Pakar",
            height=140,
            placeholder=(
                "Tulis atau paste ringkasan referensi pakar di sini...\n"
                "Contoh: 'Forum membahas reformasi birokrasi dan transformasi "
                "digital TVRI. Rapat menyepakati pembentukan tim gugus tugas...'"
            ),
        )

        # Penjelasan rumus ROUGE
        st.markdown(
            '<div class="fbox"><code>'
            'ROUGE-N: Precision = |overlap N-gram| / |N-gram sistem|  '
            '·  Recall = |overlap| / |N-gram referensi|  '
            '·  F1 = 2PR/(P+R)<br>'
            'ROUGE-L: Berbasis LCS (Longest Common Subsequence) '
            'via Dynamic Programming — O(m×n) waktu, O(n) ruang'
            '</code></div>',
            unsafe_allow_html=True,
        )

        if st.button("🧮  Hitung Skor ROUGE", type="primary"):
            if not ref_text.strip():
                st.warning("⚠️ Masukkan ringkasan referensi pakar terlebih dahulu.")
            else:
                evaluator = Evaluator()
                scores    = evaluator.evaluate(R["summary_text"], ref_text)
                # Simpan ke session_state agar hasil & tombol "Simpan ke
                # Database" tetap tampil walau halaman rerun (mis. saat
                # tombol simpan itu sendiri diklik).
                st.session_state["rouge_scores"]   = scores
                st.session_state["rouge_ref_text"] = ref_text

        # ── Tampilkan hasil evaluasi (DI LUAR blok tombol "Hitung") ────────
        # Sengaja diletakkan di luar agar tombol "Simpan Skor ROUGE ke
        # Database" di bawah tidak ikut hilang ketika halaman Streamlit
        # rerun akibat klik tombol tersebut.
        if "rouge_scores" in st.session_state:
            scores   = st.session_state["rouge_scores"]
            ref_used = st.session_state["rouge_ref_text"]

            st.markdown("---")
            st.markdown("#### 📈 Hasil Evaluasi ROUGE")

            # Kartu skor per metrik
            palette = {
                "ROUGE-1": "#3b82f6",
                "ROUGE-2": "#8b5cf6",
                "ROUGE-L": "#22c55e",
            }
            rouge_cols = st.columns(3)
            for col, (metric_name, color) in zip(rouge_cols, palette.items()):
                s = scores[metric_name]
                col.markdown(
                    f'<div class="rcard" style="border-top-color:{color}">'
                    f'<div style="font-size:1.0rem; font-weight:700; '
                    f'color:{color}; margin-bottom:10px">{metric_name}</div>'
                    f'<div style="font-size:.74rem; color:#64748b">PRECISION</div>'
                    f'<div style="font-size:1.3rem; font-weight:700">'
                    f'{s["precision"]:.4f}</div>'
                    f'<div style="font-size:.74rem; color:#64748b; margin-top:7px">'
                    f'RECALL</div>'
                    f'<div style="font-size:1.3rem; font-weight:700">'
                    f'{s["recall"]:.4f}</div>'
                    f'<div style="font-size:.74rem; color:#64748b; margin-top:7px">'
                    f'F1-SCORE</div>'
                    f'<div style="font-size:1.55rem; font-weight:800; color:{color}">'
                    f'{s["f1"]:.4f}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Bar chart perbandingan P / R / F1
            metric_names = list(palette.keys())
            P_values = [scores[m]["precision"] for m in metric_names]
            R_values = [scores[m]["recall"]    for m in metric_names]
            F_values = [scores[m]["f1"]        for m in metric_names]

            fig_rouge = go.Figure()
            for vals, label, color in [
                (P_values, "Precision", "#3b82f6"),
                (R_values, "Recall",    "#22c55e"),
                (F_values, "F1-Score",  "#f59e0b"),
            ]:
                fig_rouge.add_trace(go.Bar(
                    name=label,
                    x=metric_names,
                    y=vals,
                    marker_color=color,
                    text=[f"{v:.3f}" for v in vals],
                    textposition="outside",
                ))
            fig_rouge.update_layout(
                title="Perbandingan Skor ROUGE — Sistem vs Referensi Pakar",
                barmode="group",
                yaxis_range=[0, 1.15],
                height=385,
                margin=dict(t=52, b=15),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_rouge, use_container_width=True)

            # Interpretasi hasil
            avg_f1 = float(np.mean(F_values))
            # Interpretasi hasil dinonaktifkan — komentar dan default kosong
            badge = ""
            desc = ""
            # Untuk mengaktifkan ulang interpretasi ROUGE, lepas komentar blok berikut:
            # if avg_f1 >= 0.30:
            #     badge = "🟢 BAIK"
            #     desc  = (
            #         "Ringkasan sangat representatif terhadap referensi pakar. "
            #         "Konfigurasi parameter sudah optimal."
            #     )
            # elif avg_f1 >= 0.25:
            #     badge = "🟡 CUKUP"
            #     desc  = (
            #         "Cukup representatif. Coba naikkan λ (Lambda MMR) "
            #         "atau tingkatkan W4 (bobot Keyword)."
            #     )
            # else:
            #     badge = "🔴 PERLU DITINGKATKAN"
            #     desc  = (
            #         "Naikkan λ mendekati 1.0, tingkatkan W1 dan W4, "
            #         "atau kurangi Compression Rate untuk hasil lebih baik."
            #     )

            md = f"""
| Indikator | Nilai |
|---|---|
| Rata-rata F1 (R1+R2+RL)/3 | `{avg_f1:.4f}` |
| ROUGE-1 F1 (unigram overlap) | `{scores["ROUGE-1"]["f1"]:.4f}` |
| ROUGE-2 F1 (bigram overlap)  | `{scores["ROUGE-2"]["f1"]:.4f}` |
| ROUGE-L F1 (LCS-based)       | `{scores["ROUGE-L"]["f1"]:.4f}` |

"""
            # | **Kualitas Keseluruhan** | {badge} |
            # _{desc}_
            st.markdown(md)

            # ── Simpan evaluasi ROUGE ke database ────────
            if DB_AVAILABLE:
                ids = st.session_state.get("db_ids", {})
                id_ring = ids.get("id_ringkasan", -1)
                if id_ring > 0:
                    if st.button("💾  Simpan Skor ROUGE ke Database", type="secondary"):
                        id_ev = simpan_evaluasi_rouge(
                            id_ringkasan   = id_ring,
                            teks_referensi = ref_used,
                            skor_rouge     = scores,
                        )
                        if id_ev > 0:
                            st.success(f"✅ ROUGE tersimpan! ID Evaluasi: {id_ev} | Avg F1: {avg_f1:.4f}")
                        else:
                            st.error("❌ Gagal simpan ROUGE. Cek koneksi database.")
                else:
                    st.caption("💡 Klik **💾 Simpan ke DB** di tab Ringkasan terlebih dahulu.")

    # ============================================================
    # TAB 5 — RIWAYAT & DATABASE
    # ============================================================
    with tab_db:
        st.subheader("🗄️ Riwayat Peringkasan & Status Database")

        if not DB_AVAILABLE:
            st.error("❌ Modul db_helper.py tidak ditemukan. Pastikan file ada di folder yang sama dengan app.py.")
            st.code("pip install psycopg2-binary", language="bash")
        else:
            col_status, col_refresh = st.columns([3, 1])
            with col_status:
                if test_koneksi():
                    st.success("✅ Database PostgreSQL terhubung dan siap digunakan.")
                else:
                    st.error("❌ Tidak bisa terhubung ke database. Cek db_helper.py dan pastikan PostgreSQL berjalan.")
            with col_refresh:
                st.button("🔄 Refresh", use_container_width=True)

            st.markdown("---")

            # ════════════════════════════════════════════════════
            # PUSTAKA DOKUMEN — pilih PDF untuk diringkas ulang
            # ════════════════════════════════════════════════════
            st.markdown("#### 📚 Pustaka Dokumen — Pilih untuk Diringkas Ulang")
            st.caption(
                "Setiap dokumen yang pernah diunggah tersimpan beserta teks "
                "lengkapnya. Pilih salah satu untuk diringkas ULANG dengan "
                "parameter berbeda — tanpa perlu mengunggah PDF lagi."
            )

            daftar_dok = ambil_daftar_dokumen()
            if not daftar_dok:
                st.info("Belum ada dokumen tersimpan. Jalankan peringkasan lalu klik **💾 Simpan ke DB**.")
            else:
                opsi_dok = {
                    f"[{d['id_dokumen']}] {d['nama_file']} — "
                    f"{(d['tema'] or '—')[:45]} "
                    f"({d['tanggal_notulensi'] or '—'})": d['id_dokumen']
                    for d in daftar_dok
                }
                pilihan_dok = st.selectbox(
                    "Pilih dokumen dari pustaka:",
                    list(opsi_dok.keys()),
                    key="pilih_dokumen_pustaka",
                )
                id_dok_pilih = opsi_dok[pilihan_dok]
                dok_detail   = ambil_teks_dokumen(id_dok_pilih)

                if dok_detail:
                    info_sel = next(d for d in daftar_dok if d["id_dokumen"] == id_dok_pilih)
                    pc1, pc2, pc3, pc4 = st.columns(4)
                    pc1.metric("Tanggal", str(info_sel["tanggal_notulensi"] or "—"))
                    pc2.metric("Kalimat", info_sel["jumlah_kalimat"] or "—")
                    pc3.metric("Kata", f"{info_sel['jumlah_kata']:,}" if info_sel["jumlah_kata"] else "—")
                    pc4.metric("Karakter Teks", f"{info_sel['panjang_teks']:,}" if info_sel["panjang_teks"] else "—")

                    with st.expander("👁️ Preview Teks Dokumen", expanded=False):
                        preview_txt = dok_detail["teks_lengkap"] or ""
                        st.text_area(
                            "Preview teks dokumen",
                            value=preview_txt[:1500] + ("..." if len(preview_txt) > 1500 else ""),
                            height=180, disabled=True, label_visibility="collapsed",
                            key=f"preview_pustaka_{id_dok_pilih}",
                        )

                    if st.button("📥  Muat Teks ke Tab Input", type="primary",
                                 key="muat_pustaka_btn"):
                        st.session_state["pdf_text"] = dok_detail["teks_lengkap"]
                        st.session_state["doc_metadata"] = [{
                            "filename":  dok_detail.get("nama_file"),
                            "tanggal":   str(dok_detail["tanggal_notulensi"]) if dok_detail.get("tanggal_notulensi") else None,
                            "tema":      dok_detail.get("tema"),
                            "pembicara": dok_detail.get("pembicara"),
                            "moderator": dok_detail.get("moderator"),
                        }]
                        st.session_state["loaded_from_db"] = True
                        # Tautkan ke id_dokumen yang sudah ada — jika
                        # peringkasan dijalankan & disimpan, akan memakai
                        # baris tb_dokumen INI (tidak membuat duplikat
                        # teks_lengkap yang sama).
                        st.session_state["pdf_doc_id"] = id_dok_pilih
                        # Paksa radio mode ke "Upload File PDF" agar
                        # teks yang dimuat langsung terlihat di Tab Input
                        # tanpa pengguna perlu mengganti mode manual.
                        st.session_state["mode_input_radio"] = "📄  Upload File PDF"
                        # Bersihkan hasil & ID lama — dokumen berbeda
                        for k in ("result", "input_text", "rouge_scores",
                                  "rouge_ref_text", "db_ids"):
                            st.session_state.pop(k, None)
                        st.success(
                            "✅ Teks dimuat! Buka tab **📥 Input & PDF Upload** "
                            "(mode *Upload File PDF*) untuk mengatur parameter "
                            "dan menjalankan peringkasan ulang."
                        )
                else:
                    st.error("❌ Gagal memuat teks dokumen dari database.")

            st.markdown("---")
            st.markdown("#### 📋 Riwayat Ringkasan Tersimpan")

            with st.spinner("Mengambil data dari database..."):
                data = ambil_riwayat(limit=50)

            if not data:
                st.info("Belum ada data tersimpan. Jalankan peringkasan lalu klik **💾 Simpan ke DB** di tab Hasil Ringkasan.")
            else:
                df_hist = pd.DataFrame(data)
                if "rasio_kompresi" in df_hist.columns:
                    df_hist["Kompresi"] = df_hist["rasio_kompresi"].apply(
                        lambda x: f"{x*100:.0f}%" if x else "—")
                if "avg_f1" in df_hist.columns:
                    df_hist["Avg F1"] = df_hist["avg_f1"].apply(
                        lambda x: f"{x:.4f}" if x else "—")
                if "tanggal_dibuat" in df_hist.columns:
                    df_hist["Tanggal"] = df_hist["tanggal_dibuat"].apply(
                        lambda x: str(x)[:16] if x else "—")

                cols_show = ["id_ringkasan","nama_file","tema",
                             "jml_kalimat_asli","jml_kalimat_ringkasan",
                             "Kompresi","Avg F1","Tanggal"]
                cols_show = [c for c in cols_show if c in df_hist.columns]
                tampil = df_hist[cols_show].rename(columns={
                    "id_ringkasan":"ID","nama_file":"File","tema":"Tema",
                    "jml_kalimat_asli":"Kal.Asli","jml_kalimat_ringkasan":"Kal.Ring",
                })
                st.dataframe(tampil, use_container_width=True, height=240)
                st.caption(f"Total tersimpan: **{len(df_hist)}** ringkasan")

                csv = tampil.to_csv(index=False)
                st.download_button(
                    "⬇️ Unduh Riwayat (.csv)", data=csv,
                    file_name="riwayat_peringkasan.csv", mime="text/csv",
                )

                # ════════════════════════════════════════════════
                # DETAIL RINGKASAN — parameter, proses & pemodelan,
                # SVD, dan evaluasi ROUGE per ringkasan
                # ════════════════════════════════════════════════
                st.markdown("---")
                st.markdown("#### 🔍 Detail Ringkasan")
                st.caption(
                    "Pilih salah satu ringkasan untuk melihat parameter yang "
                    "dipakai, skor white-box per kalimat (Proses & Pemodelan), "
                    "info model LSA/SVD, dan hasil evaluasi ROUGE."
                )

                opsi_ring = {
                    f"[{r['id_ringkasan']}] {r['nama_file']} — "
                    f"{str(r['tanggal_dibuat'])[:16]}": r
                    for r in data
                }
                pilih_ring = st.selectbox(
                    "Pilih ringkasan:", list(opsi_ring.keys()),
                    key="pilih_detail_ringkasan",
                )
                Rh = opsi_ring[pilih_ring]

                # ── Info Dokumen ──────────────────────────────────
                with st.expander("📄 Info Dokumen", expanded=True):
                    ic1, ic2 = st.columns(2)
                    with ic1:
                        st.markdown(
                            f"**File:** {Rh['nama_file']}  \n"
                            f"**Tanggal Notulensi:** {Rh['tanggal_notulensi'] or '—'}  \n"
                            f"**Tema:** {Rh['tema'] or '—'}"
                        )
                    with ic2:
                        st.markdown(
                            f"**Pembicara:** {Rh['pembicara'] or '—'}  \n"
                            f"**Moderator:** {Rh['moderator'] or '—'}  \n"
                            f"**Metode Ekstraksi:** {Rh['metode_ekstraksi'] or '—'}"
                        )

                # ── Parameter yang Digunakan ──────────────────────
                with st.expander("⚙️ Parameter yang Digunakan", expanded=False):
                    pcol1, pcol2, pcol3 = st.columns(3)
                    with pcol1:
                        st.metric("Compression Rate", f"{(Rh['compression_rate'] or 0)*100:.0f}%")
                        st.metric("λ (Lambda MMR)", f"{Rh['lambda_mmr']:.2f}" if Rh['lambda_mmr'] is not None else "—")
                    with pcol2:
                        st.metric("k (Komponen SVD)", Rh['k_svd'] or "—")
                        st.metric("α (Alpha Fusion)", f"{Rh['alpha_fusion']:.2f}" if Rh['alpha_fusion'] is not None else "—")
                    with pcol3:
                        st.metric("Sim. Threshold MMR", f"{Rh['sim_threshold']:.2f}" if Rh['sim_threshold'] is not None else "—")
                        st.metric("Ekspansi Singkatan", "Ya" if Rh['expand_abbr'] else "Tidak")

                    st.markdown("**Bobot Fusi (4 Fitur):**")
                    bw1, bw2, bw3, bw4 = st.columns(4)
                    bw1.metric("W1 Konten",  f"{Rh['w1_konten']:.2f}", f"Ŵ={Rh['w1_norm']:.2f}" if Rh['w1_norm'] is not None else None)
                    bw2.metric("W2 Posisi",  f"{Rh['w2_posisi']:.2f}", f"Ŵ={Rh['w2_norm']:.2f}" if Rh['w2_norm'] is not None else None)
                    bw3.metric("W3 Panjang", f"{Rh['w3_panjang']:.2f}", f"Ŵ={Rh['w3_norm']:.2f}" if Rh['w3_norm'] is not None else None)
                    bw4.metric("W4 Keyword", f"{Rh['w4_keyword']:.2f}", f"Ŵ={Rh['w4_norm']:.2f}" if Rh['w4_norm'] is not None else None)

                # ── Ringkasan & Statistik ──────────────────────────
                st.markdown("**📝 Teks Ringkasan**")
                st.markdown(f'<div class="summary-box">{Rh["teks_ringkasan"]}</div>', unsafe_allow_html=True)

                sc1, sc2, sc3, sc4 = st.columns(4)
                sc1.metric("Kalimat Asli", Rh["jml_kalimat_asli"])
                sc2.metric("Kalimat Ringkasan", Rh["jml_kalimat_ringkasan"])
                sc3.metric("Kata Asli", f"{Rh['jml_kata_asli']:,}" if Rh["jml_kata_asli"] else "—")
                kompresi_pct = f"-{Rh['rasio_kompresi']*100:.0f}%" if Rh["rasio_kompresi"] is not None else "—"
                sc4.metric("Kata Ringkasan", f"{Rh['jml_kata_ringkasan']:,}" if Rh["jml_kata_ringkasan"] else "—", delta=kompresi_pct, delta_color="inverse")

                # ── Proses & Pemodelan (skor per kalimat) ──────────
                with st.expander("🔬 Proses & Pemodelan — Skor per Kalimat", expanded=False):
                    kalimat_data = ambil_kalimat_dokumen(Rh["id_dokumen"])
                    if not kalimat_data:
                        st.info("Detail skor per kalimat tidak tersedia untuk dokumen ini.")
                    else:
                        df_k = pd.DataFrame(kalimat_data)
                        df_disp = pd.DataFrame({
                            "No":      df_k["nomor_urut"],
                            "Kalimat": df_k["teks_kalimat"].apply(
                                lambda s: (s[:75] + "...") if isinstance(s, str) and len(s) > 75 else s),
                            "TF-IDF":  df_k["skor_tfidf"],
                            "LSA":     df_k["skor_lsa"],
                            "Posisi":  df_k["skor_posisi"],
                            "Panjang": df_k["skor_panjang"],
                            "Keyword": df_k["skor_keyword"],
                            "Fusi":    df_k["skor_fusi"],
                            "Dipilih": df_k["dipilih"].apply(lambda x: "✅" if x else ""),
                        })

                        def _hl_riwayat(row):
                            return (["background-color:#dcfce7;font-weight:600"] * len(row)
                                    if row["Dipilih"] == "✅" else [""] * len(row))

                        st.dataframe(
                            df_disp.style.apply(_hl_riwayat, axis=1)
                                   .format({c: "{:.4f}" for c in
                                            ["TF-IDF","LSA","Posisi","Panjang","Keyword","Fusi"]}),
                            use_container_width=True,
                            height=min(65 + 38 * len(df_disp), 450),
                        )

                        # Preprocessing: kalimat asli vs token bersih
                        if df_k["token_bersih"].notna().any():
                            with st.expander("📋 Preprocessing — Kalimat Asli vs Token Bersih", expanded=False):
                                pcol_a, pcol_b = st.columns(2)
                                with pcol_a:
                                    st.markdown("**Kalimat Asli**")
                                    for _, r in df_k.iterrows():
                                        txt = r["teks_kalimat"] or ""
                                        st.markdown(f"`K{r['nomor_urut']}` {txt[:85]}{'...' if len(txt)>85 else ''}")
                                with pcol_b:
                                    st.markdown("**Token Setelah Preprocessing**")
                                    for _, r in df_k.iterrows():
                                        tb = r["token_bersih"] or ""
                                        toks = tb.split()
                                        pv = " | ".join(toks[:7])
                                        st.markdown(f"`K{r['nomor_urut']}` `{pv}{'...' if len(toks)>7 else ''}`")

                        # Info Model LSA/SVD
                        if Rh.get("singular_values"):
                            st.markdown("**🧮 Model LSA / SVD**")
                            sv_list = [float(x) for x in Rh["singular_values"].split(",") if x.strip()]
                            ev_list = ([float(x) for x in Rh["explained_var"].split(",") if x.strip()]
                                       if Rh.get("explained_var") else [])
                            mcol1, mcol2 = st.columns(2)
                            with mcol1:
                                st.metric("Komponen Laten (k)", Rh.get("n_components_used") or len(sv_list))
                                if ev_list:
                                    st.metric("Total Varians Dijelaskan", f"{sum(ev_list)*100:.1f}%")
                            with mcol2:
                                fig_sv_h = go.Figure(go.Bar(
                                    x=[f"Σ{i+1}" for i in range(len(sv_list))],
                                    y=sv_list,
                                    marker=dict(color=sv_list, colorscale="Blues", showscale=False),
                                    text=[f"{v:.3f}" for v in sv_list],
                                    textposition="outside",
                                ))
                                fig_sv_h.update_layout(title="Singular Values", height=220, margin=dict(t=36,b=10))
                                st.plotly_chart(fig_sv_h, use_container_width=True)

                # ── Evaluasi ROUGE ──────────────────────────────────
                with st.expander("📊 Evaluasi ROUGE", expanded=False):
                    if Rh.get("avg_f1") is None:
                        st.info(
                            "Ringkasan ini belum dievaluasi ROUGE. Buka tab "
                            "**📊 Evaluasi ROUGE**, hitung skor, lalu klik "
                            "**💾 Simpan Skor ROUGE ke Database**."
                        )
                    else:
                        rcol1, rcol2, rcol3 = st.columns(3)
                        for col, (mname, pre, rec, f1, color) in zip(
                            [rcol1, rcol2, rcol3],
                            [("ROUGE-1", Rh["r1_precision"], Rh["r1_recall"], Rh["r1_f1"], "#3b82f6"),
                             ("ROUGE-2", Rh["r2_precision"], Rh["r2_recall"], Rh["r2_f1"], "#8b5cf6"),
                             ("ROUGE-L", Rh["rl_precision"], Rh["rl_recall"], Rh["rl_f1"], "#22c55e")]
                        ):
                            col.markdown(
                                f'<div class="rcard" style="border-top-color:{color}">'
                                f'<div style="font-size:1.0rem;font-weight:700;color:{color};margin-bottom:10px">{mname}</div>'
                                f'<div style="font-size:.74rem;color:#64748b">PRECISION</div>'
                                f'<div style="font-size:1.2rem;font-weight:700">{pre:.4f}</div>'
                                f'<div style="font-size:.74rem;color:#64748b;margin-top:6px">RECALL</div>'
                                f'<div style="font-size:1.2rem;font-weight:700">{rec:.4f}</div>'
                                f'<div style="font-size:.74rem;color:#64748b;margin-top:6px">F1-SCORE</div>'
                                f'<div style="font-size:1.4rem;font-weight:800;color:{color}">{f1:.4f}</div>'
                                f'</div>', unsafe_allow_html=True)
                        st.markdown(f"**Rata-rata F1 (R1+R2+RL)/3:** `{Rh['avg_f1']:.4f}`")
                        st.markdown("**Teks Referensi Pakar:**")
                        st.info(Rh.get("teks_referensi") or "—")

            st.markdown("---")
            st.markdown("#### ⚙️ Konfigurasi Koneksi Database")
            st.code("""# Edit di file db_helper.py
DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "database": "db_peringkasan_notulensi",
    "user":     "postgres",
    "password": "GANTI_INI",   # ← password PostgreSQL Anda
}""", language="python")

    # ── Footer ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#94a3b8; "
        "font-size:.73rem; padding:8px 0'>"
        "Peringkas Notulensi Forum RB — LPP TVRI &nbsp;·&nbsp; "
        "BM25 TF-IDF + Keyword Boost + LSA (SVD) + Fusion + MMR "
        "&nbsp;·&nbsp; "
        "Sastrawi · scikit-learn · PyMuPDF &nbsp;·&nbsp; "
        "<b>100% White-Box · Hybrid OCR · Zero Deep Learning</b>"
        "</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()
