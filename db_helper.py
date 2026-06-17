"""
============================================================
db_helper.py — Helper Koneksi dan Operasi Database
Sistem Peringkasan Notulensi Forum RB LPP TVRI
============================================================
Cara pakai:
  1. Pastikan sudah install: pip install psycopg2-binary
  2. Ubah DB_PASSWORD sesuai password PostgreSQL Anda
  3. Jalankan: python db_helper.py  (untuk uji koneksi)
  4. Import fungsi-fungsi ini di app Streamlit Anda
============================================================
"""

import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, date
import json
import re

# ============================================================
# KONFIGURASI KONEKSI — UBAH SESUAI INSTALASI ANDA
# ============================================================
DB_CONFIG = {
    "host":     "localhost",    # jangan diubah jika PostgreSQL di PC sendiri
    "port":     5432,           # port default PostgreSQL
    "database": "db_peringkasan_notulensi",
    "user":     "postgres",     # username default PostgreSQL
    "password": "admin123",     # ← GANTI dengan password PostgreSQL Anda
}


# ============================================================
# FUNGSI DASAR: KONEKSI
# ============================================================
def get_connection():
    """Buka koneksi ke database PostgreSQL."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except psycopg2.OperationalError as e:
        print(f"❌ Gagal terhubung ke database: {e}")
        print("   Cek apakah PostgreSQL sudah berjalan dan password benar.")
        return None


def test_koneksi():
    """Uji apakah koneksi ke database berhasil."""
    conn = get_connection()
    if conn:
        cur = conn.cursor()
        cur.execute("SELECT version();")
        versi = cur.fetchone()[0]
        cur.close()
        conn.close()
        print(f"✅ Koneksi berhasil!")
        print(f"   PostgreSQL: {versi[:40]}...")
        return True
    return False


# ============================================================
# HELPER: PARSING TANGGAL BERBAHASA INDONESIA
# ============================================================
_BULAN_ID = {
    "januari": 1, "jan": 1,
    "februari": 2, "feb": 2,
    "maret": 3, "mar": 3,
    "april": 4, "apr": 4,
    "mei": 5,
    "juni": 6, "jun": 6,
    "juli": 7, "jul": 7,
    "agustus": 8, "ags": 8, "agu": 8,
    "september": 9, "sep": 9, "sept": 9,
    "oktober": 10, "okt": 10,
    "november": 11, "nov": 11,
    "desember": 12, "des": 12,
}
_BULAN_PATTERN = "|".join(sorted(_BULAN_ID.keys(), key=len, reverse=True))


def _parse_tanggal_indonesia(raw):
    """
    Konversi string tanggal hasil ekstraksi metadata PDF (berbahasa
    Indonesia, mis. "30 Maret 2026") menjadi objek date Python yang
    siap dimasukkan ke kolom DATE PostgreSQL.

    Dirancang tahan terhadap artefak OCR seperti karakter angka
    nyasar di depan (mis. "130 Maret 2026" — sisa nomor urut yang
    menyatu dengan tanggal saat ekstraksi). Karena menggunakan
    re.search (bukan re.match), regex akan mencari titik awal yang
    cocok di mana pun dalam string, sehingga "130 Maret 2026" tetap
    menghasilkan tanggal 30 Maret 2026 (bukan 13 atau gagal total).

    Juga mendukung format ISO "YYYY-MM-DD" (mis. saat nilai berasal
    dari str() atas objek date yang sudah tersimpan sebelumnya).

    Mengembalikan None jika raw bukan string / kosong / tidak cocok
    pola apa pun — sehingga tanggal_notulensi akan diisi NULL alih-
    alih menyebabkan error "sintaks masukan tidak valid untuk tipe
    date" saat INSERT.
    """
    if not raw or not isinstance(raw, str):
        return None
    raw = raw.strip()

    # Pola 1: "D(D) NamaBulanIndonesia YYYY", mis. "30 Maret 2026"
    #         atau dengan noise OCR "130 Maret 2026" → tetap 30.
    m = re.search(
        r"(\d{1,2})\s+(" + _BULAN_PATTERN + r")\s+(\d{4})",
        raw, re.IGNORECASE,
    )
    if m:
        try:
            day   = int(m.group(1))
            month = _BULAN_ID[m.group(2).lower()]
            year  = int(m.group(3))
            return date(year, month, day)
        except ValueError:
            return None

    # Pola 2: format ISO "YYYY-MM-DD" (mis. dari str(date_obj))
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", raw)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None

    return None
def simpan_dokumen(nama_file, tanggal_notulensi=None, tema=None,
                   pembicara=None, moderator=None,
                   metode_ekstraksi='ocr', ukuran_file_kb=None,
                   jumlah_halaman=None, jumlah_kalimat=None,
                   jumlah_kata=None, teks_lengkap=None) -> int:
    """
    Simpan metadata dokumen PDF ke tb_dokumen.
    Mengembalikan id_dokumen (int) atau -1 jika gagal.

    Parameter teks_lengkap menyimpan seluruh teks hasil ekstraksi
    PDF (sebelum tokenisasi kalimat), sehingga dokumen ini dapat
    dipilih kembali dari Tab Riwayat untuk diringkas ulang dengan
    parameter berbeda tanpa perlu upload PDF lagi.

    Contoh pemakaian:
        id_dok = simpan_dokumen(
            nama_file="27_April_2026.pdf",
            tema="Pengembangan Media Sosial",
            metode_ekstraksi="ocr",
            jumlah_kalimat=87,
            jumlah_kata=2166,
            teks_lengkap=input_text,
        )
    """
    # Konversi tanggal berbahasa Indonesia / artefak OCR ke date object
    # PostgreSQL. Jika tidak cocok pola apa pun, hasilnya None (NULL)
    # — INSERT tetap berhasil tanpa error sintaks tipe date.
    if isinstance(tanggal_notulensi, str):
        tanggal_notulensi = _parse_tanggal_indonesia(tanggal_notulensi)

    sql = """
        INSERT INTO tb_dokumen
            (nama_file, tanggal_notulensi, tema, pembicara, moderator,
             metode_ekstraksi, ukuran_file_kb, jumlah_halaman,
             jumlah_kalimat, jumlah_kata, teks_lengkap)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id_dokumen;
    """
    conn = get_connection()
    if not conn:
        return -1
    try:
        cur = conn.cursor()
        cur.execute(sql, (
            nama_file, tanggal_notulensi, tema, pembicara, moderator,
            metode_ekstraksi, ukuran_file_kb, jumlah_halaman,
            jumlah_kalimat, jumlah_kata, teks_lengkap
        ))
        id_dok = cur.fetchone()[0]
        conn.commit()
        print(f"✅ Dokumen disimpan: id={id_dok}, file={nama_file}")
        return id_dok
    except Exception as e:
        conn.rollback()
        print(f"❌ Gagal simpan dokumen: {e}")
        return -1
    finally:
        cur.close()
        conn.close()


# ============================================================
# FUNGSI 2: SIMPAN PARAMETER
# ============================================================
def simpan_parameter(compression_rate=0.10, lambda_mmr=0.80,
                     w1=1.00, w2=0.30, w3=0.30, w4=1.00,
                     k_svd=7, alpha=0.55, expand_abbr=True,
                     nama_preset=None) -> int:
    """
    Simpan konfigurasi parameter ke tb_parameter.
    Mengembalikan id_param (int).

    Contoh pemakaian:
        id_par = simpan_parameter(
            compression_rate=0.10,
            lambda_mmr=0.80,
            w1=1.0, w2=0.3, w3=0.3, w4=1.0,
            k_svd=7
        )
    """
    total  = w1 + w2 + w3 + w4 or 1.0
    w1n, w2n, w3n, w4n = w1/total, w2/total, w3/total, w4/total

    sql = """
        INSERT INTO tb_parameter
            (nama_preset, compression_rate, lambda_mmr,
             w1_konten, w2_posisi, w3_panjang, w4_keyword,
             w1_norm, w2_norm, w3_norm, w4_norm,
             k_svd, alpha_fusion, expand_abbr)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id_param;
    """
    conn = get_connection()
    if not conn:
        return -1
    try:
        cur = conn.cursor()
        cur.execute(sql, (
            nama_preset, compression_rate, lambda_mmr,
            w1, w2, w3, w4,
            round(w1n,4), round(w2n,4), round(w3n,4), round(w4n,4),
            k_svd, alpha, 1 if expand_abbr else 0
        ))
        id_par = cur.fetchone()[0]
        conn.commit()
        print(f"✅ Parameter disimpan: id={id_par}, λ={lambda_mmr}, k={k_svd}")
        return id_par
    except Exception as e:
        conn.rollback()
        print(f"❌ Gagal simpan parameter: {e}")
        return -1
    finally:
        cur.close()
        conn.close()


# ============================================================
# FUNGSI 3: SIMPAN KALIMAT (batch insert)
# ============================================================
def simpan_kalimat_batch(id_dokumen: int, scoring_df,
                          original_sentences: list = None,
                          processed_sentences: list = None) -> bool:
    """
    Simpan semua kalimat + skor dari DataFrame hasil pipeline.
    Parameter scoring_df adalah DataFrame output run_pipeline().

    Parameter opsional original_sentences (list[str]) dan
    processed_sentences (list[list[str]]) — jika diberikan,
    akan menyimpan TEKS KALIMAT LENGKAP (scoring_df["Kalimat"]
    terpotong 60 karakter) dan token_bersih hasil preprocessing
    Sastrawi (stopword + stemming), agar tampilan "Proses &
    Pemodelan" di Tab Riwayat sama lengkapnya dengan Tab 2.

    Contoh pemakaian:
        result = run_pipeline(text, ...)
        simpan_kalimat_batch(
            id_dok, result['scoring_df'],
            original_sentences=result['original_sentences'],
            processed_sentences=result['processed_sentences'],
        )
    """
    sql = """
        INSERT INTO tb_kalimat
            (id_dokumen, nomor_urut, teks_kalimat, token_bersih,
             skor_tfidf, skor_lsa, skor_posisi,
             skor_panjang, skor_keyword, skor_fusi, dipilih)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id_dokumen, nomor_urut) DO UPDATE SET
            teks_kalimat = EXCLUDED.teks_kalimat,
            token_bersih = EXCLUDED.token_bersih,
            skor_tfidf  = EXCLUDED.skor_tfidf,
            skor_lsa    = EXCLUDED.skor_lsa,
            skor_posisi = EXCLUDED.skor_posisi,
            skor_panjang= EXCLUDED.skor_panjang,
            skor_keyword= EXCLUDED.skor_keyword,
            skor_fusi   = EXCLUDED.skor_fusi,
            dipilih     = EXCLUDED.dipilih;
    """
    conn = get_connection()
    if not conn:
        return False
    try:
        cur  = conn.cursor()
        data = []
        for pos, (_, baris) in enumerate(scoring_df.iterrows()):
            teks_full = (
                original_sentences[pos]
                if original_sentences is not None and pos < len(original_sentences)
                else baris["Kalimat"]
            )
            token_str = (
                " ".join(processed_sentences[pos])
                if processed_sentences is not None and pos < len(processed_sentences)
                else None
            )
            data.append((
                id_dokumen,
                int(baris["No"]),
                teks_full,
                token_str,
                float(baris["TF-IDF"]),
                float(baris["LSA"]),
                float(baris["Posisi"]),
                float(baris["Panjang"]),
                float(baris["Keyword"]),
                float(baris["Fusi"]),
                baris["Dipilih"] == "✅",
            ))
        cur.executemany(sql, data)
        conn.commit()
        print(f"✅ {len(data)} kalimat disimpan untuk id_dokumen={id_dokumen}")
        return True
    except Exception as e:
        conn.rollback()
        print(f"❌ Gagal simpan kalimat: {e}")
        return False
    finally:
        cur.close()
        conn.close()


# ============================================================
# FUNGSI 4: SIMPAN RINGKASAN
# ============================================================
def simpan_ringkasan(id_dokumen: int, id_param: int,
                     result: dict, teks_asli: str) -> int:
    """
    Simpan hasil ringkasan ke tb_ringkasan.
    Parameter result adalah dict output run_pipeline().
    Mengembalikan id_ringkasan (int).

    Selain teks ringkasan dan statistik kompresi, fungsi ini juga
    menyimpan informasi model LSA/SVD (singular_values, explained_var,
    n_components_used) sebagai bagian dari rekam jejak "Proses &
    Pemodelan" yang ditampilkan di Tab Riwayat.

    Contoh pemakaian:
        result = run_pipeline(text, ...)
        id_ring = simpan_ringkasan(id_dok, id_par, result, text)
    """
    jml_kal_asli  = len(result["original_sentences"])
    jml_kal_ring  = len(result["summary_sentences"])
    jml_kata_asli = len(teks_asli.split())
    jml_kata_ring = len(result["summary_text"].split())
    rasio         = round(1 - jml_kata_ring / jml_kata_asli, 4) if jml_kata_asli > 0 else 0
    indeks_str    = ",".join(str(i) for i in result["selected_indices"])

    # Info model LSA/SVD — disimpan sebagai string dipisah koma
    sv = result.get("singular_values")
    ev = result.get("explained_var")
    singular_str  = ",".join(f"{v:.6f}" for v in sv) if sv is not None else None
    explained_str = ",".join(f"{v:.6f}" for v in ev) if ev is not None else None
    n_comp_used   = result.get("n_components_used")

    sql = """
        INSERT INTO tb_ringkasan
            (id_dokumen, id_param, teks_ringkasan, indeks_kalimat,
             jml_kalimat_asli, jml_kalimat_ringkasan,
             jml_kata_asli, jml_kata_ringkasan, rasio_kompresi,
             singular_values, explained_var, n_components_used)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id_ringkasan;
    """
    conn = get_connection()
    if not conn:
        return -1
    try:
        cur = conn.cursor()
        cur.execute(sql, (
            id_dokumen, id_param, result["summary_text"], indeks_str,
            jml_kal_asli, jml_kal_ring,
            jml_kata_asli, jml_kata_ring, rasio,
            singular_str, explained_str, n_comp_used,
        ))
        id_ring = cur.fetchone()[0]
        conn.commit()
        print(f"✅ Ringkasan disimpan: id={id_ring}, kompresi={rasio*100:.0f}%")
        return id_ring
    except Exception as e:
        conn.rollback()
        print(f"❌ Gagal simpan ringkasan: {e}")
        return -1
    finally:
        cur.close()
        conn.close()


# ============================================================
# FUNGSI 5: SIMPAN EVALUASI ROUGE
# ============================================================
def simpan_evaluasi_rouge(id_ringkasan: int, teks_referensi: str,
                           skor_rouge: dict) -> int:
    """
    Simpan hasil evaluasi ROUGE ke tb_evaluasi_rouge.
    Parameter skor_rouge adalah dict output Evaluator().evaluate().
    Mengembalikan id_evaluasi (int).

    Contoh pemakaian:
        ev   = Evaluator()
        skor = ev.evaluate(result['summary_text'], ref_text)
        simpan_evaluasi_rouge(id_ring, ref_text, skor)
    """
    r1 = skor_rouge.get("ROUGE-1", {})
    r2 = skor_rouge.get("ROUGE-2", {})
    rl = skor_rouge.get("ROUGE-L", {})
    avg = round((r1.get("f1",0) + r2.get("f1",0) + rl.get("f1",0)) / 3, 4)

    sql = """
        INSERT INTO tb_evaluasi_rouge
            (id_ringkasan, teks_referensi,
             r1_precision, r1_recall, r1_f1,
             r2_precision, r2_recall, r2_f1,
             rl_precision, rl_recall, rl_f1,
             avg_f1)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id_evaluasi;
    """
    conn = get_connection()
    if not conn:
        return -1
    try:
        cur = conn.cursor()
        cur.execute(sql, (
            id_ringkasan, teks_referensi,
            r1.get("precision"), r1.get("recall"), r1.get("f1"),
            r2.get("precision"), r2.get("recall"), r2.get("f1"),
            rl.get("precision"), rl.get("recall"), rl.get("f1"),
            avg
        ))
        id_ev = cur.fetchone()[0]
        conn.commit()
        print(f"✅ ROUGE disimpan: id={id_ev}, Avg F1={avg:.4f}")
        return id_ev
    except Exception as e:
        conn.rollback()
        print(f"❌ Gagal simpan evaluasi: {e}")
        return -1
    finally:
        cur.close()
        conn.close()


# ============================================================
# FUNGSI 6: AMBIL RIWAYAT RINGKASAN
# ============================================================
def ambil_riwayat(limit=20) -> list:
    """
    Ambil riwayat ringkasan lengkap (join 4 tabel: dokumen,
    parameter, ringkasan, evaluasi ROUGE).

    Mengembalikan SEMUA kolom yang dibutuhkan untuk menampilkan
    detail lengkap di Tab Riwayat: parameter yang dipakai, info
    model LSA/SVD, statistik ringkasan, dan skor ROUGE + teks
    referensi pakar.
    """
    sql = """
        SELECT
            r.id_ringkasan,
            r.id_dokumen,
            r.id_param,
            d.nama_file,
            d.tanggal_notulensi,
            d.tema,
            d.pembicara,
            d.moderator,
            d.metode_ekstraksi,
            -- Parameter yang digunakan
            p.compression_rate,
            p.lambda_mmr,
            p.w1_konten, p.w2_posisi, p.w3_panjang, p.w4_keyword,
            p.w1_norm, p.w2_norm, p.w3_norm, p.w4_norm,
            p.k_svd, p.alpha_fusion, p.sim_threshold, p.expand_abbr,
            -- Statistik ringkasan
            r.jml_kalimat_asli,
            r.jml_kalimat_ringkasan,
            r.jml_kata_asli,
            r.jml_kata_ringkasan,
            r.rasio_kompresi,
            r.teks_ringkasan,
            r.indeks_kalimat,
            -- Info model LSA/SVD
            r.singular_values,
            r.explained_var,
            r.n_components_used,
            -- Evaluasi ROUGE
            e.id_evaluasi,
            e.teks_referensi,
            e.avg_f1,
            e.r1_precision, e.r1_recall, e.r1_f1,
            e.r2_precision, e.r2_recall, e.r2_f1,
            e.rl_precision, e.rl_recall, e.rl_f1,
            r.tanggal_dibuat
        FROM  tb_ringkasan  r
        JOIN  tb_dokumen    d ON r.id_dokumen = d.id_dokumen
        JOIN  tb_parameter  p ON r.id_param   = p.id_param
        LEFT JOIN tb_evaluasi_rouge e ON e.id_ringkasan = r.id_ringkasan
        ORDER BY r.tanggal_dibuat DESC
        LIMIT %s;
    """
    conn = get_connection()
    if not conn:
        return []
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(sql, (limit,))
        rows = cur.fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        print(f"❌ Gagal ambil riwayat: {e}")
        return []
    finally:
        cur.close()
        conn.close()


# ============================================================
# FUNGSI 7: AMBIL DAFTAR DOKUMEN (untuk pustaka/pilih ulang)
# ============================================================
def ambil_daftar_dokumen() -> list:
    """
    Ambil daftar semua dokumen yang teks lengkapnya tersimpan
    di database (kolom teks_lengkap TIDAK NULL).

    Digunakan untuk menampilkan pilihan "Pustaka Dokumen" di
    Tab Riwayat, sehingga pengguna dapat memilih dokumen yang
    PERNAH diunggah untuk diringkas ULANG dengan parameter
    berbeda — tanpa perlu upload PDF lagi.

    Returns
    -------
    list[dict] dengan keys: id_dokumen, nama_file, tema,
    tanggal_notulensi, pembicara, moderator, metode_ekstraksi,
    jumlah_kalimat, jumlah_kata, tanggal_upload, panjang_teks
    """
    sql = """
        SELECT
            id_dokumen, nama_file, tema, tanggal_notulensi,
            pembicara, moderator, metode_ekstraksi,
            jumlah_kalimat, jumlah_kata, tanggal_upload,
            LENGTH(teks_lengkap) AS panjang_teks
        FROM tb_dokumen
        WHERE teks_lengkap IS NOT NULL
          AND LENGTH(teks_lengkap) > 0
        ORDER BY tanggal_upload DESC;
    """
    conn = get_connection()
    if not conn:
        return []
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(sql)
        rows = cur.fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        print(f"❌ Gagal ambil daftar dokumen: {e}")
        return []
    finally:
        cur.close()
        conn.close()


# ============================================================
# FUNGSI 8: AMBIL TEKS LENGKAP SATU DOKUMEN
# ============================================================
def ambil_teks_dokumen(id_dokumen: int) -> dict:
    """
    Ambil teks lengkap + metadata satu dokumen berdasarkan id.
    Digunakan saat pengguna klik "Muat Teks Ini" di Tab Riwayat —
    hasilnya dimasukkan ke session_state agar bisa langsung
    diringkas ulang di Tab Input.

    Returns
    -------
    dict dengan keys: teks_lengkap, nama_file, tanggal_notulensi,
    tema, pembicara, moderator  — atau None jika tidak ditemukan.
    """
    sql = """
        SELECT teks_lengkap, nama_file, tanggal_notulensi,
               tema, pembicara, moderator
        FROM tb_dokumen
        WHERE id_dokumen = %s;
    """
    conn = get_connection()
    if not conn:
        return None
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(sql, (id_dokumen,))
        row = cur.fetchone()
        return dict(row) if row else None
    except Exception as e:
        print(f"❌ Gagal ambil teks dokumen: {e}")
        return None
    finally:
        cur.close()
        conn.close()


# ============================================================
# FUNGSI 9: AMBIL TABEL SKOR KALIMAT (Proses & Pemodelan)
# ============================================================
def ambil_kalimat_dokumen(id_dokumen: int) -> list:
    """
    Ambil seluruh baris tb_kalimat untuk satu dokumen, terurut
    berdasarkan nomor_urut. Setara dengan `scoring_df` pada Tab 2
    (Proses & Pemodelan) — digunakan untuk menampilkan ulang
    tabel skor white-box (BM25, LSA, Posisi, Panjang, Keyword,
    Fusi, dan status Dipilih) pada Tab Riwayat.

    Returns
    -------
    list[dict] dengan keys: nomor_urut, teks_kalimat, token_bersih,
    skor_tfidf, skor_lsa, skor_posisi, skor_panjang, skor_keyword,
    skor_fusi, dipilih
    """
    sql = """
        SELECT nomor_urut, teks_kalimat, token_bersih,
               skor_tfidf, skor_lsa, skor_posisi,
               skor_panjang, skor_keyword, skor_fusi, dipilih
        FROM tb_kalimat
        WHERE id_dokumen = %s
        ORDER BY nomor_urut;
    """
    conn = get_connection()
    if not conn:
        return []
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(sql, (id_dokumen,))
        rows = cur.fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        print(f"❌ Gagal ambil kalimat dokumen: {e}")
        return []
    finally:
        cur.close()
        conn.close()


# ============================================================
# FUNGSI 7: SIMPAN SEMUA (satu panggilan lengkap)
# ============================================================
def simpan_hasil_pipeline(
    nama_file, teks_asli, result, metadata,
    params, ref_text=None, skor_rouge=None
) -> dict:
    """
    Wrapper: simpan satu sesi peringkasan penuh ke semua tabel.
    Mengembalikan dict berisi semua ID yang dibuat.

    Contoh pemakaian di Streamlit:
        ids = simpan_hasil_pipeline(
            nama_file  = "27_April_2026.pdf",
            teks_asli  = input_text,
            result     = result,          # output run_pipeline()
            metadata   = doc_metadata[0], # metadata dari PDFExtractor
            params     = {                # parameter dari sidebar
                "compression_rate": 0.10,
                "lambda_mmr": 0.80,
                "w1": 1.0, "w2": 0.30, "w3": 0.30, "w4": 1.0,
                "k_svd": 7
            },
            ref_text   = ref_text,        # opsional: teks referensi pakar
            skor_rouge = skor_rouge       # opsional: dict ROUGE
        )
    """
    ids = {}

    # 1. Simpan dokumen (termasuk teks lengkap untuk Pustaka Dokumen)
    ids["id_dokumen"] = simpan_dokumen(
        nama_file         = nama_file,
        tanggal_notulensi = metadata.get("tanggal"),
        tema              = metadata.get("tema"),
        pembicara         = metadata.get("pembicara"),
        moderator         = metadata.get("moderator"),
        metode_ekstraksi  = metadata.get("metode", "ocr"),
        jumlah_kalimat    = len(result["original_sentences"]),
        jumlah_kata       = len(teks_asli.split()),
        teks_lengkap      = teks_asli,
    )

    # 2. Simpan parameter
    ids["id_param"] = simpan_parameter(
        compression_rate = params.get("compression_rate", 0.10),
        lambda_mmr       = params.get("lambda_mmr", 0.80),
        w1 = params.get("w1", 1.0), w2 = params.get("w2", 0.30),
        w3 = params.get("w3", 0.30), w4 = params.get("w4", 1.0),
        k_svd            = params.get("k_svd", 7),
    )

    # 3. Simpan kalimat + skor (lengkap dengan teks penuh & token_bersih)
    simpan_kalimat_batch(
        ids["id_dokumen"], result["scoring_df"],
        original_sentences  = result.get("original_sentences"),
        processed_sentences = result.get("processed_sentences"),
    )

    # 4. Simpan ringkasan
    ids["id_ringkasan"] = simpan_ringkasan(
        ids["id_dokumen"], ids["id_param"], result, teks_asli
    )

    # 5. Simpan evaluasi ROUGE (opsional)
    if ref_text and skor_rouge and ids["id_ringkasan"] > 0:
        ids["id_evaluasi"] = simpan_evaluasi_rouge(
            ids["id_ringkasan"], ref_text, skor_rouge
        )

    print(f"\n{'='*50}")
    print(f"✅ Semua data tersimpan ke database!")
    print(f"   id_dokumen   : {ids.get('id_dokumen')}")
    print(f"   id_param     : {ids.get('id_param')}")
    print(f"   id_ringkasan : {ids.get('id_ringkasan')}")
    print(f"   id_evaluasi  : {ids.get('id_evaluasi', 'belum ada evaluasi')}")
    print(f"{'='*50}\n")
    return ids


# ============================================================
# ENTRY POINT: jalankan file ini untuk uji koneksi
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("  UJI KONEKSI DATABASE POSTGRESQL")
    print("  Sistem Peringkasan Notulensi LPP TVRI")
    print("=" * 50)
    test_koneksi()
    print()
    print("Riwayat ringkasan yang tersimpan:")
    riwayat = ambil_riwayat(limit=5)
    if riwayat:
        for r in riwayat:
            print(f"  [{r['id_ringkasan']}] {r['nama_file']} | "
                  f"Avg F1: {r.get('avg_f1','—')}")
    else:
        print("  (belum ada data — database masih kosong)")