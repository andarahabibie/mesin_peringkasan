-- ============================================================
-- MIGRASI v2: Riwayat Lengkap (Teks PDF + Preprocessing + SVD)
-- Sistem Peringkasan Notulensi Forum RB LPP TVRI
-- ============================================================
-- JALANKAN INI jika database Anda SUDAH ADA (sudah pernah
-- menjalankan setup_database.sql sebelumnya).
--
-- Aman dijalankan berkali-kali (idempotent) — tidak akan
-- menghapus data yang sudah ada.
--
-- CARA PAKAI:
--   1. Buka pgAdmin 4 → Query Tool pada db_peringkasan_notulensi
--   2. Buka file ini → Execute (F5)
-- ============================================================


-- ── tb_dokumen: tambah kolom teks lengkap hasil ekstraksi PDF ──
ALTER TABLE tb_dokumen
    ADD COLUMN IF NOT EXISTS teks_lengkap TEXT;

COMMENT ON COLUMN tb_dokumen.teks_lengkap IS
    'Teks lengkap hasil ekstraksi PDF (sebelum tokenisasi kalimat). '
    'Digunakan untuk fitur "pilih dokumen dari database untuk diringkas ulang".';


-- ── tb_ringkasan: tambah info model LSA/SVD ─────────────────
ALTER TABLE tb_ringkasan
    ADD COLUMN IF NOT EXISTS singular_values    TEXT,
    ADD COLUMN IF NOT EXISTS explained_var      TEXT,
    ADD COLUMN IF NOT EXISTS n_components_used  INT;

COMMENT ON COLUMN tb_ringkasan.singular_values IS
    'Nilai singular SVD (Σ), dipisah koma. Contoh: "12.34,8.21,5.67"';
COMMENT ON COLUMN tb_ringkasan.explained_var IS
    'Proporsi varians terjelas tiap komponen SVD, dipisah koma.';
COMMENT ON COLUMN tb_ringkasan.n_components_used IS
    'Jumlah komponen laten (k) yang benar-benar digunakan SVD.';


-- ── Verifikasi: tampilkan struktur kolom baru ───────────────
SELECT table_name, column_name, data_type
FROM information_schema.columns
WHERE table_schema = 'public'
  AND (
        (table_name = 'tb_dokumen'   AND column_name = 'teks_lengkap')
     OR (table_name = 'tb_ringkasan' AND column_name IN
            ('singular_values', 'explained_var', 'n_components_used'))
      )
ORDER BY table_name, column_name;

-- ============================================================
-- SELESAI. Jika muncul 4 baris hasil → migrasi berhasil.
-- token_bersih pada tb_kalimat TIDAK perlu migrasi karena
-- kolom tersebut sudah ada sejak setup_database.sql awal,
-- hanya belum diisi (akan terisi otomatis untuk ringkasan baru).
-- ============================================================