# -*- coding: utf-8 -*-
"""PDF → paragraf JSONL (chunk_jsonl.py girdisi).

fitz bloklarını (y, x) koordinatına göre sıralayıp okuma sırasını düzeltir;
blok içi sarılmış satırları birleştirir, liste maddelerini korur.
Başlık tespiti chunk_jsonl.py'e bırakılır (numaralı/büyük-harf kalıpları).
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import fitz

BULLET = re.compile(r"^[•\-\u2013\u2014*]\s")
BULLET_ONLY = re.compile(r"^[•\-\u2013\u2014*]+$")
NUMBERED = re.compile(r"^\d+[.)]\s")
SENTENCE_END = (".", "!", "?", ":", ";")

# Satır içi başlık ("Cinsel Yaşam", "Sosyal Haklar"): 2-5 kelime, Title-Case, noktalama yok.
def _is_title_case_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) < 8 or len(s) > 60 or s[-1] in ".?!:;,":
        return False
    if s[0] in "•-–—*" or re.match(r"^\d", s):
        return False
    words = re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşü]+", s)
    if not 2 <= len(words) <= 5:
        return False
    return sum(1 for w in words if w[0].isupper()) >= 2

# Referans / içindekiler bölümlerini RAG için düşük değerli sayıp atlar.
REF_HEADING = re.compile(r"^(KAYNAKLAR|KAYNAKÇA|REFERANSLAR)$", re.IGNORECASE)
CHAPTER_HEADING = re.compile(r"^BÖLÜM\b", re.IGNORECASE)
# Sayfa numarası / altbilgi kalıntısı: yalnız rakam-nokta-tire ("2306-11.").
STRAY_TOKEN = re.compile(r"^[\d\-./]+\s*$")

# Paragraf seviyesinde doz talimatı (hasta RAG'ına girmemeli).
DOSE_INSTRUCTION = re.compile(
    r"doz\w*\s+(?:azalt|artır|ayarla|değiştir|düşür|kesil|geçilme)|"
    r"doz\s*%\s*\d|"
    r"%(?:\d+|\d+-\d+)\s*(?:azalt|artır|düşür|değiştir)|"
    r"titrasyon|"
    r"(?:IU|U|ünite)\s*/\s*(?:kg|gün)|"
    r"ünite\s+insülin",
    re.IGNORECASE,
)


def clean_records(records: list[dict], exclude_blocks: list[tuple[str, str]]) -> list[dict]:
    """Kaynakça, içindekiler, artık tokenları ve isteğe bağlı bölüm aralıklarını çıkarır.

    exclude_blocks: [(start_regex, end_regex), ...] — start eşleşen kayıttan
    end eşleşen kayıda (dahil) kadar olan aralık atlanır (doz bölümleri için).
    """
    out: list[dict] = []
    in_refs = False
    skip_until: str | None = None

    for r in records:
        text = (r["text"] or "").strip()
        up = text.upper()

        if STRAY_TOKEN.match(text):
            continue

        if REF_HEADING.match(up):
            in_refs = True
            continue
        if in_refs:
            if CHAPTER_HEADING.match(up):
                in_refs = False
            else:
                continue

        if skip_until:
            if re.search(skip_until, text, re.IGNORECASE):
                skip_until = None
                out.append(r)  # bitiş kaydı korunur (sonraki bölümün başı)
            continue

        started = None
        for start_pat, end_pat in exclude_blocks:
            if re.search(start_pat, text, re.IGNORECASE):
                started = end_pat
                break
        if started:
            skip_until = started
            continue  # başlangıç kaydı atlanır

        out.append(r)
    return out


def fix_pdf_font(text: str) -> str:
    """PDF font bozulmalarını düzeltir (ligatürler + 'ti' ligatürü).

    Bazı PDF'lerde "ti" ligatürü yanlış Unicode'a (3, $, )) eşlenir:
      "Diyabe3n" → "Diyabetin", "mul$disipliner" → "multidisipliner".
    Yalnızca harf arasındaki bozuk karakterler düzeltilir (rakam/noktalama korunur).
    """
    for lig, repl in (("ﬁ", "fi"), ("ﬂ", "fl"), ("ﬃ", "ffi"), ("ﬄ", "ffl")):
        text = text.replace(lig, repl)
    # Türkçe İ/ı glifi bozulması: bazı fontlarda İ→‹, ı→› olarak çıkar.
    text = text.replace("‹", "İ").replace("›", "ı")
    return re.sub(r"(?<=[a-zçğıöşü])[3$)](?=[a-zçğıöşü])", "ti", text)


def slugify_pdf(name: str) -> str:
    text = name.lower().strip()
    text = re.sub(r"\.pdf$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-z0-9çğıöşü\-]+", "-", text, flags=re.IGNORECASE)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "document"


def reconstruct_block(block_text: str) -> list[str]:
    """Blok içi sarılmış satırları birleştir; başlık/liste maddelerini ayrı paragraf yapar."""
    lines = [ln.strip() for ln in (block_text or "").split("\n")]
    lines = [ln for ln in lines if ln]
    if not lines:
        return []

    out: list[str] = []
    buf = ""
    for ln in lines:
        if not buf:
            buf = ln
            continue
        # Yeni satır: cümle bitmiş, liste maddesi, numaralı/üst başlık veya BÖLÜM.
        is_break = (
            buf.rstrip().endswith(SENTENCE_END)
            or BULLET.match(ln)
            or BULLET_ONLY.match(ln)  # "•" kendi satırında
            or NUMBERED.match(ln)
            or re.match(r"^\d+(?:\.\d+)+\.?\s+\S", ln)  # 12.1. / 3.2.1.
            or re.match(r"^BÖLÜM\b", ln, re.IGNORECASE)
            or (ln.isupper() and len(ln) <= 80)  # ÖNSÖZ, GİRİŞ, KAYNAKLAR...
            or _is_title_case_heading(ln)  # Cinsel Yaşam, Sosyal Haklar...
        )
        if is_break:
            out.append(buf)
            buf = ln
        else:
            buf = f"{buf} {ln}"
    if buf:
        out.append(buf)
    return out


def detect_column_split_x(blocks, page_height: float | None = None) -> float | None:
    """İki sütunlu sayfayı tespit eder; sütun sınırının x koordinatını döndürür (yoksa None).

    Sayfanın ortasında (%15-%85 arası) hiçbir bloğun geçmediği dikey bir
    boşluk bandı varsa iki sütun vardır. Bu bandın orta noktası döndürülür.
    Üst/alt kenar payındaki bloklar (koşu başlığı, sayfa numarası) tespitten çıkarılır.
    """
    if page_height:
        blocks = [b for b in blocks if 60 <= b[1] <= page_height - 40]
    if len(blocks) < 4:
        return None

    min_x = min(b[0] for b in blocks)
    max_x = max(b[2] for b in blocks)
    span = max_x - min_x
    if span < 200:  # çok dar sayfa -> tek sütun kabul et
        return None

    # x eksenini 5 puanlık dilimlere böl; her dilim metinle kaplı mı?
    step = 5
    n = int(span / step) + 1
    covered = [False] * n
    for b in blocks:
        a = int((b[0] - min_x) / step)
        c = int((b[2] - min_x) / step)
        for j in range(max(0, a), min(n, c + 1)):
            covered[j] = True

    # Ortadaki en geniş boş bandı bul.
    lo = int(0.15 * n)
    hi = int(0.85 * n)
    best_start = best_end = -1
    start = None
    for j in range(lo, hi):
        if not covered[j] and start is None:
            start = j
        elif covered[j] and start is not None:
            if (j - start) > (best_end - best_start):
                best_start, best_end = start, j
            start = None
    if start is not None and (hi - start) > (best_end - best_start):
        best_start, best_end = start, hi

    if best_start < 0 or (best_end - best_start) * step < 20:
        return None

    return min_x + (best_start + best_end) / 2 * step


def order_blocks(blocks, page_height: float | None = None) -> None:
    """Blokları okuma sırasına dizer.

    Tek sütun: (y, x). İki sütun: önce sütun üstü içerik (başlık/lead),
    sonra sol sütun (y sırası), sonra sağ sütun (y sırası).
    """
    split_x = detect_column_split_x(blocks, page_height)
    if split_x is None:
        blocks.sort(key=lambda b: (round(b[1], 1), round(b[0], 1)))
        return

    top_margin = 60
    bottom_margin = (page_height - 40) if page_height else float("inf")

    # Sütun sınıflandırması için üst/alt kenar payı bloklarını hariç tut.
    body = [
        b for b in blocks
        if top_margin <= b[1] <= bottom_margin
    ]
    left_min_y = min((b[1] for b in body if b[2] <= split_x), default=float("inf"))
    right_min_y = min((b[1] for b in body if b[0] >= split_x), default=float("inf"))
    if left_min_y == float("inf") or right_min_y == float("inf"):
        blocks.sort(key=lambda b: (round(b[1], 1), round(b[0], 1)))
        return
    col_start_y = max(left_min_y, right_min_y)

    def key(b):
        if b[1] > bottom_margin:  # altbilgi (sayfa numarası) en sona
            group = 3
        elif b[1] < top_margin:  # üstbilgi/koşu başlığı
            group = 0
        elif b[0] < split_x < b[2]:  # tam genişlik (başlık)
            group = 0
        elif b[1] < col_start_y:  # sütunların üstündeki içerik (başlık/lead)
            group = 0
        elif b[0] < split_x:
            group = 1  # sol sütun
        else:
            group = 2  # sağ sütun
        return (group, round(b[1], 1), round(b[0], 1))

    blocks.sort(key=key)


def extract_pdf(pdf_path: Path, two_column: bool = False) -> list[dict]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"Dosya bulunamadı: {pdf_path}")
    doc = fitz.open(pdf_path)
    document_id = slugify_pdf(pdf_path.name)
    records: list[dict] = []
    index = 0

    for page in doc:
        # İçindekiler sayfasını (kapak + TOC) atla. Font bozukluğunu da düzelt.
        page_text = page.get_text().replace("‹", "İ").replace("›", "ı")
        if "İÇİNDEKİLER" in page_text.upper():
            continue
        blocks = [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
        # Okuma sırası: iki sütunlu sayfalarda sütun bilinciyle; tek sütunda (y, x).
        if two_column:
            order_blocks(blocks, page.rect.height)
        else:
            blocks.sort(key=lambda b: (round(b[1], 1), round(b[0], 1)))
        for b in blocks:
            for para in reconstruct_block(b[4]):
                records.append(
                    {
                        "document_id": document_id,
                        "source_file": pdf_path.name,
                        "paragraph_index": index,
                        "text": para,
                        "block_type": "paragraph",
                        "x0": round(b[0], 1),
                    }
                )
                index += 1
    doc.close()
    return records


def filter_repeated(records: list[dict], min_count: int = 3) -> list[dict]:
    """Koşu başlığı/altbilgi gibi sayfa boyunca tekrar eden kayıtları çıkarır."""
    def norm(text: str) -> str:
        return re.sub(r"[\d\s\-]+$", "", (text or "").strip())

    counts = Counter(norm(r["text"]) for r in records)
    return [r for r in records if counts[norm(r["text"])] < min_count]


def filter_adjacent_duplicates(records: list[dict]) -> list[dict]:
    """Ardışık özdeş kayıtları teke düşürür (PDF gölge katmanı çift metin artefaktı)."""
    out: list[dict] = []
    prev: str | None = None
    for r in records:
        text = (r["text"] or "").strip()
        if text == prev:
            continue
        out.append(r)
        prev = text
    return out


def merge_drop_caps(records: list[dict]) -> list[dict]:
    """Ayrı kayıt olarak çıkan gömme harfi (drop cap) sonraki kelimeye ekler.

    "U" + "luslararası Diyabet" -> "Uluslararası Diyabet" gibi. Yalnızca
    tek başına büyük harf kaydı, sonraki kayıt küçük harfle başlıyorsa birleştirilir.
    Sonraki kayıt büyük harfle başlıyorsa (yeri kaymış gömme harf) atılır.
    """
    out: list[dict] = []
    i = 0
    while i < len(records):
        text = (records[i]["text"] or "").strip()
        if len(text) == 1 and text.isupper() and i + 1 < len(records):
            nxt = (records[i + 1]["text"] or "").strip()
            if nxt and nxt[0].islower():
                out.append({**records[i + 1], "text": text + nxt})
                i += 2
                continue
            i += 1  # yeri kaymış gömme harf: at
            continue
        if len(text) == 1 and text.isupper():
            i += 1  # son kayıtta tek başına kalmış gömme harf: at
            continue
        out.append(records[i])
        i += 1
    return out


# Görsel gömme harf (resim) yüzünden ilk harfi tamamen kaybolan kelimeler.
# Drop cap her zaman paragraf başında olduğu için ^ ile eşleştirilir.
MISSING_DROPCAP_FIXES = [
    (re.compile(r"^iyabet"), "Diyabet"),
    (re.compile(r"^okuz\b"), "Dokuz"),
    (re.compile(r"^ir çok\b"), "Bir çok"),
    (re.compile(r"^erhiz\b"), "Perhiz"),
    (re.compile(r"^an şekerinde\b"), "Kan şekerinde"),
    (re.compile(r"^irketlerin\b"), "Şirketlerin"),
    (re.compile(r"^an şekeri\b"), "Kan şekeri"),
    (re.compile(r"^öbrek\b"), "Böbrek"),
    (re.compile(r"^nlu\b"), "Unlu"),
    (re.compile(r"^ir kişinin\b"), "Bir kişinin"),
    (re.compile(r"^az yaklaşıyor\b"), "Yaz yaklaşıyor"),
    (re.compile(r"^zellikle\b"), "Özellikle"),
    (re.compile(r"^odrum\b"), "Bodrum"),
    (re.compile(r"^ntakya\b"), "Antakya"),
    # Sayı 43: "Seyahat" kontrol listesi (tamamı büyük harf gömme başlıklar)
    (re.compile(r"^OKTORUNUZA\b"), "DOKTORUNUZA"),
    (re.compile(r"^ANINIZA\b"), "YANINIZA"),
    (re.compile(r"^LUKAGON\b"), "GLUKAGON"),
    (re.compile(r"^NSÜLİN\b"), "İNSÜLİN"),
    (re.compile(r"^EDAVİ\b"), "TEDAVİ"),
    (re.compile(r"^AHAT\b"), "RAHAT"),
    (re.compile(r"^OLA\b"), "YOLA"),
    (re.compile(r"^OLDA\b"), "YOLDA"),
    (re.compile(r"^AAT\b"), "SAAT"),
]


def fix_missing_dropcaps(records: list[dict]) -> list[dict]:
    """Görsel gömme harf yüzünden ilk harfi kaybolan kelimeleri onarır."""
    out: list[dict] = []
    for r in records:
        text = r["text"]
        for pat, repl in MISSING_DROPCAP_FIXES:
            text = pat.sub(repl, text, count=1)
        out.append({**r, "text": text} if text != r["text"] else r)
    return out


_GARBLED_REPEAT = re.compile(r"(?i)\b(\w{1,3})\b(?:\s+\1){2,}")


def _is_garbled(text: str) -> bool:
    """Dekoratif yazı/harf aralığı yüzünden bozulmuş kısa metinleri tanır.

    Dergi grafiklerindeki "ko ko ko", "D fı y kötü ko6" gibi tekrarlı ya da
    harf aralıklı metinler RAG'a değer katmaz.
    """
    s = (text or "").strip()
    if not s or len(s) > 60:
        return False
    if _GARBLED_REPEAT.search(s):
        return True
    toks = re.findall(r"\S+", s)
    if len(toks) >= 4:
        short = sum(1 for t in toks if len(t) <= 2)
        if short / len(toks) >= 0.5:
            return True
    return False


def filter_garbled(records: list[dict]) -> list[dict]:
    """Dekoratif/harf aralıklı bozuk kısa metinleri çıkarır."""
    return [r for r in records if not _is_garbled(r["text"])]


def _is_upper_fragment(text: str) -> bool:
    """Tamamı büyük harfli kısa başlık parçası (dergi başlıkları satırlara bölünür).

    Tek harf (drop cap) başlık parçası sayılmaz; böylece başlık birleştirmede
    gömme harfin yanlışlıkla başlığa eklenmesi engellenir.
    """
    if not text or len(text) < 2 or len(text) > 60:
        return False
    return text.isupper()


def merge_title_fragments(records: list[dict]) -> list[dict]:
    """Satırlara bölünmüş tamamı-büyük dergi başlıklarını tek kayıtta birleştirir.

    Yan yana sütunlardaki sözlük terimleri (farklı x koordinatı) birleştirilmez.
    """
    COLUMN_GAP = 100  # sütunlar arası x farkı; üstü farklı sütun sayılır.
    out: list[dict] = []
    i = 0
    while i < len(records):
        text = (records[i]["text"] or "").strip()
        if not _is_upper_fragment(text):
            out.append(records[i])
            i += 1
            continue
        frag = [records[i]]
        anchor_x = records[i].get("x0")
        j = i + 1
        while j < len(records) and _is_upper_fragment((records[j]["text"] or "").strip()):
            xj = records[j].get("x0")
            if anchor_x is not None and xj is not None and abs(xj - anchor_x) > COLUMN_GAP:
                break
            frag.append(records[j])
            j += 1
        if len(frag) == 1:
            out.append(frag[0])
        else:
            out.append(
                {**frag[0], "text": " ".join((r["text"] or "").strip() for r in frag)}
            )
        i = j
    return out


def filter_dose(records: list[dict]) -> list[dict]:
    """Doz/ünite talimatı içeren paragrafları çıkarır (hasta RAG'ına girmemeli)."""
    return [r for r in records if not DOSE_INSTRUCTION.search(r["text"])]


def filter_drop_patterns(records: list[dict], patterns: list[str]) -> list[dict]:
    """Verilen regex'lerden herhangi biriyle eşleşen kayıtları çıkarır."""
    compiled = [re.compile(p, re.IGNORECASE) for p in patterns]
    return [r for r in records
            if not any(p.search(r["text"]) for p in compiled)]


def filter_redact(records: list[dict], patterns: list[str]) -> list[dict]:
    """Verilen regex'lerle eşleşen alt dizileri metinden siler (e-posta/tel)."""
    compiled = [re.compile(p) for p in patterns]
    out: list[dict] = []
    for r in records:
        text = r["text"]
        for p in compiled:
            text = p.sub("", text)
        text = re.sub(r"\s{2,}", " ", text).strip()
        if text:
            out.append({**r, "text": text})
    return out


LETTER_HEADING = re.compile(r"^([A-ZÇĞİÖŞÜ])\.\s+\S")
NUMBERED_HEADING = re.compile(r"^\d+(?:\.\d+)+\.?\s+\S")


def filter_letter_sections(records: list[dict], drop_letters: str) -> list[dict]:
    """Belirtilen tek-harfli bölümleri (A. AMAÇ, B. HEDEFLER...) ve içeriklerini atlar.

    Eğitimci rehberlerindeki meta bölümler (AMAÇ, ÖĞRENİM HEDEFLERİ, YÖNTEM,
    MATERYALLER, DEĞERLENDİRME) hasta RAG'ına değer katmaz; numaralı içerik
    bölümleri (1.1, 3.2.1) ve F (mesajlar) / G (özet) korunur.
    """
    drop = set(drop_letters.upper())
    out: list[dict] = []
    skipping = False
    for r in records:
        t = (r["text"] or "").strip()
        m = LETTER_HEADING.match(t)
        if m:
            if m.group(1).upper() in drop:
                skipping = True
                continue
            skipping = False
            out.append(r)
            continue
        if NUMBERED_HEADING.match(t) or re.match(r"^MODÜL\b", t, re.IGNORECASE):
            skipping = False
            out.append(r)
            continue
        if skipping:
            continue
        out.append(r)
    return out


def save_jsonl(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF'ten paragraf JSONL çıkar.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--skip-block",
        action="append",
        nargs=2,
        metavar=("START_REGEX", "END_REGEX"),
        default=[],
        help="Atlanacak bölüm aralığı (başlangıç ve bitiş başlık regex'i). Tekrarlanabilir.",
    )
    parser.add_argument(
        "--skip-until",
        metavar="REGEX",
        default=None,
        help="Bu regex eşleşene kadar baştaki kayıtları atla (ön kapak/komiteler).",
    )
    parser.add_argument(
        "--skip-after",
        metavar="REGEX",
        default=None,
        help="Bu regex eşleştikten sonrasını (dahil) atla (kısaltmalar/kaynakça kuyruğu).",
    )
    parser.add_argument(
        "--dose-filter",
        action="store_true",
        help="Doz/ünite talimatı içeren paragrafları çıkar.",
    )
    parser.add_argument(
        "--drop-letter-sections",
        metavar="HARFLER",
        default=None,
        help="Tek-harfli meta bölümleri atla (ör. ABDEH = AMAÇ/HEDEFLER/YÖNTEM/MATERYAL/DEĞERLENDİRME).",
    )
    parser.add_argument(
        "--fix-font",
        action="store_true",
        help="PDF font bozulmasını düzelt (ligatürler + 't' glifi → 3/$/)).",
    )
    parser.add_argument(
        "--drop-pattern",
        action="append",
        metavar="REGEX",
        default=[],
        help="Eşleşen kayıtları at (reklam/künye/ISSN gibi gürültü). Tekrarlanabilir.",
    )
    parser.add_argument(
        "--redact",
        action="append",
        metavar="REGEX",
        default=[],
        help="Eşleşen alt dizileri metinden sil (e-posta/tel gibi). Tekrarlanabilir.",
    )
    parser.add_argument(
        "--merge-title-fragments",
        action="store_true",
        help="Satırlara bölünmüş tamamı-büyük dergi başlıklarını tek kayıtta birleştir.",
    )
    parser.add_argument(
        "--two-column",
        action="store_true",
        help="İki sütunlu sayfaları sütun bilinciyle oku (sol sütun, sonra sağ sütun).",
    )
    args = parser.parse_args()

    records = extract_pdf(args.input, two_column=args.two_column)
    if args.fix_font:
        records = [{**r, "text": fix_pdf_font(r["text"])} for r in records]
    records = filter_adjacent_duplicates(records)
    records = merge_drop_caps(records)
    records = fix_missing_dropcaps(records)
    if args.redact:
        records = filter_redact(records, args.redact)
    if args.skip_until:
        pat = re.compile(args.skip_until, re.IGNORECASE)
        for i, r in enumerate(records):
            if pat.search(r["text"]):
                records = records[i:]
                break
        else:
            records = []
    records = clean_records(records, args.skip_block)
    if args.drop_letter_sections:
        records = filter_letter_sections(records, args.drop_letter_sections)
    if args.drop_pattern:
        records = filter_drop_patterns(records, args.drop_pattern)
    if args.dose_filter:
        records = filter_dose(records)
    if args.skip_after:
        pat = re.compile(args.skip_after, re.IGNORECASE)
        for i, r in enumerate(records):
            if pat.search(r["text"]):
                records = records[:i]
                break
    records = filter_repeated(records)
    if args.merge_title_fragments:
        records = merge_title_fragments(records)
    records = filter_garbled(records)
    output = args.output or Path("data/processed") / f"{slugify_pdf(args.input.name)}.raw.jsonl"
    save_jsonl(records, output)
    print(f"Kayıt: {len(records)}")
    print(f"Kaydedildi: {output}")


if __name__ == "__main__":
    main()
