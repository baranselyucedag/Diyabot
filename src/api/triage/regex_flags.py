"""Adım 2 — Regex / morfoloji bayrakları.

Sayıdan bağımsız deterministik dil katmanı:
  - EMERGENCY: bilinç/bayılma/göğüs/nefes/112 (112 canned)
  - REFUSE: doz / tanı / reçete talebi (112 yok; ayrı canned)
  - JAILBREAK: prompt injection / rol yapma → seviye REFUSE, ayrı canned
  - YELLOW: yumuşak uyarı kalıpları (RAG devam; soft sinyal)

Glukoz sayısı eşikleri burada YOK (Adım 1 numeric).
Örtük skor / LLM / score fusion burada YOK.

Pattern'ler `norm` sonrası ASCII metinde; negatif lookahead ile FP azaltılır.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from src.api.triage.text_utils import norm

RegexLevel = Literal["EMERGENCY", "REFUSE", "YELLOW"]

# Jailbreak flag etiketleri — canned_response güvenlik metni için
JAILBREAK_FLAG_LABELS = frozenset(
    {"sistem_prompt", "prompt_yok_say", "doktor_gibi"}
)

# ---------------------------------------------------------------------------
# Hard EMERGENCY — sayı şart değil (acil dil / bilinç / kardiyopulmoner)
# ---------------------------------------------------------------------------
_EMERGENCY: list[tuple[str, str]] = [
    (r"(?<![0-9])112(?![0-9])", "112"),
    (r"(?<![a-z])bayil", "bayilma"),
    (r"(?<![a-z])baygin", "bayginlik"),
    (r"(?<![a-z])bilinc", "bilinc"),
    (r"cevap\s*vermiyor", "cevap_vermiyor"),
    # konusamiyorum / konusamiyor / konusamaz…
    (r"konus(amiyor|amaz)", "konusamama"),
    (r"konus(masi)?\s*bozul", "konusma_bozulmasi"),
    (r"kendinde\s*degil", "kendinde_degil"),  # #3 boundary bilinçli atlandı
    # gogus / gogsum / gogusum… + agr*
    (r"gog\w*\s*agr", "gogus_agrisi"),
    (r"nefes\s*alamiyorum", "nefes_alamama"),
    (r"nefes\s*darligi", "nefes_darligi"),
]

# ---------------------------------------------------------------------------
# REFUSE — doz / tanı / reçete (üçlü). Eğitim soruları ("doz nedir?") kaçınır.
# ---------------------------------------------------------------------------
_REFUSE_DOSE: list[tuple[str, str]] = [
    (r"kac\s*unite", "kac_unite"),
    (
        r"doz(um|unu|u|unu)?\s*(hesapla|ayarla|artir|azalt|onayla|yaz|soyle)",
        "doz_talep",
    ),
    (r"dozum.*(iki\s*)?kat", "doz_iki_kat"),
    (
        r"(insulin|metformin|ozempic|glukagon|sulfonilure|ilac).{0,48}"
        r"(kac\s*unite|doz|artir|azalt|onayla|hesapla|ayarla|iki\s*kat)",
        "ilac_doz_talep",
    ),
    (
        r"(kac\s*unite|doz|artir|azalt|onayla|hesapla|ayarla|iki\s*kat).{0,48}"
        r"(insulin|metformin|ozempic|glukagon|sulfonilure)",
        "ilac_doz_talep_ters",
    ),
    # "2000 mg'a çık" tipi doz artırma (sayı zorunlu)
    (r"\d+\s*mg'?a\s*cik", "mg_cikarma"),
    (r"recetesiz", "recetesiz"),
    (r"doz\s*semasi", "doz_semasi"),
]

_REFUSE_DIAGNOSIS: list[tuple[str, str]] = [
    (r"tani\s*koy", "tani_koy"),
    (r"tanisi\s*koy", "tanisi_koy"),
    (r"yetmezligi\s*var\s*mi", "yetmezlik_tani"),
    (r"evrem\s*nedir", "evre_tani"),
    # tıbbi bağlam zorunlu; "bende bu telefon var mı" FP olmasın
    (
        r"bende\s+.{0,40}?(yetmezlik|diyabet|hastalik|bobrek|kanser).{0,20}?var\s*mi",
        "bende_var_mi_tani",
    ),
]

_REFUSE_PRESCRIBE: list[tuple[str, str]] = [
    (r"ilac\s*(kombinasyon(u)?)?\s*yaz", "ilac_yaz"),
    (r"kombinasyon(u)?\s*yaz", "kombinasyon_yaz"),
    (r"recete\s*yaz", "recete_yaz"),
]

# Birleşik REFUSE (tek geçiş)
_REFUSE: list[tuple[str, str]] = (
    _REFUSE_DOSE + _REFUSE_DIAGNOSIS + _REFUSE_PRESCRIBE
)

# ---------------------------------------------------------------------------
# JAILBREAK / güvenlik — seviye REFUSE, canned ayrı
# ---------------------------------------------------------------------------
_JAILBREAK: list[tuple[str, str]] = [
    (r"sistem\s*prompt", "sistem_prompt"),
    (r"prompt(un|u)?\s*yok\s*say", "prompt_yok_say"),
    # iltifat ("doktor gibi konuşuyorsun") tetiklemesin
    (r"doktor\s*gibi\s*(davran|rol\s*yap|yaz)", "doktor_gibi"),
]

# ---------------------------------------------------------------------------
# Soft YELLOW — hard değil; fusion skoruna katkı (level üretmez)
# Pattern'ler norm() sonrası ASCII metinde (ı→i, ş→s, ğ→g, ü→u, ö→o, ç→c).
# HARD (bayil/bilinc/nefes/gogus) ve REFUSE (doz talep) ile çakışmayı bilerek kaçın.
# ---------------------------------------------------------------------------
_SOFT_YELLOW: list[tuple[str, str]] = [
    # --- Şiddet sözcükleri (eski 4'ün genişletilmiş çekirdeği) ---
    (r"(?<![a-z])cok\s*yuksek", "cok_yuksek"),
    (r"(?<![a-z])asiri\s*yuksek", "cok_yuksek"),
    (r"(?<![a-z])cok\s*dusuk", "cok_dusuk"),
    (r"(?<![a-z])asiri\s*dusuk", "cok_dusuk"),
    # hızla / ani / birden + yön; firladi/coktu (yazım: firladi, fırladı→firladi)
    (
        r"(hizla|ani|birden)\s*(yukseliyor|yukseldi|dusuyor|dustu|firliyor|firladi|cokuyor|coktu)",
        "hizli_degisim",
    ),
    (r"(grafik|okuma|sensor|cgm).{0,24}(hizla|ani|birden).{0,16}(yuksel|dus|firla|cok)", "hizli_degisim"),
    (r"(alarm|uyari)\s*(verdi|caldi|geldi)", "hizli_degisim"),
    # --- Süre / persistans (sadece "üç gündür" değil) ---
    # Flag adı geriye uyum: uc_gundur = süre ailesi
    (
        r"(?<![a-z])("
        r"\d{1,2}|"
        r"bir|iki|uc|dort|bes|alti|yedi|sekiz|dokuz|on|"
        r"birkac|kac"
        r")\s*gun(dur|den\s*beri)",
        "uc_gundur",
    ),
    (r"(?<![a-z])(hafta|ay)(dir|dan\s*beri)", "uc_gundur"),
    (r"uzun\s*suredir", "uc_gundur"),
    (r"bir\s*turlu\s*(duzel|inm|gec|kontrol)", "uc_gundur"),
    (r"hala\s*(gec|duzel|inm)m?iyor", "uc_gundur"),
    (r"(duzelmiyor|inmiyor|gecmiyor|cikmiyor)", "uc_gundur"),
    # --- Hipo dili (sayı yok; bilinç/bayılma HARD'da) ---
    (r"seker(im|i|in)?\s*(dustu|dusuyor|dusmus|dusuk\s*(cikti|geldi|oldu))", "hipo_dil"),
    (r"(?<![a-z])hipo(\s*(oldum|oldu|oluyor|yapti|yasiyorum))?", "hipo_dil"),
    (r"ellerim\s*titriyor", "hipo_dil"),
    (r"titriyor(um|lar)?", "hipo_dil"),
    (r"aclik\s*hissi", "hipo_dil"),
    (r"seker(im|i)?\s*dusuk", "hipo_dil"),
    # --- Hiper dili (sayı yok; DKA+kusma+bilinç HARD/numeric) ---
    (r"seker(im|i|in)?\s*(yukseldi|yukseliyor|yukselmis|yuksek\s*(cikti|geldi|oldu))", "hiper_dil"),
    (r"(cok\s*)?sus(uyor|adim|ama)\b", "hiper_dil"),
    (r"sik\s*idrar", "hiper_dil"),
    (r"agz(im|i)?\s*kur(udu|uyor)", "hiper_dil"),
    # --- Fonksiyonel bozulma (bayılma/bilinç HARD kelimesi yok) ---
    (r"yol\s*kenar\w*\s*cek", "fonksiyonel_bozulma"),
    (r"(araba|araci|direksiyon).{0,40}(cekmek\s*zorunda|dur(mak|dum)|cektim)", "fonksiyonel_bozulma"),
    (r"bacaklar(im)?\s*(tutmadi|titredi|gucsuz)", "fonksiyonel_bozulma"),
    (r"(tutamadim|dusurdum|odaklanamadim|karistirdim)", "fonksiyonel_bozulma"),
    (r"(toplantidan|toplantida|isteyken).{0,40}(cikmak\s*zorunda|odaklan)", "fonksiyonel_bozulma"),
    (r"kendimde\s*degil(im|dim)?", "fonksiyonel_bozulma"),
    (r"fenalas(tim|iyor|di|tik)", "fonksiyonel_bozulma"),
    # --- Örüntü / tekrar (süre kelimesi şart değil) ---
    (r"(bu\s*ay|son\s*zaman|gecen\s*hafta).{0,40}(ucuncu|ikinci|tekrar|yine|hep)", "oruntu_tekrar"),
    (r"her\s*(spor|antrenman|yemek|sabah).{0,24}(sonra|sonrasi)", "oruntu_tekrar"),
    (r"(tekrar|yine)\s*(oldu|oluyor|ayni)", "oruntu_tekrar"),
    (r"kac\s*kez\s*(oldu|yasiyorum|tekrar)", "oruntu_tekrar"),
    # --- İlaç/insülin uygulama hatası (olmuş bitmiş; doz TALEBİ değil) ---
    (r"(iki\s*kat|cift)\s*(yaptim|ictim|aldim|vurdum)", "uygulama_hatasi"),
    (r"(igneyi|ilaci|ilacimi|insulini|hap(i|imi)?)\s*unut", "uygulama_hatasi"),
    (r"yanlis\s*(kalem|doz|insulin|igneyi?)", "uygulama_hatasi"),
    (r"(olcmeden|tahminen)\s*(doz|insulin|yaptim)", "uygulama_hatasi"),
    (r"(sabah|aksam)\s*(ilac|hap|insulin).{0,24}(yanlislikla|tekrar|bir\s*daha)", "uygulama_hatasi"),
    (r"fazla\s*(insulin|doz|hap)\s*(yaptim|aldim|ictim)", "uygulama_hatasi"),
    # --- Diyabet tükenmişliği / bakımı bırakma ---
    (r"(olcmekten|diyabetten|kontrolden)\s*(biktim|yoruldum|usandim)", "bakim_birakma"),
    (r"(kontrolu|tedaviyi|olcmeyi|diyabeti)\s*birak", "bakim_birakma"),
    (r"(umursamiyorum|bos\s*verdim|bosverdim)", "bakim_birakma"),
    (r"doktora\s*gitme(yi)?\s*birak", "bakim_birakma"),
    (r"artik\s*(olcmiyorum|gitmiyorum|umursamiyorum)", "bakim_birakma"),
    # --- Gece / örtük hipo ---
    (r"yastig(im)?\s*.{0,16}(ter|islak)", "gece_hipo"),
    (r"(gece|sabah).{0,24}(terleyerek|terli)\s*uyan", "gece_hipo"),
    (r"kabus.{0,20}(ter|uyan)", "gece_hipo"),
    (r"esim\s*.{0,32}(tuhaf\s*konus|gece).{0,24}(soyledi|dedi)", "gece_hipo"),
    (r"uyaninca\s*(basim\s*agri|saskin)", "gece_hipo"),
    # --- Ayak / yara / enjeksiyon yeri ---
    (r"ayak.{0,32}(yara|iyilesm|kokuyor|renk)", "ayak_yara"),
    (r"(yara|sivirik).{0,24}iyilesm", "ayak_yara"),
    (r"(parmak|ayak).{0,24}(renk|mor|siyah|soluk)", "ayak_yara"),
    (r"(enjeksiyon|igne)\s*yer.{0,24}(sert|kizar|sis|iltihap)", "ayak_yara"),
    # --- Hasta günü (sick-day) kafa karışıklığı ---
    (r"(grip|ates|ishal|kusma|mide\s*bulant).{0,40}(seker|ilac|olcum|ne\s*yap)", "hasta_gunu"),
    (r"(hastayken|hasta\s*oldugumda|hastayim).{0,40}(seker|ilac)", "hasta_gunu"),
    (r"olcum(ler)?\s*.{0,16}(tutmuyor|karisik|oynam)", "hasta_gunu"),
    # --- Egzersiz sonrası gecikmiş etki ---
    (r"spordan\s*(saatler\s*)?sonra", "egzersiz_gecikmeli"),
    (r"(antrenman|kosu|egzersiz).{0,32}(sonra|sonrasi).{0,32}(fenala|tuhaf|dus|gece)", "egzersiz_gecikmeli"),
    (r"gecikmis\s*(hipo|seker\s*dus)", "egzersiz_gecikmeli"),
    # --- Alkol ---
    (r"(ictim|icki|alkol).{0,40}(seker|hipo|tuhaf|dus)", "alkol_risk"),
    (r"(seker|hipo).{0,40}(ictim|icki|alkol)", "alkol_risk"),
    # --- Gebelik / oruç ihmal + semptom ---
    (r"(hamile|gebe|gebeyim).{0,48}(unut|kotu|bas\s*don|seker)", "gebelik_oruc"),
    (r"oruc.{0,40}(bas\s*don|acmadim|kotu|fenala|seker)", "gebelik_oruc"),
    (r"sahur.{0,32}(unut|ilac|insulin)", "gebelik_oruc"),
    # --- İkinci el / yakını ---
    (r"(annem|babam|esim|kardesim|dedem|ninem).{0,40}(seker|hipo|tuhaf)", "ikinci_el"),
    (r"(anne|baba|es|kardes)m(in|in)?\s*seker", "ikinci_el"),
    # --- Cihaz güvensizliği ---
    (r"(sensor|cgm|pompa|cihaz|olcum\s*cihazi).{0,32}(yalan|bozuk|garip|inanmiyorum|yanlis)", "cihaz_guvensizlik"),
    (r"(yalan\s*soyluyor|bozuk\s*olmali|inanmiyorum\s*sonuca)", "cihaz_guvensizlik"),
    # --- Normalize / alışkanlık dili ---
    (r"(bana\s*)?hep\s*(olur|oluyor)", "normalize_dil"),
    (r"aliskin(im|iz)?", "normalize_dil"),
    (r"normal\s*(benim\s*icin|bende|galiba)", "normalize_dil"),
    (r"sorun\s*etmiyorum", "normalize_dil"),
    # --- Hekim / bakım kaçınma ---
    (r"(doktor|hastane).{0,24}(gitmek\s*istemiyorum|gitmedim|guvenmiyorum)", "hekim_kacinma"),
    (r"(hastaneye|doktora)\s*(gitmek\s*)?istemiyorum", "hekim_kacinma"),
    (r"kendim\s*halledeyim", "hekim_kacinma"),
    # --- İlaç yan etki (doz değiştirme TALEBİ değil) ---
    (r"midem\s*bulaniyor", "ilac_yan_etki"),
    (r"(?<![a-z])ishal(?![a-z])", "ilac_yan_etki"),
    (r"sisiyorum|sislik", "ilac_yan_etki"),
    (r"mide\s*boz", "ilac_yan_etki"),
    (r"(metformin|ozempic|insulin|ilac).{0,32}(mide|ishal|bulanti|yan\s*etki)", "ilac_yan_etki"),
]

# Soft ağırlık kilidi (soft_label_set precision + klinik prior / Adım C).
# Aralık: yeni flag'ler sadece [0.30, 0.85]. Fusion soft_signal = sum(w) clamp [0,1].
# TODO (sonraki faz): soft regex coverage büyüt (aile içi pattern + eksik aileler);
#   her yeni aile: label set → shrunken prec → prior → bu aralığa kilitle.
SOFT_YELLOW_WEIGHTS: dict[str, float] = {
    # soft-4 (ölçüm + log-odds; dokunma)
    "hizli_degisim": 0.85,
    "cok_yuksek": 0.50,
    "uc_gundur": 0.58,  # süre ailesi; prec yüksek ama tek başına acil değil
    "cok_dusuk": 0.35,
    # geniş soft aileler (prec + klinik prior)
    "uygulama_hatasi": 0.80,
    "hipo_dil": 0.80,
    "fonksiyonel_bozulma": 0.75,
    "ayak_yara": 0.75,
    "gebelik_oruc": 0.70,
    "hiper_dil": 0.65,
    "hasta_gunu": 0.65,
    "gece_hipo": 0.65,
    "bakim_birakma": 0.60,
    "egzersiz_gecikmeli": 0.55,
    "alkol_risk": 0.55,
    "ilac_yan_etki": 0.55,
    "oruntu_tekrar": 0.55,
    "cihaz_guvensizlik": 0.50,
    "ikinci_el": 0.45,
    "hekim_kacinma": 0.45,
    "normalize_dil": 0.32,
}


@dataclass
class RegexTriageResult:
    """Regex katmanı çıktısı. level=None → bu katman karar vermedi."""

    level: RegexLevel | None
    reason: str
    flags: list[str] = field(default_factory=list)
    all_flags: list[str] = field(default_factory=list)
    suppressed: list[tuple[str, list[str]]] = field(default_factory=list)


def _match_flags(norm_text: str, rules: list[tuple[str, str]]) -> list[str]:
    """Eşleşen kural etiketlerini döner (sıra korunur, tekil)."""
    hit: list[str] = []
    seen: set[str] = set()
    for pat, label in rules:
        if re.search(pat, norm_text) and label not in seen:
            hit.append(label)
            seen.add(label)
    return hit


def _dedupe_preserve(items: list[str]) -> list[str]:
    """Sıra koruyarak tekilleştir."""
    out: list[str] = []
    seen: set[str] = set()
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def flag_emergency(norm_text: str) -> list[str]:
    """Hard EMERGENCY bayrakları."""
    return _match_flags(norm_text, _EMERGENCY)


def flag_refuse(norm_text: str) -> list[str]:
    """REFUSE bayrakları (doz + tanı + reçete) — tek geçiş."""
    return _match_flags(norm_text, _REFUSE)


def flag_jailbreak(norm_text: str) -> list[str]:
    """Jailbreak / prompt-injection bayrakları."""
    return _match_flags(norm_text, _JAILBREAK)


def flag_soft_yellow(norm_text: str) -> list[str]:
    """Yumuşak YELLOW uyarı bayrakları."""
    return _match_flags(norm_text, _SOFT_YELLOW)


def evaluate_regex_flags(message: str) -> RegexTriageResult | None:
    """Regex/morfoloji triage. Eşleşme yoksa None.

    Öncelik (katman içi): EMERGENCY > REFUSE > JAILBREAK.
    Soft YELLOW tek başına level üretmez (Adım 3 fusion karar verir):
    level=None, flags=soft, reason=soft_flags_only_for_fusion.
    JAILBREAK seviye olarak REFUSE döner.
    """
    text = message or ""
    if not text.strip():
        return None

    n = norm(text)
    emerg = flag_emergency(n)
    refuse = flag_refuse(n)
    jail = flag_jailbreak(n)
    soft = flag_soft_yellow(n)

    all_flags = _dedupe_preserve(emerg + refuse + jail + soft)
    if not all_flags:
        return None

    suppressed: list[tuple[str, list[str]]] = []

    if emerg:
        if refuse:
            suppressed.append(("REFUSE", refuse))
        if jail:
            suppressed.append(("JAILBREAK", jail))
        if soft:
            suppressed.append(("YELLOW", soft))
        return RegexTriageResult(
            level="EMERGENCY",
            reason=f"regex EMERGENCY: {', '.join(emerg)}",
            flags=emerg,
            all_flags=all_flags,
            suppressed=suppressed,
        )

    if refuse:
        if jail:
            suppressed.append(("JAILBREAK", jail))
        if soft:
            suppressed.append(("YELLOW", soft))
        return RegexTriageResult(
            level="REFUSE",
            reason=f"regex REFUSE: {', '.join(refuse)}",
            flags=refuse,
            all_flags=all_flags,
            suppressed=suppressed,
        )

    if jail:
        if soft:
            suppressed.append(("YELLOW", soft))
        return RegexTriageResult(
            level="REFUSE",
            reason=f"regex JAILBREAK->REFUSE: {', '.join(jail)}",
            flags=jail,
            all_flags=all_flags,
            suppressed=suppressed,
        )

    # Soft only → fusion için flag; seviye None (Adım 3 migration)
    return RegexTriageResult(
        level=None,
        reason="soft_flags_only_for_fusion",
        flags=soft,
        all_flags=all_flags,
        suppressed=suppressed,
    )
