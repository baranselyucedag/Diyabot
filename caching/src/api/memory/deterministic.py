"""Deterministik kurallar — LLM'siz, test edilebilir, hızlı.

Plan dokümanına (cahing_mimari_tasarim.md §3) birebir uyumludur:
- CONFLICT_RULES, FIELD_TO_NOTE_CATEGORY, STALENESS_KEYWORD_RULES, TRIAGE_FIELD_RISK
  sabitleri plan'daki gibidir.
- detect_stale_notes SAFTIR: yan etki (disk yazma / log) üretmez, sadece tespit
  listesi döner. Yazma + loglama çağıranın (maintenance.py) sorumluluğundadır.
"""

from __future__ import annotations

import re
from typing import Optional

from src.api.memory.config import MEMORY_CONFIG
from src.api.memory.models import (
    CandidateNote,
    Note,
    PendingConflictStatus,
    Profile,
    Turn,
)
from src.api.memory.storage import load_pending_conflicts
from src.api.memory.timeutil import utcnow


# ---------------------------------------------------------------------------
# Sabitler (plan dosyasından birebir)
# ---------------------------------------------------------------------------

CONFLICT_RULES = {
    "medications": {
        "add": {"duplicate_name": {"action": "FLAG_PENDING", "conflict_type": "DUPLICATE_MED"}},
        "remove": {"default": {"action": "REWRITE_AS_UPDATE"}},
    },
    "complications": {
        "add": {"duplicate": {"action": "IGNORE"}},
    },
    "allergies": {
        "add": {"duplicate": {"action": "FLAG_PENDING", "conflict_type": "DUPLICATE_ALLERGY"}},
    },
    "goals": {
        "update": {"default": {"action": "LATEST_WINS"}},
    },
}

FIELD_TO_NOTE_CATEGORY = {
    "medications": ["observation"],
    "goals": ["plan", "advice"],
    "complications": ["symptom"],
    "monitoring": ["measurement"],
    "allergies": ["observation"],
}

# Bilinen sınır (P3, bilinçli ödünleşim): keyword eşleşmesi substring tabanlıdır;
# Türkçe morfoloji/stemming ve bağlam analizi yoktur ("değiştirdim" her bağlamda
# yakalanır). LLM tabanlı doğrulamaya geçilmeden giderilemez — kabul edildi.
STALENESS_KEYWORD_RULES = {
    "observation": [   # ilaç / tedavi notları
        (["bıraktım", "durduruldu", "kesildi", "kullanmıyorum", "bıraktırdım", "sıfırlandı", "değiştirdim"], "status_conflict_if_active"),
        (["dozu arttır", "dozu azalt", "doz değişti"], "dose_conflict"),
    ],
    "plan": [          # hedef notları
        (["hedef değiştirdim", "yeni hedef", "artık hedefim"], "goal_replaced"),
    ],
    "advice": [        # tavsiye/hedef notları
        (["hedef değiştirdim", "yeni hedef", "artık hedefim"], "goal_replaced"),
    ],
    "symptom": [       # komplikasyon / semptom notları
        (["öneldi", "geçti", "iyileşti"], "resolved_conflict"),
    ],
}

# v4.1: Profil alanı risk haritası (mesaj triage'i DEĞİL, profil alanı riski)
TRIAGE_FIELD_RISK = {
    "medications": {"is_high_risk": True},
    "complications": {"is_high_risk": True},
    "goals": {"is_high_risk": False},
    "monitoring": {"is_high_risk": False},
    "allergies": {"is_high_risk": True},
}


def triage_classify(field: str) -> dict:
    """Profil alanı bazlı risk — mesaj triage'i DEĞİL."""
    return TRIAGE_FIELD_RISK.get(field, {"is_high_risk": True})  # bilinmeyen = güvenli tarafta


# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------


def normalize(text: str) -> str:
    """Türkçe ASCII fold (ı→i, ş→s, vb.) — src.api.triage.text_utils.norm ile aynı."""
    t = text or ""
    t = t.replace("İ", "i").replace("I", "i")
    t = t.casefold().replace("\u0307", "")  # combining dot above
    return (
        t.replace("ı", "i")
        .replace("ğ", "g")
        .replace("ü", "u")
        .replace("ş", "s")
        .replace("ö", "o")
        .replace("ç", "c")
    )


def find_turn_by_id(turn_id: str, turns: list[Turn]) -> Optional[Turn]:
    """ID ile turn bul."""
    for turn in turns:
        if turn.turn_id == turn_id:
            return turn
    return None


# Türkçe stopword'ler — grounding overlap hesabında anlam taşımayan bağlaç/ek
# kelimeleri dışarıda bırakırız; aksi halde "ve", "ile", "bir" gibi kelimeler
# örtüşme oranını yapay olarak şişirip yanlış "grounded" kararlarına yol açar.
TURKISH_STOPWORDS: frozenset[str] = frozenset(
    {
        "ve", "ile", "bir", "bu", "şu", "o", "ben", "sen", "biz", "siz",
        "de", "da", "ki", "mi", "mı", "mu", "mü", "ama", "fakat", "lakin",
        "çok", "az", "daha", "en", "gibi", "kadar", "için", "sonra", "önce",
        "evet", "hayır", "acaba", "mıyım", "misin", "değil", "var", "yok",
        "nasıl", "neden", "niye", "hangi", "kaç",
    }
)
NORMALIZED_TURKISH_STOPWORDS = frozenset(normalize(word) for word in TURKISH_STOPWORDS)


# ---------------------------------------------------------------------------
# Grounding doğrulama
# ---------------------------------------------------------------------------


def word_set(text: str) -> set[str]:
    """Normalize edilmiş, stopword'lerden arındırılmış kelime kümesi."""
    norm_text = normalize(text)
    return set(re.findall(r"[\w]+", norm_text)) - NORMALIZED_TURKISH_STOPWORDS


def verify_grounding_deterministic(
    note: CandidateNote,
    source_turns: list[Turn],
    min_overlap: Optional[float] = None,
) -> dict:
    """
    Kelime örtüşmesi + source_turn_id validasyonu.

    Plan (§3.b): source_turn_id MUTLAKA zorunludur. NULL/eksise veya
    source_turns içinde yoksa not reddedilir.

    Returns:
        {"grounded": bool, "overlap": float, "source_turn_found": bool, "reason": str}
    """
    if min_overlap is None:
        min_overlap = MEMORY_CONFIG["grounding_min_word_overlap"]

    # source_turn_id zorunlu validasyonu
    if not note.source_turn_id:
        return {
            "grounded": False,
            "overlap": 0.0,
            "source_turn_found": False,
            "reason": "no_source_id",
        }

    source_turn = find_turn_by_id(note.source_turn_id, source_turns)
    if source_turn is None:
        return {
            "grounded": False,
            "overlap": 0.0,
            "source_turn_found": False,
            "reason": "source_id_not_in_context",
        }

    source_turn_found = True

    note_words = word_set(note.content)
    if not note_words:
        return {
            "grounded": False,
            "overlap": 0.0,
            "source_turn_found": True,
            "reason": "note_empty",
        }

    turn_words = word_set(source_turn.content)
    if not turn_words:
        max_overlap = 0.0
    else:
        max_overlap = len(note_words & turn_words) / len(note_words)

    grounded = max_overlap >= min_overlap
    reason = "ok" if grounded else "low_overlap"

    return {
        "grounded": grounded,
        "overlap": round(max_overlap, 3),
        "source_turn_found": source_turn_found,
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# Çakışma kontrolü (CONFLICT_RULES dict-driven)
# ---------------------------------------------------------------------------


def check_conflicts(proposed_change: dict, current_profile: Profile) -> dict:
    """
    Args:
        proposed_change: {"field": str, "action": "add"|"remove"|"update", "detail": dict}
        current_profile: mevcut profil

    Returns: {"has_conflict": bool, "action": str, "conflict_type": str | None, "details": dict}
    """
    field = proposed_change.get("field", "")
    action = proposed_change.get("action", "")
    detail = proposed_change.get("detail", {})

    if not field or not action:
        return {"has_conflict": False, "action": "PROCEED", "conflict_type": None, "details": {}}

    rules = CONFLICT_RULES.get(field, {})
    action_rules = rules.get(action, {})

    # medications/allergies add: duplicate name check
    if field in {"medications", "allergies"} and action == "add":
        name = normalize(str(detail.get("name", detail.get("value", ""))).strip())
        if name:
            existing_values = (
                [med.name for med in current_profile.patient.medications]
                if field == "medications"
                else current_profile.patient.allergies
            )
            for existing in existing_values:
                if normalize(existing.strip()) == name:
                    rule = action_rules.get("duplicate_name", action_rules.get("duplicate", {}))
                    return {
                        "has_conflict": True,
                        "action": rule.get("action", "FLAG_PENDING"),
                        "conflict_type": rule.get("conflict_type", "DUPLICATE_MED" if field == "medications" else "DUPLICATE_ALLERGY"),
                        "details": {"existing": existing},
                    }
        return {"has_conflict": False, "action": "PROCEED", "conflict_type": None, "details": {}}

    # medications remove: rewrite as update (status: stopped)
    if field == "medications" and action == "remove":
        rule = action_rules.get("default", {})
        return {
            "has_conflict": True,
            "action": rule.get("action", "REWRITE_AS_UPDATE"),
            "conflict_type": None,
            "details": {"new_status": "stopped"},
        }

    # complications add: duplicate ignore
    if field == "complications" and action == "add":
        value = normalize(str(detail.get("value", "")).strip())
        if value:
            for comp in current_profile.patient.complications:
                if normalize(comp.strip()) == value:
                    rule = action_rules.get("duplicate", {})
                    return {
                        "has_conflict": False,
                        "action": rule.get("action", "IGNORE"),
                        "conflict_type": None,
                        "details": {},
                    }
        return {"has_conflict": False, "action": "PROCEED", "conflict_type": None, "details": {}}

    # goals update: latest wins
    if field == "goals" and action == "update":
        rule = action_rules.get("default", {})
        return {
            "has_conflict": True,
            "action": rule.get("action", "LATEST_WINS"),
            "conflict_type": None,
            "details": {},
        }

    # Diğer alanlar için default: proceed
    return {"has_conflict": False, "action": "PROCEED", "conflict_type": None, "details": {}}


# ---------------------------------------------------------------------------
# Staleness tespiti (changed_field filtresi + keyword + plan yaş) — SAF, yan etkisiz
# ---------------------------------------------------------------------------


def _check_profile_field_active(profile: Profile, note: Note) -> bool:
    """Note içeriğinde geçen bir ilacın profilde hâlâ "active" olup olmadığını döner."""
    content_lower = normalize(note.content)
    for med in profile.patient.medications:
        if normalize(med.name) in content_lower and med.status.value == "active":
            return True
    return False


def _pending_conflict_fields(conv_id: str) -> set[str]:
    """conv_id için PENDING durumdaki çakışmaların profil alanlarını döner.

    conv_id boşsa (test/doğrudan çağrı) boş küme döner — disk erişimi yapılmaz.
    """
    if not conv_id:
        return set()
    store = load_pending_conflicts(conv_id)
    if store is None:
        return set()
    fields: set[str] = set()
    for item in store.items:
        if item.status != PendingConflictStatus.PENDING:
            continue
        field = (item.existing or {}).get("field") or (item.proposed or {}).get("field")
        if field:
            fields.add(field)
    return fields


def detect_stale_notes(
    profile: Profile,
    notes: list[Note],
    changed_field: Optional[str] = None,
    conv_id: str = "",
) -> list[dict]:
    """
    Profil + notlar + değişen alan → stale olabilecek notları tespit eder.

    SAF fonksiyondur: disk yazmaz, log atmaz; yalnızca tespit listesi döner
    ({"type": str, "note_id": str}). Yazma (toplu) ve loglama çağırana aittir
    (maintenance.py). Aynı not için en fazla bir kayıt döner (ilk eşleşen sebep).

    İki bağımsız kontrol:
      1. Keyword-bazlı çelişki: SADECE changed_field ile ilgili kategorilerdeki notlarda.
         Ancak changed_field için PENDING bir çakışma varsa bu blok atlanır —
         profil henüz güncellenmediği için doğru bilgi taşıyan yeni not
         (ör. "bıraktım" notu, ilaç hâlâ active göründüğünden) yanlışlıkla stale
         işaretlenirdi. Pending conflict çözülene kadar beklenir.
      2. Yaş kontrolü: tüm "plan" notlarında (> staleness_plan_max_days gün).
         Profil durumundan bağımsız olduğu için pending kontrolünden etkilenmez.
    """
    stale_items: list[dict] = []
    stale_note_ids: set[str] = set()

    # 1. Keyword-bazlı çelişki — SADECE ilgili kategorilerde, pending yoksa
    if changed_field not in _pending_conflict_fields(conv_id):
        target_categories = FIELD_TO_NOTE_CATEGORY.get(changed_field, [])
        target_notes = [
            n for n in notes
            if n.category in target_categories and n.staleness.value != "stale"
        ]

        for note in target_notes:
            rules = STALENESS_KEYWORD_RULES.get(note.category, [])
            for keywords, kural in rules:
                if any(normalize(k) in normalize(note.content) for k in keywords):
                    if kural == "status_conflict_if_active":
                        if _check_profile_field_active(profile, note):
                            if note.note_id not in stale_note_ids:
                                stale_items.append({"type": "PROFILE_CONFLICT", "note_id": note.note_id})
                                stale_note_ids.add(note.note_id)
                    elif kural == "dose_conflict":
                        if note.note_id not in stale_note_ids:
                            stale_items.append({"type": "DOSE_CONFLICT", "note_id": note.note_id})
                            stale_note_ids.add(note.note_id)
                    elif kural == "goal_replaced":
                        if note.note_id not in stale_note_ids:
                            stale_items.append({"type": "GOAL_REPLACED", "note_id": note.note_id})
                            stale_note_ids.add(note.note_id)
                    elif kural == "resolved_conflict":
                        if note.note_id not in stale_note_ids:
                            stale_items.append({"type": "RESOLVED_CONFLICT", "note_id": note.note_id})
                            stale_note_ids.add(note.note_id)
                    break

    # 2. Yaş kontrolü — tüm "plan" notlarında (pending'den bağımsız)
    for note in notes:
        if note.staleness.value == "stale":
            continue
        if note.category == "plan":
            age_days = (utcnow() - note.created_at).days
            if age_days > MEMORY_CONFIG["staleness_plan_max_days"]:
                if note.note_id not in stale_note_ids:
                    stale_items.append({"type": "OLD_PLAN", "note_id": note.note_id})
                    stale_note_ids.add(note.note_id)

    return stale_items
