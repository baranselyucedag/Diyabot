"""Deterministik kurallar — LLM'siz, test edilebilir, hızlı.

Plan dokümanına (cahing_mimari_tasarim.md §3) birebir uyumludur:
- CONFLICT_RULES, FIELD_TO_NOTE_CATEGORY, STALENESS_KEYWORD_RULES, TRIAGE_FIELD_RISK
  sabitleri plan'daki gibidir.
- check_staleness_deterministic, tespit ettiği notları doğrudan update_note_staleness
  ile işaretler ve log_event ile kaydeder.
"""

from __future__ import annotations

import re
from typing import Optional

from src.api.memory.config import MEMORY_CONFIG
from src.api.memory.logger import log_event
from src.api.memory.memory_store import update_note_staleness
from src.api.memory.models import CandidateNote, Note, Profile, Turn
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
    "goals": {
        "update": {"default": {"action": "LATEST_WINS"}},
    },
}

FIELD_TO_NOTE_CATEGORY = {
    "medications": ["observation"],
    "goals": ["plan", "advice"],
    "complications": ["symptom"],
    "monitoring": ["measurement"],
}

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


# ---------------------------------------------------------------------------
# Grounding doğrulama
# ---------------------------------------------------------------------------


def word_set(text: str) -> set[str]:
    """Normalize edilmiş, stopword'lerden arındırılmış kelime kümesi."""
    norm_text = normalize(text)
    return set(re.findall(r"[\w]+", norm_text)) - TURKISH_STOPWORDS


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

    # medications add: duplicate name check
    if field == "medications" and action == "add":
        name = detail.get("name", "").strip().lower()
        if name:
            for med in current_profile.patient.medications:
                if med.name.strip().lower() == name:
                    rule = action_rules.get("duplicate_name", {})
                    return {
                        "has_conflict": True,
                        "action": rule.get("action", "FLAG_PENDING"),
                        "conflict_type": rule.get("conflict_type", "DUPLICATE_MED"),
                        "details": {"existing_med": med.name},
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
        value = detail.get("value", "").strip().lower()
        if value:
            for comp in current_profile.patient.complications:
                if comp.strip().lower() == value:
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
# Staleness kontrolü (changed_field filtresi + keyword + plan yaş)
# ---------------------------------------------------------------------------


def _check_profile_field_active(profile: Profile, note: Note) -> bool:
    """Note içeriğinde geçen bir ilacın profilde hâlâ "active" olup olmadığını döner."""
    content_lower = note.content.lower()
    for med in profile.patient.medications:
        if med.name.lower() in content_lower and med.status.value == "active":
            return True
    return False


def check_staleness_deterministic(
    profile: Profile,
    notes: list[Note],
    changed_field: Optional[str] = None,
    conv_id: str = "",
) -> list[dict]:
    """
    Profil + notlar + değişen alan → stale olabilecek notları tespit eder.

    Tespit edilen her not doğrudan `update_note_staleness` ile "stale" işaretlenir
    ve `log_event` ile kaydedilir. Return, tespit edilen kalemlerin listesidir
    ({"type": str, "note_id": str}).

    İki bağımsız kontrol:
      1. Keyword-bazlı çelişki: SADECE changed_field ile ilgili kategorilerdeki notlarda.
      2. Yaş kontrolü: tüm "plan" notlarında (> staleness_plan_max_days gün).
    """
    stale_items: list[dict] = []

    # 1. Keyword-bazlı çelişki — SADECE ilgili kategorilerde
    target_categories = FIELD_TO_NOTE_CATEGORY.get(changed_field, [])
    target_notes = [
        n for n in notes
        if n.category in target_categories and n.staleness.value != "stale"
    ]

    for note in target_notes:
        rules = STALENESS_KEYWORD_RULES.get(note.category, [])
        for keywords, kural in rules:
            if any(k in note.content.lower() for k in keywords):
                if kural == "status_conflict_if_active":
                    if _check_profile_field_active(profile, note):
                        stale_items.append({"type": "PROFILE_CONFLICT", "note_id": note.note_id})
                elif kural == "dose_conflict":
                    stale_items.append({"type": "DOSE_CONFLICT", "note_id": note.note_id})
                elif kural == "goal_replaced":
                    stale_items.append({"type": "GOAL_REPLACED", "note_id": note.note_id})
                elif kural == "resolved_conflict":
                    stale_items.append({"type": "RESOLVED_CONFLICT", "note_id": note.note_id})
                break

    # 2. Yaş kontrolü — tüm "plan" notlarında
    for note in notes:
        if note.staleness.value == "stale":
            continue
        if note.category == "plan":
            age_days = (utcnow() - note.created_at).days
            if age_days > MEMORY_CONFIG["staleness_plan_max_days"]:
                stale_items.append({"type": "OLD_PLAN", "note_id": note.note_id})

    # 3. Tespit edilenleri işaretle + logla
    for item in stale_items:
        update_note_staleness(conv_id, item["note_id"], "stale")
        log_event("staleness", "staleness_conflict", conv_id, **item)

    return stale_items
