from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal, Optional
from pydantic import BaseModel, Field

from src.api.memory.timeutil import utcnow


class MedicationStatus(str, Enum):
    ACTIVE = "active"
    STOPPED = "stopped"
    CHANGED = "changed"


class Medication(BaseModel):
    name: str
    dose: str
    frequency: str
    started: str
    status: MedicationStatus = MedicationStatus.ACTIVE


class PatientProfile(BaseModel):
    diabetes_type: str
    diagnosis_year: int
    medications: list[Medication] = Field(default_factory=list)
    complications: list[str] = Field(default_factory=list)
    monitoring: dict = Field(default_factory=dict)
    allergies: list[str] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)
    lifestyle: dict = Field(default_factory=dict)


class ProfileMetadata(BaseModel):
    source_turn_ids: list[str] = Field(default_factory=list)
    confidence_scores: dict[str, float] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=utcnow)


class Profile(BaseModel):
    version: int = 1
    updated_at: datetime = Field(default_factory=utcnow)
    patient: PatientProfile
    metadata: ProfileMetadata = Field(default_factory=ProfileMetadata)


class CandidateNote(BaseModel):
    content: str
    category: Literal["symptom", "observation", "plan", "measurement", "advice"]
    source_turn_id: Optional[str] = None


class NoteStaleness(str, Enum):
    FRESH = "fresh"
    STALE = "stale"


class Note(BaseModel):
    note_id: str
    content: str
    category: str
    source_turns: list[str] = Field(default_factory=list)
    confidence: float = 1.0
    staleness: NoteStaleness = NoteStaleness.FRESH
    created_at: datetime = Field(default_factory=utcnow)


class NotesStore(BaseModel):
    version: int = 1
    updated_at: datetime = Field(default_factory=utcnow)
    items: list[Note] = Field(default_factory=list)


class Summary(BaseModel):
    text: str
    updated_at: datetime = Field(default_factory=utcnow)


class TurnRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class Turn(BaseModel):
    turn_id: str
    role: TurnRole
    content: str
    timestamp: datetime = Field(default_factory=utcnow)
    triage: Optional[str] = None


class PendingConflictStatus(str, Enum):
    PENDING = "pending"
    RESOLVED = "resolved"
    EXPIRED_REJECTED = "expired_rejected"


class PendingConflict(BaseModel):
    conflict_id: str
    type: str
    existing: dict
    proposed: dict
    status: PendingConflictStatus = PendingConflictStatus.PENDING
    created_at: datetime = Field(default_factory=utcnow)


class PendingConflictsStore(BaseModel):
    items: list[PendingConflict] = Field(default_factory=list)


class MemoryIndex(BaseModel):
    turn_count: int = 0
    last_summary_at_turn: int = 0
    last_note_extraction_at_turn: int = 0
    last_profile_update_at_turn: int = 0
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class ProfileUpdate(BaseModel):
    """Maintenance agent tarafından üretilen profil güncelleme teklifi.

    Plan (MAINTENANCE_PROMPT) çıktı şemasına birebir uyumludur:
      - has_update: güncelleme olup olmadığı (false ise diğer alanlar anlamsızdır).
      - proposed_change: {"action": "add|remove|update", "detail": {...}}
        Örnekler:
          - add medication    : {"action": "add", "detail": {"name": "Metformin", "dose": "500mg", ...}}
          - remove medication : {"action": "remove", "detail": {"name": "Metformin"}}
          - add complication  : {"action": "add", "detail": {"value": "retinopati"}}
          - update goals      : {"action": "update", "detail": {"value": "Haftada 3 yürüyüş"}}
    """

    has_update: bool
    field: Literal["medications", "complications", "goals", "monitoring", "allergies"]
    proposed_change: dict
    evidence_span: str
    confidence: float


class MaintenanceResult(BaseModel):
    profile_update: Optional[ProfileUpdate] = None
    candidate_notes: list[CandidateNote] = Field(default_factory=list)
    summary: Optional[str] = None  # LLM düz metin üretir; updated_at kod tarafından eklenir


class CriticalVerifyResult(BaseModel):
    onayla: bool
    gerekce: str
    eminlik: float
