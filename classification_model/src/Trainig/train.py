# -*- coding: utf-8 -*-
"""
Grey-band sınıflandırıcı eğitim pipeline'ı (embedding-only).

Özellik: sadece bge-m3 embedding (1024 boyut). Regex/metadata özellikleri
kaldırıldı (ablasyon: katkı ~0).

Akış:
  1) train + test CSV okunur.
  2) Her cümle için bge-m3 embedding üretilir.
  3) GroupKFold (origin_id) ile Logistic Regression vs LightGBM karşılaştırılır.
  4) Model seçimi recall öncelikli: min_recall şartını sağlayanlar specificity'ye
     göre seçilir; hiçbiri sağlamazsa recall ağırlıklı yedek skor.
  5) Eşik kalibrasyonu eğitim setinde DEĞİL, OOF olasılıklar üzerinde yapılır.
  6) OOF eşik ile saf gerçek test seti değerlendirilir.
  7) Model + scaler + threshold joblib ile kaydedilir.

Kullanım:
    python train.py --train ../../data/augmented/train_v9.csv \
        --test ../../data/augmented/test_v2.csv --outdir ../../model_output_v10

Not:
    bge-m3 embedding hesaplama internet + (tercihen) GPU gerektirir.
    --mock-embeddings yalnızca pipeline mantığını test etmek içindir.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

LABEL_POSITIVE = "YELLOW"
LABEL_NEGATIVE = "GREEN"

# LightGBM'i modül seviyesinde bir kez dene.
try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LGBMClassifier = None
    LIGHTGBM_AVAILABLE = False


# ---------------------------------------------------------------------------
# 2) BGE-M3 EMBEDDING
# ---------------------------------------------------------------------------

def get_embeddings(texts: list[str], mock: bool = False) -> np.ndarray:
    """bge-m3 ile embedding hesaplar."""
    if mock:
        print("      [UYARI] --mock-embeddings aktif; gerçek eğitimde KULLANMAYIN.")
        rng = np.random.RandomState(42)
        return rng.rand(len(texts), 1024).astype(np.float32)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence-transformers kurulu değil. "
            "Kurulum: pip install sentence-transformers --break-system-packages"
        ) from e

    print("      bge-m3 modeli yükleniyor (ilk çalıştırmada indirilir)...")
    model = SentenceTransformer("BAAI/bge-m3")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def build_feature_matrix(df: pd.DataFrame, mock_embeddings: bool = False) -> np.ndarray:
    """Cümle embedding'lerini üretir (embedding-only)."""
    if "sentence" not in df.columns:
        raise ValueError("'sentence' sütunu bulunamadı.")

    texts = df["sentence"].fillna("").astype(str).tolist()
    print(f"      Embedding hesaplanıyor ({len(texts):,} cümle)...")
    return get_embeddings(texts, mock=mock_embeddings)


# ---------------------------------------------------------------------------
# 3) DEĞERLENDİRME
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """recall = YELLOW'u yakalama, specificity = GREEN'i doğru koruma."""
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[LABEL_NEGATIVE, LABEL_POSITIVE],
    )
    tn, fp, fn, tp = cm.ravel()
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return {
        "recall": recall,
        "specificity": specificity,
        "precision": precision,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def print_metrics(name: str, m: dict) -> None:
    print(
        f"      {name}: recall(YELLOW yakalama)={m['recall']:.3f}  "
        f"specificity(GREEN koruma)={m['specificity']:.3f}  "
        f"precision={m['precision']:.3f}  "
        f"[TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}]"
    )


def save_test_error_analysis(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    proba_positive: np.ndarray,
    threshold: float,
    outdir_path: Path,
) -> None:
    """Gerçek testteki FP/FN örneklerini ayrıntılı olarak kaydeder ve yazdırır."""
    y_pred = np.where(
        proba_positive >= threshold, LABEL_POSITIVE, LABEL_NEGATIVE
    )

    errors = test_df.copy().reset_index(drop=True)
    errors["true_label"] = y_true
    errors["positive_probability"] = proba_positive
    errors["predicted_label"] = y_pred
    errors["threshold"] = threshold

    errors["error_type"] = "CORRECT"
    errors.loc[
        (errors["true_label"] == LABEL_POSITIVE)
        & (errors["predicted_label"] == LABEL_NEGATIVE),
        "error_type",
    ] = "FALSE_NEGATIVE"
    errors.loc[
        (errors["true_label"] == LABEL_NEGATIVE)
        & (errors["predicted_label"] == LABEL_POSITIVE),
        "error_type",
    ] = "FALSE_POSITIVE"

    error_rows = errors[errors["error_type"] != "CORRECT"].copy()
    error_rows["error_priority"] = np.where(
        error_rows["error_type"] == "FALSE_NEGATIVE", 0, 1
    )
    error_rows = error_rows.sort_values(
        ["error_priority", "positive_probability"],
        ascending=[True, True],
    ).drop(columns=["error_priority"])

    output_path = outdir_path / "test_error_analysis.csv"
    error_rows.to_csv(output_path, index=False, encoding="utf-8-sig")

    fn = error_rows[error_rows["error_type"] == "FALSE_NEGATIVE"]
    fp = error_rows[error_rows["error_type"] == "FALSE_POSITIVE"]

    print("\n=== HATA ANALİZİ: FALSE NEGATIVE (YELLOW → GREEN) ===")
    print(f"      Toplam: {len(fn)}")
    for idx, (_, row) in enumerate(fn.iterrows(), start=1):
        print(
            f"\n      [{idx}] olasılık={row['positive_probability']:.3f} "
            f"| eşik={threshold:.3f}"
        )
        print(f"          Cümle: {row['sentence']}")

    print("\n=== HATA ANALİZİ: FALSE POSITIVE (GREEN → YELLOW) ===")
    print(f"      Toplam: {len(fp)}")
    for idx, (_, row) in enumerate(fp.iterrows(), start=1):
        print(
            f"\n      [{idx}] olasılık={row['positive_probability']:.3f} "
            f"| eşik={threshold:.3f}"
        )
        print(f"          Cümle: {row['sentence']}")

    print(f"\n      Ayrıntılı hata dosyası kaydedildi: {output_path}")


# ---------------------------------------------------------------------------
# 4) GROUPKFOLD
# ---------------------------------------------------------------------------

def _positive_probability(model, X: np.ndarray) -> np.ndarray:
    classes = list(model.classes_)
    if LABEL_POSITIVE not in classes:
        raise ValueError(f"Model classes içinde {LABEL_POSITIVE} bulunamadı.")
    return model.predict_proba(X)[:, classes.index(LABEL_POSITIVE)]


def evaluate_model_cv(
    model_fn,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    threshold: float = 0.5,
) -> dict:
    """GroupKFold ile model performansını 0.5 veya verilen eşikte ölçer."""
    gkf = GroupKFold(n_splits=n_splits)
    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        gkf.split(X, y, groups=groups),
        start=1,
    ):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        model = model_fn()
        model.fit(X_train_scaled, y_train)

        proba = _positive_probability(model, X_val_scaled)
        y_pred = np.where(
            proba >= threshold,
            LABEL_POSITIVE,
            LABEL_NEGATIVE,
        )

        m = compute_metrics(y_val, y_pred)
        m["fold"] = fold_idx
        fold_metrics.append(m)

    avg = {
        key: float(np.mean([fm[key] for fm in fold_metrics]))
        for key in ["recall", "specificity", "precision"]
    }
    std = {
        key: float(np.std([fm[key] for fm in fold_metrics]))
        for key in ["recall", "specificity", "precision"]
    }

    return {"avg": avg, "std": std, "folds": fold_metrics}


V6_CLASS_WEIGHTS = {
    LABEL_NEGATIVE: 1.0,
    LABEL_POSITIVE: 1.5,
}


def make_logistic_regression():
    return LogisticRegression(
        penalty="l2",
        C=1.0,
        max_iter=2000,
        class_weight=V6_CLASS_WEIGHTS,
        random_state=42,
    )


def make_lightgbm():
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("lightgbm kurulu değil.")
    return LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        class_weight=V6_CLASS_WEIGHTS,
        random_state=42,
        verbosity=-1,
    )


# ---------------------------------------------------------------------------
# 5) OOF OLASILIKLARI + EŞİK KALİBRASYONU
# ---------------------------------------------------------------------------

def collect_oof_probabilities(
    model_fn,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Her örnek için, örneği görmemiş fold modeliyle üretilmiş OOF olasılık döndürür.

    Böylece threshold eğitim setine fit edilmiş final modelin kendi tahminlerinden
    öğrenilmez; modelin görmediği örnekler üzerinden kalibre edilir.
    """
    gkf = GroupKFold(n_splits=n_splits)
    oof_proba = np.full(len(y), np.nan, dtype=np.float64)

    for fold_idx, (train_idx, val_idx) in enumerate(
        gkf.split(X, y, groups=groups),
        start=1,
    ):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train = y[train_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        model = model_fn()
        model.fit(X_train_scaled, y_train)

        oof_proba[val_idx] = _positive_probability(model, X_val_scaled)
        print(f"      OOF fold {fold_idx}/{n_splits} tamamlandı.")

    if np.isnan(oof_proba).any():
        raise RuntimeError("OOF olasılıklarının tamamı üretilemedi.")

    return oof_proba, y.copy()


def calibrate_threshold_from_oof(
    oof_proba: np.ndarray,
    y_oof: np.ndarray,
    min_recall: float = 0.92,
) -> float:
    """
    OOF olasılıklarından eşik seçer.

    Öncelik:
      1) recall >= min_recall şartını sağlamak
      2) şartı sağlayan eşikler arasında specificity'yi en yüksek olanı seçmek
      3) eşitlikte daha yüksek recall ve precision'ı tercih etmek
    """
    y_binary = (y_oof == LABEL_POSITIVE).astype(int)
    precisions, recalls, thresholds = precision_recall_curve(y_binary, oof_proba)

    candidates = []
    for threshold, precision, recall in zip(
        thresholds,
        precisions[:-1],
        recalls[:-1],
    ):
        if recall >= min_recall:
            y_pred = np.where(
                oof_proba >= threshold,
                LABEL_POSITIVE,
                LABEL_NEGATIVE,
            )
            metrics = compute_metrics(y_oof, y_pred)
            candidates.append(
                (float(threshold), float(precision), float(recall), metrics["specificity"])
            )

    if not candidates:
        print(
            f"      [UYARI] OOF üzerinde recall>={min_recall:.3f} sağlayan "
            "eşik bulunamadı; en yüksek recall'a sahip eşik seçilecek."
        )

        # Minimum recall hedefi erişilemiyorsa, en yüksek recall;
        # eşitlikte specificity; sonra precision.
        fallback = []
        for threshold, precision, recall in zip(
            thresholds,
            precisions[:-1],
            recalls[:-1],
        ):
            y_pred = np.where(
                oof_proba >= threshold,
                LABEL_POSITIVE,
                LABEL_NEGATIVE,
            )
            metrics = compute_metrics(y_oof, y_pred)
            fallback.append(
                (
                    float(threshold),
                    float(precision),
                    float(recall),
                    metrics["specificity"],
                )
            )

        if not fallback:
            return 0.5

        best = max(fallback, key=lambda c: (c[2], c[3], c[1]))
        print(
            f"      OOF yedek eşiği: {best[0]:.3f} "
            f"(recall={best[2]:.3f}, precision={best[1]:.3f}, "
            f"specificity={best[3]:.3f})"
        )
        return best[0]

    best = max(candidates, key=lambda c: (c[3], c[2], c[1]))
    print(
        f"      OOF seçilen eşik: {best[0]:.3f} "
        f"(recall={best[2]:.3f}, precision={best[1]:.3f}, "
        f"specificity={best[3]:.3f})"
    )
    return best[0]


# ---------------------------------------------------------------------------
# 6) ÖĞRENME EĞRİSİ
# ---------------------------------------------------------------------------

def learning_curve_check(
    model_fn,
    X: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
    fractions: list[float] | None = None,
) -> None:
    if fractions is None:
        fractions = [0.25, 0.5, 0.75, 1.0]

    print("\n=== ÖĞRENME EĞRİSİ (veri miktarı yeterli mi kontrolü) ===")
    rng = np.random.RandomState(42)
    n = len(X)
    indices = rng.permutation(n)

    for frac in fractions:
        n_sub = max(10, int(n * frac))
        n_sub = min(n_sub, n)
        sub_idx = indices[:n_sub]
        X_sub, y_sub = X[sub_idx], y[sub_idx]

        scaler = StandardScaler()
        X_sub_scaled = scaler.fit_transform(X_sub)
        X_test_scaled = scaler.transform(X_test)

        model = model_fn()
        model.fit(X_sub_scaled, y_sub)

        proba = _positive_probability(model, X_test_scaled)
        y_pred = np.where(
            proba >= threshold,
            LABEL_POSITIVE,
            LABEL_NEGATIVE,
        )
        m = compute_metrics(y_test, y_pred)

        print(
            f"      %{int(frac * 100):>3} veri (n={n_sub:>5}): "
            f"recall={m['recall']:.3f}  specificity={m['specificity']:.3f}"
        )

    print(
        "      Not: recall %100 veriyle doygunlaşıyor ama specificity dalgalanıyorsa, "
        "daha fazla augmentation eklemeden önce hata örneklerini incelemek daha anlamlıdır."
    )


# ---------------------------------------------------------------------------
# 7) ANA PIPELINE
# ---------------------------------------------------------------------------

def select_best_model_by_oof(
    model_fns: dict,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    min_recall: float,
) -> tuple[str, object, float, dict]:
    """
    Model seçimini threshold=0.5 ile değil, her modelin kendi OOF
    olasılıkları + OOF üzerinde kalibre edilmiş eşiği ile yapar.

    Bu önemli: Önceki sürümde Logistic Regression 0.5 eşiğinde daha iyi
    specificity verdiği için seçilebiliyordu; fakat sonrasında eşik yeniden
    kalibre ediliyordu. Yani model seçimi ile gerçek karar kuralı aynı değildi.
    Burada ikisi aynı değerlendirme prosedürüne bağlanır.
    """
    results = {}

    for name, model_fn in model_fns.items():
        print(f"\n  --- {name} OOF değerlendirmesi ---")
        oof_proba, y_oof = collect_oof_probabilities(
            model_fn, X, y, groups, n_splits
        )
        threshold = calibrate_threshold_from_oof(
            oof_proba, y_oof, min_recall
        )
        y_pred = np.where(
            oof_proba >= threshold, LABEL_POSITIVE, LABEL_NEGATIVE
        )
        metrics = compute_metrics(y_oof, y_pred)

        results[name] = {
            "model_fn": model_fn,
            "threshold": threshold,
            "metrics": metrics,
            "oof_proba": oof_proba,
        }

        print(
            f"      OOF son karar kuralı: threshold={threshold:.3f} | "
            f"recall={metrics['recall']:.3f} | "
            f"specificity={metrics['specificity']:.3f} | "
            f"precision={metrics['precision']:.3f}"
        )

    eligible = {
        name: result
        for name, result in results.items()
        if result["metrics"]["recall"] >= min_recall
    }

    if eligible:
        best_name = max(
            eligible,
            key=lambda name: (
                eligible[name]["metrics"]["specificity"],
                eligible[name]["metrics"]["recall"],
                eligible[name]["metrics"]["precision"],
            ),
        )
        print(
            f"\n      Recall hedefini ({min_recall:.3f}) sağlayan "
            f"{len(eligible)} model bulundu; OOF specificity önceliğiyle seçildi."
        )
    else:
        best_name = max(
            results,
            key=lambda name: (
                2.0 * results[name]["metrics"]["recall"]
                + results[name]["metrics"]["specificity"],
                results[name]["metrics"]["recall"],
            ),
        )
        print(
            "\n      [UYARI] Hiçbir model minimum recall hedefini karşılamadı. "
            "Yedek seçim: 2*recall + specificity."
        )

    best = results[best_name]
    return (
        best_name,
        best["model_fn"],
        best["threshold"],
        best["metrics"],
    )



def _label_from_json_value(value) -> str:
    """JSON test setindeki True/False etiketini GREEN/YELLOW'a çevirir."""
    if isinstance(value, (bool, np.bool_)):
        return LABEL_POSITIVE if bool(value) else LABEL_NEGATIVE
    if isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value):
        return LABEL_POSITIVE if int(value) == 1 else LABEL_NEGATIVE

    text = str(value).strip().upper()
    if text in {"YELLOW", "TRUE", "1", "YES"}:
        return LABEL_POSITIVE
    if text in {"GREEN", "FALSE", "0", "NO"}:
        return LABEL_NEGATIVE

    raise ValueError(f"Tanımsız JSON etiketi: {value!r}")


def load_test_dataset(test_path: str, test_format: str = "auto") -> pd.DataFrame:
    """
    Test verisini CSV / JSON / JSONL olarak okur.

    CSV:
        sentence, label[, origin_id, ...]

    JSON / JSONL:
        question + is_really_yellow
        veya sentence + label
        veya text + label

    JSON test seti model seçimine ve eşik kalibrasyonuna dahil edilmez;
    yalnızca son test aşamasında kullanılır.
    """
    path = Path(test_path)
    if not path.exists():
        raise FileNotFoundError(f"Test dosyası bulunamadı: {path}")

    if test_format == "auto":
        suffix = path.suffix.lower()
        test_format = {
            ".csv": "csv",
            ".json": "json",
            ".jsonl": "jsonl",
        }.get(suffix)
        if test_format is None:
            raise ValueError(
                f"Test uzantısı tanınmıyor: {path.suffix}. "
                "Gerekirse --test-format csv|json|jsonl kullan."
            )

    if test_format == "csv":
        df = pd.read_csv(path)
        required = {"sentence", "label"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV test dosyasında eksik sütunlar: {sorted(missing)}"
            )
        return df

    if test_format == "json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    elif test_format == "jsonl":
        data = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"JSONL satırı okunamadı ({line_no}. satır): {exc}"
                    ) from exc
    else:
        raise ValueError(
            f"Geçersiz --test-format: {test_format}. "
            "csv, json veya jsonl kullan."
        )

    if isinstance(data, dict):
        for key in ("data", "items", "examples", "records"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            data = [data]

    if not isinstance(data, list) or not data:
        raise ValueError("JSON/JSONL test setinde örnek bulunamadı.")

    rows = []
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(
                f"{index}. JSON örneği object olmalı; "
                f"{type(item).__name__} bulundu."
            )

        if "question" in item:
            sentence = item["question"]
        elif "sentence" in item:
            sentence = item["sentence"]
        elif "text" in item:
            sentence = item["text"]
        else:
            raise ValueError(
                f"{index}. JSON örneğinde metin yok: "
                "'question', 'sentence' veya 'text' bekleniyor."
            )

        if "is_really_yellow" in item:
            label = _label_from_json_value(item["is_really_yellow"])
        elif "label" in item:
            label = _label_from_json_value(item["label"])
        else:
            raise ValueError(
                f"{index}. JSON örneğinde etiket yok: "
                "'is_really_yellow' veya 'label' bekleniyor."
            )

        row = dict(item)
        row["sentence"] = str(sentence)
        row["label"] = label
        row.setdefault("origin_id", f"json_test_{index}")
        rows.append(row)

    df = pd.DataFrame(rows)

    if df["sentence"].isna().any():
        raise ValueError("JSON test setinde boş soru/metin bulundu.")

    if not set(df["label"].unique()).issubset(
        {LABEL_POSITIVE, LABEL_NEGATIVE}
    ):
        raise ValueError("JSON test setinde geçersiz sınıf bulundu.")

    return df


def run_training(
    train_path: str,
    test_path: str,
    outdir: str,
    n_splits: int = 5,
    min_recall: float = 0.92,
    mock_embeddings: bool = False,
    test_format: str = "auto",
) -> None:
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    print("[1/8] Veri okunuyor...")
    train_df = pd.read_csv(train_path)
    test_df = load_test_dataset(test_path, test_format)

    required_train = {"sentence", "label", "origin_id"}
    missing_train = required_train - set(train_df.columns)
    if missing_train:
        raise ValueError(
            f"Eğitim CSV'sinde eksik sütunlar var: {sorted(missing_train)}"
        )

    required_test = {"sentence", "label"}
    missing_test = required_test - set(test_df.columns)
    if missing_test:
        raise ValueError(
            f"Test setinde eksik sütunlar var: {sorted(missing_test)}"
        )

    print(
        f"      Eğitim: {len(train_df):,} satır | "
        f"Test: {len(test_df):,} satır | "
        f"format={test_format}"
    )

    print("\n[2/8] Özellik matrisi oluşturuluyor (eğitim seti)...")
    X_train_full = build_feature_matrix(train_df, mock_embeddings)
    y_train_full = train_df["label"].values
    groups_full = train_df["origin_id"].values

    print("\n[3/8] Özellik matrisi oluşturuluyor (test seti)...")
    X_test = build_feature_matrix(test_df, mock_embeddings)
    y_test = test_df["label"].values

    print(f"\n[4/8] GroupKFold ({n_splits} kat) ile OOF model karşılaştırması...")

    model_fns = {"logistic_regression": make_logistic_regression}
    if LIGHTGBM_AVAILABLE:
        model_fns["lightgbm"] = make_lightgbm
    else:
        print(
            "      [UYARI] lightgbm kurulu değil, atlanıyor. "
            "Kurulum: pip install lightgbm --break-system-packages"
        )

    print("\n[5/8] Her model OOF ile değerlendiriliyor ve kendi eşiği kalibre ediliyor...")
    best_name, best_model_fn, threshold, best_oof_metrics = select_best_model_by_oof(
        model_fns,
        X_train_full,
        y_train_full,
        groups_full,
        n_splits,
        min_recall,
    )

    print(
        f"\n      Seçilen model: {best_name} | "
        f"OOF recall={best_oof_metrics['recall']:.3f} | "
        f"OOF specificity={best_oof_metrics['specificity']:.3f} | "
        f"threshold={threshold:.3f}"
    )

    print(
        f"\n[6/8] '{best_name}' tüm eğitim setiyle eğitiliyor ve "
        "saf gerçek test setinde değerlendiriliyor..."
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test)

    final_model = best_model_fn()
    final_model.fit(X_train_scaled, y_train_full)

    proba_test = _positive_probability(final_model, X_test_scaled)
    y_pred_test = np.where(
        proba_test >= threshold, LABEL_POSITIVE, LABEL_NEGATIVE
    )
    test_metrics = compute_metrics(y_test, y_pred_test)

    print("\n  === SAF GERÇEK TEST SETİ SONUCU (Run C) ===")
    print_metrics("Test", test_metrics)
    print(
        f"      Test dağılımı: YELLOW={int(np.sum(y_test == LABEL_POSITIVE))} | "
        f"GREEN={int(np.sum(y_test == LABEL_NEGATIVE))}"
    )
    print("\n  Karşılaştırma (önceki konuşmalardan):")
    print("    Run A (LLM yok)         : recall=0.94  specificity=0.70")
    print("    Run B (LLM grey'de)     : recall=0.90  specificity=0.95")
    print(
        f"    Run C (bu model)         : recall={test_metrics['recall']:.2f}  "
        f"specificity={test_metrics['specificity']:.2f}"
    )

    print("\n[7/8] Test hataları ayrıntılı olarak inceleniyor...")
    save_test_error_analysis(
        test_df=test_df,
        y_true=y_test,
        proba_positive=proba_test,
        threshold=threshold,
        outdir_path=outdir_path,
    )

    print("\n[8/9] Öğrenme eğrisi kontrol ediliyor...")
    learning_curve_check(
        best_model_fn,
        X_train_full,
        y_train_full,
        X_test,
        y_test,
        threshold,
    )

    print("\n[9/9] Model ve özellik bilgisi kaydediliyor...")
    import joblib

    joblib.dump(
        {
            "model": final_model,
            "scaler": scaler,
            "threshold": threshold,
            "model_name": best_name,
            "label_positive": LABEL_POSITIVE,
            "label_negative": LABEL_NEGATIVE,
            "pipeline_version": "V10_embedding_only",
            "threshold_calibration": "groupkfold_oof",
            "min_recall": min_recall,
        },
        outdir_path / "grey_band_classifier.joblib",
    )

    print(
        f"      Kaydedildi: "
        f"{outdir_path / 'grey_band_classifier.joblib'}"
    )
    print(
        "\nInference sırasında: aynı build_feature_matrix() fonksiyonuyla embedding "
        "üretip, scaler.transform() + model.predict_proba() + threshold ile karar verin."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grey-band sınıflandırıcı eğitim script'i."
    )
    parser.add_argument(
        "--train",
        default="../../data/augmented/train_v9.csv",
        help="train CSV yolu",
    )
    parser.add_argument(
        "--test",
        default="../../data/augmented/test_v2.csv",
        help="test CSV yolu",
    )
    parser.add_argument(
        "--test-format",
        choices=["auto", "csv", "json", "jsonl"],
        default="auto",
        help="Test formatı. auto dosya uzantısından seçer.",
    )
    parser.add_argument(
        "--outdir",
        default="../../model_output_v10",
        help="Çıktı klasörü",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="GroupKFold kat sayısı",
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.92,
        help="Model/threshold seçiminde hedeflenen minimum recall",
    )
    parser.add_argument(
        "--mock-embeddings",
        action="store_true",
        help="SADECE pipeline testi için sahte embedding kullan",
    )
    args = parser.parse_args()

    run_training(
        args.train,
        args.test,
        args.outdir,
        args.n_splits,
        args.min_recall,
        args.mock_embeddings,
        args.test_format,
    )


if __name__ == "__main__":
    main()