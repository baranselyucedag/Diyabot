# -*- coding: utf-8 -*-
"""
evaluate_threshold.py — Kayıtlı modelin test setindeki recall/specificity eğrisini
farklı eşiklerde basar. İstenirse joblib'deki eşiği günceller.

Kullanım:
    python evaluate_threshold.py --model ..\..\data\model_output_v9\grey_band_classifier.joblib ^
        --test ..\..\data\augmented\test_v2.csv --set-threshold 0.42
"""
import argparse
import importlib.util
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# train.py'yi main() çalıştırmadan modül olarak yükle
train_py = Path(__file__).parent / "train.py"
spec = importlib.util.spec_from_file_location("train_mod", train_py)
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--set-threshold", type=float, default=None,
                    help="Bu değere eşiği güncelle ve joblib'i kaydet")
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    model = bundle["model"]
    scaler = bundle["scaler"]
    old_threshold = bundle["threshold"]
    print(f"kayıtlı eşik: {old_threshold:.3f} | model: {bundle.get('model_name')}")

    test_df = pd.read_csv(args.test, encoding="utf-8-sig")
    print(f"test: {len(test_df)} satır (embedding hesaplanıyor...)")

    X = T.build_feature_matrix(test_df)
    X_scaled = scaler.transform(X)
    y_true = test_df["label"].values
    proba = T._positive_probability(model, X_scaled)

    print(f"\n{'eşik':>6} {'recall':>8} {'spec':>8} {'prec':>8} {'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5}")
    print("-" * 58)
    for th in [0.30, 0.338, 0.36, 0.38, 0.40, 0.42, 0.45, 0.48, 0.50]:
        y_pred = np.where(proba >= th, "YELLOW", "GREEN")
        m = T.compute_metrics(y_true, y_pred)
        mark = " <-- kayıtlı" if abs(th - old_threshold) < 0.001 else ""
        print(f"{th:>6.3f} {m['recall']:>8.3f} {m['specificity']:>8.3f} "
              f"{m['precision']:>8.3f} {m['tp']:>5} {m['fp']:>5} {m['tn']:>5} {m['fn']:>5}{mark}")

    if args.set_threshold is not None:
        bundle["threshold"] = args.set_threshold
        joblib.dump(bundle, args.model)
        print(f"\n[eşik güncellendi] {args.model} -> threshold={args.set_threshold:.3f}")


if __name__ == "__main__":
    main()
