"""
modelling.py — MLProject entry point
=====================================
File ini adalah versi modelling.py yang disesuaikan untuk MLflow Project.
Menerima hyperparameter via argparse agar bisa dipanggil oleh MLProject.

Penulis : Fina Lestari
Tanggal : 2026-06-09
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import warnings

import dagshub
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.utils import estimator_html_repr

warnings.filterwarnings("ignore")

# ── Konfigurasi ───────────────────────────────────────────────────────────────
DATASET_PATH = "water_potability_preprocessing.csv"
TARGET_COL   = "Potability"
EXPERIMENT   = "Water-Potability-CI"

DAGSHUB_OWNER = "finalestari2712"
DAGSHUB_REPO  = "Membangun_model"

# Ambil token dari GitHub Actions environment
if os.getenv("DAGSHUB_TOKEN"):
    os.environ["DAGSHUB_USER_TOKEN"] = os.getenv("DAGSHUB_TOKEN")

# ══════════════════════════════════════════════════════════════════════════════
# ARGPARSE — parameter dari MLProject
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="Train Water Potability RF Model")
    parser.add_argument("--n_estimators",      type=int,   default=200)
    parser.add_argument("--max_depth",         type=float, default=10)
    parser.add_argument("--min_samples_split", type=int,   default=2)
    parser.add_argument("--min_samples_leaf",  type=int,   default=1)
    parser.add_argument("--max_features",      type=str,   default="sqrt")
    parser.add_argument("--test_size",         type=float, default=0.2)
    parser.add_argument("--random_state",      type=int,   default=42)
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# ARTEFAK
# ══════════════════════════════════════════════════════════════════════════════

def make_estimator_html(model, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(estimator_html_repr(model))


def make_metric_info_json(y_test, y_pred, y_prob, cv_scores, save_path):
    info = {
        "accuracy_score":            accuracy_score(y_test, y_pred),
        "f1_score_weighted":         f1_score(y_test, y_pred, average="weighted"),
        "f1_score_macro":            f1_score(y_test, y_pred, average="macro"),
        "precision_score_weighted":  precision_score(y_test, y_pred, average="weighted"),
        "recall_score_weighted":     recall_score(y_test, y_pred, average="weighted"),
        "roc_auc_score":             roc_auc_score(y_test, y_prob),
        "cross_val_f1_mean":         float(cv_scores.mean()),
        "cross_val_f1_std":          float(cv_scores.std()),
        "classification_report":     classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix":          confusion_matrix(y_test, y_pred).tolist(),
    }
    with open(save_path, "w") as f:
        json.dump(info, f, indent=2)


def make_training_confusion_matrix(y_test, y_pred, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_test, y_pred),
        display_labels=["Tidak Layak (0)", "Layak (1)"],
    ).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — Test Set", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def make_roc_curve(model, X_test, y_test, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, color="darkorange")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title("ROC Curve", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def make_feature_importance(model, feature_names, save_path):
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(len(importances)), importances[idx],
           color="steelblue", edgecolor="white")
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in idx], rotation=40, ha="right")
    ax.set_title("Feature Importances", fontsize=13, fontweight="bold")
    ax.set_ylabel("Importance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # max_depth
    if args.max_depth == 0:
        max_depth = None
    else:
        max_depth = args.max_depth

    print("=" * 55)
    print("MLflow Project — Water Potability CI")
    print("=" * 55)
    print(f"  n_estimators      : {args.n_estimators}")
    print(f"  max_depth         : {max_depth}")
    print(f"  min_samples_split : {args.min_samples_split}")
    print(f"  min_samples_leaf  : {args.min_samples_leaf}")
    print(f"  max_features      : {args.max_features}")

    # ── Setup DagsHub ─────────────────────────────────────────────────────────
    print("DAGSHUB_TOKEN exists:", bool(os.getenv("DAGSHUB_TOKEN")))
    
    dagshub.init(repo_owner=DAGSHUB_OWNER, repo_name=DAGSHUB_REPO, mlflow=True)
    mlflow.set_experiment(EXPERIMENT)

    # ── Load data ─────────────────────────────────────────────────────────────
    df = pd.read_csv(DATASET_PATH)
    X  = df.drop(columns=[TARGET_COL])
    y  = df[TARGET_COL]
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = args.test_size,
        random_state = args.random_state,
        stratify     = y,
    )
    print(f"\nData: train={len(X_train)} | test={len(X_test)}")

    # ── Training ──────────────────────────────────────────────────────────────
    model = RandomForestClassifier(
        n_estimators      = args.n_estimators,
        max_depth         = max_depth,
        min_samples_split = args.min_samples_split,
        min_samples_leaf  = args.min_samples_leaf,
        max_features      = args.max_features,
        random_state      = args.random_state,
        n_jobs            = -1,
    )
    model.fit(X_train, y_train)

    # ── Evaluasi ──────────────────────────────────────────────────────────────
    y_pred    = model.predict(X_test)
    y_prob    = model.predict_proba(X_test)[:, 1]
    cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_weighted")

    accuracy  = accuracy_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred, average="weighted")
    f1_macro  = f1_score(y_test, y_pred, average="macro")
    precision = precision_score(y_test, y_pred, average="weighted")
    recall    = recall_score(y_test, y_pred, average="weighted")
    roc_auc   = roc_auc_score(y_test, y_prob)

    # ── MLflow Manual Logging ─────────────────────────────────────────────────
    with mlflow.start_run(run_name="RF_CI_Pipeline"):

        # Parameter
        mlflow.log_param("n_estimators",      args.n_estimators)
        mlflow.log_param("max_depth",         max_depth)
        mlflow.log_param("min_samples_split", args.min_samples_split)
        mlflow.log_param("min_samples_leaf",  args.min_samples_leaf)
        mlflow.log_param("max_features",      args.max_features)
        mlflow.log_param("test_size",         args.test_size)
        mlflow.log_param("random_state",      args.random_state)

        # Metrik
        mlflow.log_metric("accuracy_score",           accuracy)
        mlflow.log_metric("f1_score_weighted",        f1)
        mlflow.log_metric("f1_score_macro",           f1_macro)
        mlflow.log_metric("precision_score_weighted", precision)
        mlflow.log_metric("recall_score_weighted",    recall)
        mlflow.log_metric("roc_auc_score",            roc_auc)
        mlflow.log_metric("cross_val_f1_mean",        cv_scores.mean())
        mlflow.log_metric("cross_val_f1_std",         cv_scores.std())

        # Model
        mlflow.sklearn.log_model(
            sk_model              = model,
            artifact_path         = "model",
            registered_model_name = "WaterPotability_CI",
        )

        # Artefak
        with tempfile.TemporaryDirectory() as tmp:
            est_path = os.path.join(tmp, "estimator.html")
            make_estimator_html(model, est_path)
            mlflow.log_artifact(est_path)

            mi_path = os.path.join(tmp, "metric_info.json")
            make_metric_info_json(y_test, y_pred, y_prob, cv_scores, mi_path)
            mlflow.log_artifact(mi_path)

            cm_path = os.path.join(tmp, "training_confusion_matrix.png")
            make_training_confusion_matrix(y_test, y_pred, cm_path)
            mlflow.log_artifact(cm_path)

            roc_path = os.path.join(tmp, "roc_curve.png")
            make_roc_curve(model, X_test, y_test, roc_path)
            mlflow.log_artifact(roc_path)

            fi_path = os.path.join(tmp, "feature_importance.png")
            make_feature_importance(model, feature_names, fi_path)
            mlflow.log_artifact(fi_path)

        # Tags
        mlflow.set_tag("model_type", "RandomForest")
        mlflow.set_tag("trigger",    "CI-GitHub-Actions")
        mlflow.set_tag("developer",  "NamaSiswa")

        run_id = mlflow.active_run().info.run_id

    print("\n" + "=" * 55)
    print("HASIL")
    print("=" * 55)
    print(f"  accuracy_score    : {accuracy:.4f}")
    print(f"  f1_score_weighted : {f1:.4f}")
    print(f"  roc_auc_score     : {roc_auc:.4f}")
    print(f"  cv_f1_mean        : {cv_scores.mean():.4f}")
    print(f"\n  Run ID : {run_id}")
    print(f"  DagsHub: https://dagshub.com/{DAGSHUB_OWNER}/{DAGSHUB_REPO}/experiments")


if __name__ == "__main__":
    main()
