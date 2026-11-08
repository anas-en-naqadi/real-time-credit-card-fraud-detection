"""Complete training, model comparison, MLflow tracking, and production registration pipeline.

Tasks covered:
- Baseline Logistic Regression
- Tree-based models: Random Forest, LightGBM
- Imbalance handling: Class Weight vs SMOTE vs Random Under-Sampling
- Hyperparameter tuning integration with Optuna
- Cost-based threshold optimization on validation set
- Final test evaluation touched strictly once
- SHAP explainability artifacts
- MLflow Model Registry publication under alias 'Production'
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import lightgbm as lgb
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.fraud.data.features import FraudFeatureEngineer
from src.fraud.data.loader import load_and_prepare_dataset
from src.fraud.models.anomaly import train_isolation_forest, train_kmeans_profiling
from src.fraud.models.baseline import build_baseline_pipeline
from src.fraud.models.explain import explain_model_shap
from src.fraud.models.metrics import evaluate_predictions, find_optimal_cost_threshold
from src.fraud.models.tune import tune_lightgbm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("fraud.train")

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "fraud-detection-benchmark")
MODEL_REGISTRY_NAME = os.getenv("MODEL_NAME", "fraud-detector")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "Production")
OUTPUT_DIR = Path("data")


def setup_mlflow() -> str:
    """Initialize MLflow tracking URI and experiment."""
    mlflow.set_tracking_uri(TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info(
        "MLflow configured: URI=%s, Experiment=%s (ID=%s)",
        TRACKING_URI,
        EXPERIMENT_NAME,
        experiment_id,
    )
    return experiment_id


def evaluate_and_log_model(
    model_name: str,
    pipeline: Pipeline,
    strategy: str,
    params: dict[str, Any],
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    val_amounts: np.ndarray,
    feature_names: list[str],
) -> dict[str, Any]:
    """Evaluate pipeline on validation set, find optimal threshold, and log run to MLflow."""
    val_probs = pipeline.predict_proba(X_val)[:, 1]

    optimal_thresh, _min_cost, _breakdown = find_optimal_cost_threshold(
        y_true=y_val, y_prob=val_probs, amounts=val_amounts, verification_cost=5.0
    )

    metrics = evaluate_predictions(
        y_true=y_val,
        y_prob=val_probs,
        amounts=val_amounts,
        threshold=optimal_thresh,
        verification_cost=5.0,
    )

    with mlflow.start_run(run_name=f"{model_name}_{strategy}") as run:
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("imbalance_strategy", strategy)
        for k, v in params.items():
            mlflow.log_param(k, v)

        mlflow.log_metrics(
            {
                "val_pr_auc": metrics["pr_auc"],
                "val_roc_auc": metrics["roc_auc"],
                "val_recall": metrics["recall"],
                "val_precision": metrics["precision"],
                "val_f1": metrics["f1"],
                "val_total_cost": metrics["total_cost"],
                "val_optimal_threshold": optimal_thresh,
                "val_tp": metrics["tp"],
                "val_fp": metrics["fp"],
                "val_fn": metrics["fn"],
            }
        )

        # Log scikit-learn model with pyfunc predict_proba for API compatibility
        input_example = X_val.iloc[:2].copy()
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            input_example=input_example,
            pyfunc_predict_fn="predict_proba",
            serialization_format="cloudpickle",
        )
        run_id = run.info.run_id

    result = {
        "model_name": model_name,
        "strategy": strategy,
        "run_id": run_id,
        "pipeline": pipeline,
        "optimal_threshold": optimal_thresh,
        **metrics,
    }
    logger.info(
        "[%s - %s] Val PR-AUC: %.4f | Recall: %.4f | Precision: %.4f | Total Cost: $%.2f | Threshold: %.4f",
        model_name,
        strategy,
        metrics["pr_auc"],
        metrics["recall"],
        metrics["precision"],
        metrics["total_cost"],
        optimal_thresh,
    )
    return result


def run_full_training_pipeline() -> tuple[dict[str, Any], pd.DataFrame]:
    """Execute the complete end-to-end training and evaluation pipeline."""
    setup_mlflow()
    train_df, val_df, test_df = load_and_prepare_dataset()

    X_train = train_df.drop(columns=["Class"])
    y_train = train_df["Class"].to_numpy()
    X_val = val_df.drop(columns=["Class"])
    y_val = val_df["Class"].to_numpy()
    val_amounts = val_df["Amount"].to_numpy()

    results: list[dict[str, Any]] = []

    # 1. Baseline: Logistic Regression (Class Weight)
    logger.info("Training Model 1: Logistic Regression (Class Weight)...")
    baseline_pipe = build_baseline_pipeline(class_weight="balanced", random_state=42)
    baseline_pipe.fit(X_train, y_train)
    res_baseline = evaluate_and_log_model(
        "LogisticRegression",
        baseline_pipe,
        "class_weight",
        {"class_weight": "balanced"},
        X_val,
        y_val,
        val_amounts,
        baseline_pipe.named_steps["features"].get_feature_names_out(),
    )
    results.append(res_baseline)

    # 2. Random Forest (Class Weight)
    logger.info("Training Model 2: Random Forest (Class Weight)...")
    rf_pipe = Pipeline(
        [
            (
                "features",
                FraudFeatureEngineer(include_hour=True, include_log_amount=True),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    class_weight="balanced",
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    rf_pipe.fit(X_train, y_train)
    res_rf = evaluate_and_log_model(
        "RandomForest",
        rf_pipe,
        "class_weight",
        {"n_estimators": 100, "max_depth": 10, "class_weight": "balanced"},
        X_val,
        y_val,
        val_amounts,
        rf_pipe.named_steps["features"].get_feature_names_out(),
    )
    results.append(res_rf)

    # 3. LightGBM (Class Weight / Scale Pos Weight)
    logger.info("Training Model 3: LightGBM (Scale Pos Weight)...")
    scale_pos = float((len(y_train) - sum(y_train)) / max(sum(y_train), 1))
    lgb_pipe = Pipeline(
        [
            (
                "features",
                FraudFeatureEngineer(include_hour=True, include_log_amount=True),
            ),
            (
                "classifier",
                lgb.LGBMClassifier(
                    n_estimators=150,
                    max_depth=6,
                    num_leaves=31,
                    scale_pos_weight=scale_pos,
                    random_state=42,
                    verbosity=-1,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    lgb_pipe.fit(X_train, y_train)
    res_lgb = evaluate_and_log_model(
        "LightGBM",
        lgb_pipe,
        "class_weight",
        {"n_estimators": 150, "max_depth": 6, "scale_pos_weight": scale_pos},
        X_val,
        y_val,
        val_amounts,
        lgb_pipe.named_steps["features"].get_feature_names_out(),
    )
    results.append(res_lgb)

    # 4. LightGBM with SMOTE Oversampling
    logger.info("Training Model 4: LightGBM with SMOTE...")
    feat_eng_smote = FraudFeatureEngineer(include_hour=True, include_log_amount=True)
    X_train_trans = feat_eng_smote.fit_transform(X_train)
    smote = SMOTE(sampling_strategy=0.1, random_state=42)
    X_smote, y_smote = smote.fit_resample(X_train_trans, y_train)
    lgb_smote_clf = lgb.LGBMClassifier(
        n_estimators=150,
        max_depth=6,
        num_leaves=31,
        random_state=42,
        verbosity=-1,
        n_jobs=-1,
    )
    lgb_smote_clf.fit(X_smote, y_smote)
    lgb_smote_pipe = Pipeline(
        [
            ("features", feat_eng_smote),
            ("classifier", lgb_smote_clf),
        ]
    )
    res_smote = evaluate_and_log_model(
        "LightGBM",
        lgb_smote_pipe,
        "SMOTE",
        {"sampling_strategy": 0.1, "n_estimators": 150},
        X_val,
        y_val,
        val_amounts,
        feat_eng_smote.get_feature_names_out(),
    )
    results.append(res_smote)

    # 5. LightGBM with Random Under-Sampling
    logger.info("Training Model 5: LightGBM with Random Under-Sampling...")
    feat_eng_under = FraudFeatureEngineer(include_hour=True, include_log_amount=True)
    X_train_trans_u = feat_eng_under.fit_transform(X_train)
    rus = RandomUnderSampler(sampling_strategy=0.1, random_state=42)
    X_under, y_under = rus.fit_resample(X_train_trans_u, y_train)
    lgb_under_clf = lgb.LGBMClassifier(
        n_estimators=150,
        max_depth=6,
        num_leaves=31,
        random_state=42,
        verbosity=-1,
        n_jobs=-1,
    )
    lgb_under_clf.fit(X_under, y_under)
    lgb_under_pipe = Pipeline(
        [
            ("features", feat_eng_under),
            ("classifier", lgb_under_clf),
        ]
    )
    res_under = evaluate_and_log_model(
        "LightGBM",
        lgb_under_pipe,
        "Undersampling",
        {"sampling_strategy": 0.1, "n_estimators": 150},
        X_val,
        y_val,
        val_amounts,
        feat_eng_under.get_feature_names_out(),
    )
    results.append(res_under)

    # 6. Optuna Hyperparameter Optimization on LightGBM
    logger.info("Training Model 6: Optuna Tuned LightGBM...")
    optuna_res = tune_lightgbm(train_df, val_df, n_trials=15, random_state=42)
    best_params = optuna_res["best_params"]
    tuned_clf = lgb.LGBMClassifier(
        **best_params, random_state=42, verbosity=-1, n_jobs=-1
    )
    tuned_feat = FraudFeatureEngineer(include_hour=True, include_log_amount=True)
    X_train_tuned = tuned_feat.fit_transform(X_train)
    tuned_clf.fit(X_train_tuned, y_train)
    tuned_pipe = Pipeline(
        [
            ("features", tuned_feat),
            ("classifier", tuned_clf),
        ]
    )
    res_tuned = evaluate_and_log_model(
        "LightGBM_Tuned",
        tuned_pipe,
        "Optuna_Tuned",
        best_params,
        X_val,
        y_val,
        val_amounts,
        tuned_feat.get_feature_names_out(),
    )
    results.append(res_tuned)

    # 7. Unsupervised Isolation Forest & K-Means profiling
    logger.info("Executing Unsupervised Anomaly Detection & Clustering...")
    train_isolation_forest(train_df, val_df, random_state=42)
    kmeans_result = train_kmeans_profiling(train_df, n_clusters=5, random_state=42)

    # Build Comparison DataFrame
    comparison_rows = []
    for r in results:
        comparison_rows.append(
            {
                "Model": r["model_name"],
                "Strategy": r["strategy"],
                "PR-AUC": r["pr_auc"],
                "ROC-AUC": r["roc_auc"],
                "Recall": r["recall"],
                "Precision": r["precision"],
                "F1": r["f1"],
                "Total Cost ($)": r["total_cost"],
                "Optimal Threshold": r["optimal_threshold"],
                "Run ID": r["run_id"],
            }
        )
    comparison_df = (
        pd.DataFrame(comparison_rows)
        .sort_values("PR-AUC", ascending=False)
        .reset_index(drop=True)
    )
    comparison_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    logger.info("Saved model comparison to %s", OUTPUT_DIR / "model_comparison.csv")

    # Select Best Model (Highest PR-AUC and lowest cost)
    best_res = max(results, key=lambda x: (x["pr_auc"], -x["total_cost"]))
    best_pipeline: Pipeline = best_res["pipeline"]
    best_threshold: float = best_res["optimal_threshold"]
    best_run_id: str = best_res["run_id"]
    logger.info(
        "Best model selected: %s (%s) with PR-AUC=%.4f and threshold=%.4f",
        best_res["model_name"],
        best_res["strategy"],
        best_res["pr_auc"],
        best_threshold,
    )

    # Final Test Set Evaluation - strictly ONCE
    logger.info("Performing FINAL evaluation on TEST set (touched once)...")
    X_test = test_df.drop(columns=["Class"])
    y_test = test_df["Class"].to_numpy()
    test_amounts = test_df["Amount"].to_numpy()

    test_probs = best_pipeline.predict_proba(X_test)[:, 1]
    final_test_metrics = evaluate_predictions(
        y_true=y_test,
        y_prob=test_probs,
        amounts=test_amounts,
        threshold=best_threshold,
        verification_cost=5.0,
    )
    logger.info(
        "FINAL TEST RESULTS: PR-AUC=%.4f | ROC-AUC=%.4f | Recall=%.4f | Precision=%.4f | F1=%.4f | Cost=$%.2f",
        final_test_metrics["pr_auc"],
        final_test_metrics["roc_auc"],
        final_test_metrics["recall"],
        final_test_metrics["precision"],
        final_test_metrics["f1"],
        final_test_metrics["total_cost"],
    )

    # Save threshold.json (Contract Deliverable)
    threshold_payload = {
        "threshold": round(best_threshold, 4),
        "model_name": MODEL_REGISTRY_NAME,
        "model_type": best_res["model_name"],
        "strategy": best_res["strategy"],
        "validation_pr_auc": best_res["pr_auc"],
        "validation_total_cost": best_res["total_cost"],
        "test_pr_auc": final_test_metrics["pr_auc"],
        "test_total_cost": final_test_metrics["total_cost"],
        "test_recall": final_test_metrics["recall"],
        "test_precision": final_test_metrics["precision"],
    }
    with open("threshold.json", "w", encoding="utf-8") as f:
        json.dump(threshold_payload, f, indent=2)
    with open(OUTPUT_DIR / "threshold.json", "w", encoding="utf-8") as f:
        json.dump(threshold_payload, f, indent=2)
    logger.info("Saved threshold.json with threshold=%.4f", best_threshold)

    # Save K-Means and Isolation Forest summaries
    with open(OUTPUT_DIR / "kmeans_profiles.json", "w", encoding="utf-8") as f:
        json.dump(kmeans_result["cluster_stats"], f, indent=2)

    # SHAP Explainability Analysis
    logger.info("Generating SHAP Explainability Artifacts...")
    feat_transformer = best_pipeline.named_steps["features"]
    tree_classifier = best_pipeline.named_steps["classifier"]

    # Select representative cases from test set (TP, FN, FP) plus a sample of legitimate
    y_pred_test = (test_probs >= best_threshold).astype(int)
    tp_indices = np.where((y_test == 1) & (y_pred_test == 1))[0]
    fn_indices = np.where((y_test == 1) & (y_pred_test == 0))[0]
    fp_indices = np.where((y_test == 0) & (y_pred_test == 1))[0]
    tn_indices = np.where((y_test == 0) & (y_pred_test == 0))[0][:200]

    chosen_indices = np.concatenate([tp_indices, fn_indices, fp_indices, tn_indices])
    chosen_indices = np.unique(chosen_indices)
    test_sample = test_df.iloc[chosen_indices].reset_index(drop=True)

    X_test_sample = test_sample.drop(columns=["Class"])
    y_test_sample = test_sample["Class"].to_numpy()
    transformed_test_sample = feat_transformer.transform(X_test_sample)
    sample_probs = best_pipeline.predict_proba(X_test_sample)[:, 1]

    shap_analysis = explain_model_shap(
        classifier=tree_classifier,
        feature_matrix=transformed_test_sample,
        feature_names=feat_transformer.get_feature_names_out(),
        raw_df=test_sample,
        y_true=y_test_sample,
        y_prob=sample_probs,
        threshold=best_threshold,
    )
    with open(OUTPUT_DIR / "shap_analysis.json", "w", encoding="utf-8") as f:
        json.dump(shap_analysis, f, indent=2)
    logger.info("Saved SHAP analysis to %s", OUTPUT_DIR / "shap_analysis.json")

    # Register Best Model in MLflow Model Registry
    client = MlflowClient()
    model_uri = f"runs:/{best_run_id}/model"

    # Ensure artifacts exist under run directory for Windows local filesystem compatibility
    import shutil

    try:
        import sqlite3

        conn = sqlite3.connect("mlruns.db")
        c = conn.cursor()
        model_artifact_rows = c.execute(
            "SELECT artifact_location FROM logged_models WHERE run_id = ?",
            (best_run_id,),
        ).fetchall()
        if model_artifact_rows:
            source_loc = (
                model_artifact_rows[0][0].replace("file:", "").replace("file:///", "")
            )
            source_path = Path(source_loc)
            dst_path = Path(f"mlruns/1/{best_run_id}/artifacts/model")
            if source_path.exists() and not dst_path.exists():
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(source_path, dst_path, dirs_exist_ok=True)
                logger.info("Synced model artifacts to %s", dst_path)
    except Exception as copy_exc:  # noqa: BLE001
        logger.warning("Artifact sync warning: %s", copy_exc)

    try:
        client.create_registered_model(MODEL_REGISTRY_NAME)
        logger.info("Created registered model '%s'", MODEL_REGISTRY_NAME)
    except Exception:  # noqa: BLE001
        logger.info("Registered model '%s' already exists", MODEL_REGISTRY_NAME)

    model_version = client.create_model_version(
        name=MODEL_REGISTRY_NAME,
        source=model_uri,
        run_id=best_run_id,
        description=f"Best model {best_res['model_name']} ({best_res['strategy']}) with PR-AUC={best_res['pr_auc']:.4f}",
    )
    # Set alias 'Production' and stage 'Production'
    try:
        client.set_registered_model_alias(
            MODEL_REGISTRY_NAME, MODEL_ALIAS, model_version.version
        )
        logger.info(
            "Set alias '%s' for model '%s' version %s",
            MODEL_ALIAS,
            MODEL_REGISTRY_NAME,
            model_version.version,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not set alias via client: %s", exc)

    try:
        client.transition_model_version_stage(
            name=MODEL_REGISTRY_NAME,
            version=model_version.version,
            stage="Production",
        )
        logger.info(
            "Transitioned model version %s to Production stage", model_version.version
        )
    except Exception as stage_exc:  # noqa: BLE001
        logger.warning("Transition to Production stage error: %s", stage_exc)

    final_summary = {
        "best_model": best_res["model_name"],
        "strategy": best_res["strategy"],
        "run_id": best_run_id,
        "model_version": model_version.version,
        "optimal_threshold": best_threshold,
        "val_metrics": {
            k: best_res[k]
            for k in ["pr_auc", "roc_auc", "recall", "precision", "f1", "total_cost"]
        },
        "test_metrics": final_test_metrics,
        "kmeans_stats": kmeans_result["cluster_stats"],
        "top_shap_features": shap_analysis["global_importance"][:5],
    }

    with open(OUTPUT_DIR / "final_model_summary.json", "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2)

    return final_summary, comparison_df


if __name__ == "__main__":
    summary, comparison = run_full_training_pipeline()
    print("\n================ MODEL COMPARISON TABLE ================\n")
    print(comparison.to_string(index=False))
    print("\n================ FINAL MODEL SUMMARY ================\n")
    print(json.dumps(summary, indent=2))
