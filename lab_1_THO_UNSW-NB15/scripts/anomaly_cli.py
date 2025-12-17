#!/usr/bin/env python3
"""
UNSW-NB15 Anomaly Detection CLI
Вариант: Isolation Forest / drop / standard / precision/recall
Поддержка:
- EDA
- Обучение нескольких моделей (Isolation Forest, LOF, One-Class SVM)
- Автоподбор гиперпараметров (GridSearchCV) для Isolation Forest
- Детекция одной моделью
- Энсембль (consensus) по нескольким моделям
- Расширенная оценка качества (precision, recall, F1, ROC-AUC)
"""

from sklearn.svm import OneClassSVM
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import typer
import yaml

import matplotlib
matplotlib.use("Agg")


app = typer.Typer(
    help="CLI для обнаружения аномалий в сетевом трафике UNSW-NB15")
logger: Optional[logging.Logger] = None


# Вспомогательные функции
def load_config(config_path: str) -> Dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(cfg: Dict) -> logging.Logger:
    global logger
    log_cfg = cfg.get("logging", {})
    logs_dir = Path(cfg["paths"].get("logs_dir", "logs"))
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("anomaly_detection")
    logger.setLevel(getattr(logging, log_cfg.get("level", "INFO")))

    if logger.handlers:
        logger.handlers.clear()

    fmt = log_cfg.get(
        "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    formatter = logging.Formatter(fmt)

    if log_cfg.get("file_enabled", True):
        log_file = logs_dir / \
            f"anomaly_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    if log_cfg.get("console_enabled", True):
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


def create_directories(cfg: Dict) -> None:
    for _, p in cfg["paths"].items():
        Path(p).mkdir(parents=True, exist_ok=True)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Простые доменные фичи для сетевого трафика:
    - sbytes_per_pkt, dbytes_per_pkt
    - total_bytes
    - pkt_rate
    """
    df = df.copy()

    if {"sbytes", "spkts"}.issubset(df.columns):
        df["sbytes_per_pkt"] = df["sbytes"] / df["spkts"].replace(0, np.nan)
        df["sbytes_per_pkt"] = df["sbytes_per_pkt"].replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0)

    if {"dbytes", "dpkts"}.issubset(df.columns):
        df["dbytes_per_pkt"] = df["dbytes"] / df["dpkts"].replace(0, np.nan)
        df["dbytes_per_pkt"] = df["dbytes_per_pkt"].replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0)

    if {"sbytes", "dbytes"}.issubset(df.columns):
        df["total_bytes"] = df["sbytes"] + df["dbytes"]

    if {"dur", "spkts", "dpkts"}.issubset(df.columns):
        df["pkt_rate"] = (df["spkts"] + df["dpkts"]) / \
            df["dur"].replace(0, np.nan)
        df["pkt_rate"] = df["pkt_rate"].replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0)

    return df


def remove_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """ Remove constant columns from DataFrame """
    return df.loc[:, df.nunique() > 1]


def load_data(file_path: str, sample_size: Optional[int] = None) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Data file is empty: {path}")

    if sample_size is not None and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)

    df = add_derived_features(df)

    return df


# 1. EDA
@app.command()
def eda(
    config: str = typer.Option("config/config.yaml", help="Путь к конфигу"),
    sample_size: Optional[int] = typer.Option(
        None, help="Размер выборки для анализа"),
):
    """
    Разведочный анализ данных: базовая статистика, корреляции, распределения, выбросы.
    """
    # Загрузка конфигурации
    cfg = load_config(config)
    log = setup_logging(cfg)
    create_directories(cfg)

    eda_dir = Path(cfg["paths"]["eda_dir"])

    # Загрузка данных
    log.info("=== EDA: загрузка данных ===")
    df = load_data(cfg["data"]["train_csv"],
                   sample_size or cfg["eda"].get("sample_size"))
    log.info(f"Размер train-датасета: {df.shape}")

    # Базовая статистика
    desc = df.describe(include="all")
    desc.to_csv(eda_dir / "basic_statistics.csv")

    # Пропуски
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    missing.to_csv(eda_dir / "missing_values.csv")

    # Корреляционная матрица
    log.info("Построение корреляционной матрицы...")

    df_corr = df.copy()
    df_corr = df_corr.loc[:, df_corr.nunique(dropna=False) > 1]
    cat_cols = df_corr.select_dtypes(include=["object"]).columns.tolist()
    if cat_cols:
        df_corr = pd.get_dummies(df_corr, columns=cat_cols, drop_first=True)

    num_cols = df_corr.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        log.warning("Нет числовых признаков для корреляционной матрицы.")
    else:
        dup_mask = df_corr[num_cols].T.duplicated()
        dup_cols = list(pd.Index(num_cols)[dup_mask])
        if dup_cols:
            log.info(f"Duplicate (identical) columns dropped for corr plot: {dup_cols[:20]}{'...' if len(dup_cols)>20 else ''}")
            df_corr = df_corr.drop(columns=dup_cols)

        thr = float(cfg["eda"].get("correlation_threshold", 0.9))
        num_cols2 = df_corr.select_dtypes(include=[np.number]).columns
        corr_abs = df_corr[num_cols2].corr().abs()
        upper = corr_abs.where(np.triu(np.ones(corr_abs.shape), k=1).astype(bool))
        to_drop = [c for c in upper.columns if (upper[c] >= thr).any()]

        if to_drop:
            log.info(f"Highly-correlated columns dropped for corr plot (thr={thr}): {to_drop[:20]}{'...' if len(to_drop)>20 else ''}")
            pd.Series(to_drop).to_csv(eda_dir / "corr_plot_dropped_columns.csv", index=False)
            df_corr = df_corr.drop(columns=to_drop)

        num_cols_final = df_corr.select_dtypes(include=[np.number]).columns
        corr = df_corr[num_cols_final].corr()

        plt.figure(figsize=tuple(cfg["eda"].get("figure_size", [12, 8])))
        sns.heatmap(corr, cmap="coolwarm", center=0)
        plt.title("Correlation matrix (numeric features) — reduced for plotting")
        plt.tight_layout()
        plt.savefig(eda_dir / "correlation_matrix.png", dpi=cfg["visualization"].get("dpi", 300))
        plt.close()


    # Распределения ключевых признаков
    log.info("Анализ распределений признаков...")
    key_features = (cfg["features"]["network_basic"] +
                    cfg["features"]["statistical"] +
                    cfg["features"]["contextual"] +
                    cfg["features"]["additional"])

    key_features = [f for f in key_features if f in df.columns][:6]

    if len(key_features) > 0:
        n = len(key_features)
        rows = int(np.ceil(n / 3))
        fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))
        axes = np.array(axes).reshape(-1)
        for i, col in enumerate(key_features):
            df[col].hist(bins=50, ax=axes[i], alpha=0.7)
            axes[i].set_title(f"Distribution of {col}")
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")
        plt.tight_layout()
        plt.savefig(eda_dir / "feature_distributions.png",
                    dpi=cfg["visualization"].get("dpi", 300))
        plt.close()
    else:
        log.warning("No valid features found for distribution plotting.")

    # Анализ выбросов (boxplot)
    log.info("Анализ выбросов...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, feature in enumerate(key_features[:6]):
        if feature in df.columns:
            df.boxplot(column=feature, ax=axes[i])
            axes[i].set_title(f'Выбросы в {feature}')

    plt.tight_layout()
    plt.savefig(eda_dir / 'outliers_analysis.png', dpi=300)
    plt.close()

    log.info(f"EDA завершён, артефакты в {eda_dir}")


# 2. TRAIN
@app.command()
def train(
    config: str = typer.Option("config/config.yaml", help="Путь к конфигу"),
    model_type: str = typer.Option(
        "isolation-forest", help="Тип модели: isolation-forest, lof, one-class-svm"
    ),
    tune_params: bool = typer.Option(
        False, help="Выполнить автоподбор гиперпараметров (GridSearchCV) для Isolation Forest"
    ),
):
    """
    Обучение модели (по умолчанию Isolation Forest).
    При tune_params=True и model_type='isolation-forest' выполняется GridSearchCV по precision/recall.
    """
    cfg = load_config(config)
    log = setup_logging(cfg)
    create_directories(cfg)

    log.info(f"=== TRAIN: модель {model_type}, tune_params={tune_params} ===")

    if model_type in ["lof", "one-class-svm"]:
        df = load_data(cfg["data"]["train_csv"], sample_size=30000)
    else:
        df = load_data(cfg["data"]["train_csv"])

    # Подготовка признаков
    all_features: List[str] = []
    all_features += cfg["features"].get("network_basic", [])
    all_features += cfg["features"].get("transport", [])
    all_features += cfg["features"].get("statistical", [])
    all_features += cfg["features"].get("contextual", [])
    all_features += cfg["features"].get("additional", [])
    all_features += cfg["features"].get("categorical", [])

    # Убираем дубликаты и несуществующие
    all_features = list(dict.fromkeys(all_features))
    all_features = [f for f in all_features if f in df.columns]

    exclude = cfg["features"].get("exclude", [])
    feature_cols = [f for f in all_features if f not in exclude]

    X = df[feature_cols].copy()
    log.info(
        f"Всего признаков: {len(all_features)}, после exclude: {len(feature_cols)}")

    # Числовые / категориальные
    cat_features = [c for c in cfg["features"].get(
        "categorical", []) if c in X.columns]
    num_features = [c for c in X.columns if c not in cat_features]

    log.info(
        f"Числовые: {len(num_features)}, категориальные: {len(cat_features)}")

    # Заполнение пропусков медианой по числовым
    median_map: Dict[str, float] = {}
    for col in num_features:
        med = float(X[col].median())
        median_map[col] = med
        X[col] = X[col].fillna(med)

    # Препроцессор: StandardScaler + OneHotEncoder
    transformers = []
    if num_features:
        transformers.append(("num", StandardScaler(), num_features))
    if cat_features:
        transformers.append(("cat", OneHotEncoder(
            handle_unknown="ignore"), cat_features))

    preprocessor = ColumnTransformer(transformers=transformers)

    # Модель
    model_cfg = cfg["models"].get(model_type.replace("-", "_"), {})
    if model_type == "isolation-forest":
        model = IsolationForest(**model_cfg)
    elif model_type == "lof":
        model = LocalOutlierFactor(**model_cfg)
    elif model_type == "one-class-svm":
        model = OneClassSVM(**model_cfg)
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", model),
    ])

    # Метки (для оценки при GridSearchCV, обучение всё равно unsupervised)
    y = df["label"].astype(int) if "label" in df.columns else None

    # === Автоподбор гиперпараметров для Isolation Forest ===
    if tune_params and model_type == "isolation-forest":
        if y is None:
            log.warning(
                "Для подбора гиперпараметров требуется метка 'label'")

            pipeline.fit(X)
        else:
            param_grid = {
                "model__n_estimators": [100, 200, 300],
                "model__contamination": [0.05, 0.1, 0.15],
            }
            log.info(f"Запуск GridSearchCV с параметрами: {param_grid}")

            def precision_recall_scorer(estimator, X_val, y_val):
                raw = estimator.predict(X_val)
                y_pred = (raw == -1).astype(int)
                precision = precision_score(y_val, y_pred)
                recall = recall_score(y_val, y_pred)
                return 2 * (precision * recall) / (precision + recall)

            grid = GridSearchCV(
                pipeline,
                param_grid=param_grid,
                scoring=precision_recall_scorer,
                cv=3,
                n_jobs=-1,
            )
            grid.fit(X, y)
            log.info(
                f"Лучшие параметры: {grid.best_params_}, лучший precision/recall: {grid.best_score_:.4f}")
            pipeline = grid.best_estimator_
    else:
        pipeline.fit(X)

    models_dir = Path(cfg["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / f"{model_type}_model.joblib"
    joblib.dump(pipeline, model_path)

    # Метаданные
    model_params = pipeline.named_steps["model"].get_params()
    metadata = {
        "model_type": model_type,
        "features": feature_cols,
        "model_params": model_params,
        "median_map": median_map,
        "timestamp": datetime.now().isoformat(),
    }
    metadata_path = models_dir / f"{model_type}_metadata.yaml"
    with open(metadata_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(metadata, f)

    log.info(f"Модель сохранена в {model_path}")
    log.info(f"Метаданные сохранены в {metadata_path}")


# 3. DETECT (одна модель)
@app.command()
def detect(
    config: str = typer.Option("config/config.yaml", help="Путь к конфигу"),
    model_path: str = typer.Option(
        None,
        help="Путь к обученной модели (по умолчанию artifacts/models/isolation-forest_model.joblib)",
    ),
    input_file: str = typer.Option(
        None,
        help="Файл с данными для детекции (по умолчанию test_csv из конфига)",
    ),
    output_file: str = typer.Option(
        "artifacts/predictions.csv",
        help="CSV для сохранения результатов",
    ),
):
    """
    Детекция аномалий одной моделью (по умолчанию Isolation Forest).
    """
    cfg = load_config(config)
    log = setup_logging(cfg)

    if model_path is None:
        model_path = str(
            Path(cfg["paths"]["models_dir"]) / "isolation-forest_model.joblib")
    if input_file is None:
        input_file = cfg["data"]["test_csv"]

    log.info(f"=== DETECT: модель {model_path} ===")
    pipeline: Pipeline = joblib.load(model_path)

    # Метаданные
    meta_path = Path(model_path).parent / \
        f"{Path(model_path).stem.replace('_model', '_metadata.yaml')}"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = yaml.safe_load(f)
        features = metadata.get("features")
        median_map = metadata.get("median_map", {})
    else:
        log.warning(
            "Метаданные не найдены, используются все числовые признаки + median_map={}")
        metadata = {}
        median_map = {}
        features = None

    df = load_data(input_file)
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()

    missing = [f for f in features if f not in df.columns]
    if missing:
        log.warning(f"В тестовых данных отсутствуют признаки: {missing}")

    used_features = [f for f in features if f in df.columns]
    X = df[used_features].copy()

    for col, med in median_map.items():
        if col in X.columns:
            X[col] = X[col].fillna(med)

    log.info(f"Детекция на {len(X)} записях, признаков: {len(used_features)}")

    raw_pred = pipeline.predict(X)   # -1 = anomaly, 1 = normal
    is_anomaly = (raw_pred == -1).astype(int)

    # anomaly_score
    model = pipeline.named_steps["model"]
    preprocess = pipeline.named_steps["preprocess"]

    try:
        X_trans = preprocess.transform(X)
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X_trans)
        elif hasattr(model, "score_samples"):
            scores = -model.score_samples(X_trans)
        else:
            scores = np.zeros(len(X))
    except Exception as e:
        log.warning(f"Не удалось получить anomaly_score: {e}")
        scores = np.zeros(len(X))

    result = df.copy()
    result["is_anomaly"] = is_anomaly
    result["anomaly_score"] = scores

    # Оценка precision, recall
    # Это предполагается, что метки для теста существуют
    y_true = result["label"]
    precision = precision_score(y_true, is_anomaly)
    recall = recall_score(y_true, is_anomaly)

    log.info(f"Precision: {precision:.4f}")
    log.info(f"Recall: {recall:.4f}")

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)

    log.info(f"Найдено аномалий: {is_anomaly.sum()} из {len(is_anomaly)} "
             f"({is_anomaly.mean() * 100:.2f}%)")
    log.info(f"Результаты сохранены в {out_path}")


# 3b. DETECT_ENSEMBLE (consensus)
@app.command(name="detect-ensemble")
def detect_ensemble(
    config: str = typer.Option("config/config.yaml", help="Путь к конфигу"),
    model_types: List[str] = typer.Option(
        ["isolation-forest", "lof", "one-class-svm"],
        help="Список моделей для энсембля (several --model-types ...)",
    ),
    input_file: str = typer.Option(
        None,
        help="Файл с данными для детекции (по умолчанию test_csv из конфига)",
    ),
    output_file: str = typer.Option(
        "artifacts/predictions_ensemble.csv",
        help="CSV для сохранения результатов энсембля",
    ),
    min_votes: int = typer.Option(
        2,
        help="Минимальное число моделей, чтобы признать точку аномальной (consensus)",
    ),
):
    """
    Энсемблевая детекция: несколько моделей голосуют, сколько из них считают точку аномальной.
    Добавляет:
      - is_anomaly_<model>
      - anomaly_score_<model>
      - consensus_votes
      - consensus_is_anomaly
      - is_anomaly (равен consensus_is_anomaly для удобной оценки)
    """
    cfg = load_config(config)
    log = setup_logging(cfg)

    if input_file is None:
        input_file = cfg["data"]["test_csv"]

    df = load_data(input_file)
    base_df = df.copy()
    votes_matrix = []

    for mtype in model_types:
        model_file = Path(cfg["paths"]["models_dir"]) / f"{mtype}_model.joblib"
        if not model_file.exists():
            log.warning(
                f"Модель {mtype} не найдена: {model_file}, пропускаем.")
            continue

        log.info(f"Загрузка модели {mtype} из {model_file}")
        pipeline: Pipeline = joblib.load(model_file)

        meta_path = model_file.parent / \
            f"{model_file.stem.replace('_model', '_metadata.yaml')}"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = yaml.safe_load(f)
            feat = metadata.get("features")
            median_map = metadata.get("median_map", {})
        else:
            log.warning(
                f"Метаданные для {mtype} не найдены, используем все числовые признаки")
            feat = None
            median_map = {}

        if feat is None:
            feat = df.select_dtypes(include=[np.number]).columns.tolist()

        used_features = [f for f in feat if f in df.columns]
        X = df[used_features].copy()

        for col, med in median_map.items():
            if col in X.columns:
                X[col] = X[col].fillna(med)

        raw_pred = pipeline.predict(X)
        is_anom = (raw_pred == -1).astype(int)

        model = pipeline.named_steps["model"]
        preprocess = pipeline.named_steps["preprocess"]
        try:
            X_trans = preprocess.transform(X)
            if hasattr(model, "decision_function"):
                scores = model.decision_function(X_trans)
            elif hasattr(model, "score_samples"):
                scores = -model.score_samples(X_trans)
            else:
                scores = np.zeros(len(X))
        except Exception as e:
            log.warning(f"Не удалось получить anomaly_score для {mtype}: {e}")
            scores = np.zeros(len(X))

        short = mtype.replace("-", "_")
        base_df[f"is_anomaly_{short}"] = is_anom
        base_df[f"anomaly_score_{short}"] = scores

        votes_matrix.append(is_anom)

    if not votes_matrix:
        log.error("Ни одна модель не была успешно загружена. Энсембль невозможен.")
        raise SystemExit(1)

    votes_matrix = np.vstack(votes_matrix)  # shape: (n_models, n_samples)
    votes_sum = votes_matrix.sum(axis=0)
    base_df["consensus_votes"] = votes_sum
    base_df["consensus_is_anomaly"] = (votes_sum >= min_votes).astype(int)

    # Для удобной оценки используем consensus как is_anomaly
    base_df["is_anomaly"] = base_df["consensus_is_anomaly"]

    # Оценка precision, recall
    y_true = base_df["label"]
    precision = precision_score(y_true, base_df["is_anomaly"])
    recall = recall_score(y_true, base_df["is_anomaly"])

    log.info(f"Precision: {precision:.4f}")
    log.info(f"Recall: {recall:.4f}")

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base_df.to_csv(out_path, index=False)

    log.info(
        f"Энсембль завершён: моделей={votes_matrix.shape[0]}, "
        f"consensus_anomalies={int(base_df['is_anomaly'].sum())} "
        f"({base_df['is_anomaly'].mean() * 100:.2f}%)"
    )
    log.info(f"Результаты энсембля сохранены в {out_path}")


# 4. EVALUATE
@app.command()
def evaluate(
    config: str = typer.Option("config/config.yaml", help="Путь к конфигу"),
    predictions_file: str = typer.Option(
        "artifacts/predictions.csv",
        help="CSV с предсказаниями (detect или detect-ensemble)",
    ),
    ground_truth_col: str = typer.Option(
        "label", help="Название колонки с истинными метками (0/1)"
    ),
):
    """
    Расширенная оценка:
    - Precision, Recall, ROC-AUC
    - Average Precision (AP)
    - precision@10/50/100/N_anom
    - FPR@TPR=0.90 и 0.95
    - confusion matrix, FP/day (если есть timestamp)
    - сохранение метрик в evaluation_metrics.yaml и PR/ROC графиков
    """
    cfg = load_config(config)
    log = setup_logging(cfg)
    create_directories(cfg)

    df = pd.read_csv(predictions_file)
    if ground_truth_col not in df.columns:
        log.error(
            f"Колонка {ground_truth_col} не найдена в {predictions_file}")
        raise SystemExit(1)

    y_true = df[ground_truth_col].astype(int).to_numpy()

    if "is_anomaly" in df.columns:
        y_pred = df["is_anomaly"].astype(int).to_numpy()
        pred_col = "is_anomaly"
    elif "prediction" in df.columns:
        y_pred = (df["prediction"] == 1).astype(int).to_numpy()
        pred_col = "prediction"
    else:
        log.error("В файле нет ни is_anomaly, ни prediction.")
        raise SystemExit(1)

    if "anomaly_score" in df.columns:
        scores = df["anomaly_score"].to_numpy()
    else:
        # если нет общего anomaly_score, берём бинарный
        scores = y_pred.astype(float)

    log.info("=== EVALUATE: расчёт базовых метрик ===")
    # Calculate precision and recall
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    # Calculate ROC-AUC
    try:
        roc_auc = roc_auc_score(y_true, scores)
    except Exception:
        roc_auc = float("nan")

    log.info(f"Precision: {precision:.4f}")
    log.info(f"Recall:    {recall:.4f}")
    log.info(f"ROC-AUC:   {roc_auc:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    log.info(f"Confusion matrix (tn, fp, fn, tp): {tn}, {fp}, {fn}, {tp}")

    # PR & AP
    precisions, recalls, _ = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)
    log.info(f"Average Precision (AP): {ap:.4f}")

    # precision@K
    def precision_at_k(y_true_bin: np.ndarray, y_score: np.ndarray, k: int) -> float:
        if k <= 0 or len(y_score) == 0:
            return 0.0
        order = np.argsort(-y_score)
        top_k = order[: min(k, len(y_score))]
        return float(y_true_bin[top_k].mean())

    n_anom = int(y_true.sum())
    prec_at_10 = precision_at_k(y_true, scores, 10)
    prec_at_50 = precision_at_k(y_true, scores, 50)
    prec_at_100 = precision_at_k(y_true, scores, 100)
    prec_at_n = precision_at_k(y_true, scores, n_anom) if n_anom > 0 else 0.0

    log.info(f"precision@10:     {prec_at_10:.4f}")
    log.info(f"precision@50:     {prec_at_50:.4f}")
    log.info(f"precision@100:    {prec_at_100:.4f}")
    log.info(f"precision@N_anom: {prec_at_n:.4f} (N={n_anom})")

    # FPR@TPR
    fpr, tpr, _ = roc_curve(y_true, scores)

    def fpr_at_tpr(target_tpr: float) -> float:
        if len(tpr) == 0:
            return float("nan")
        idx = np.argmin(np.abs(tpr - target_tpr))
        return float(fpr[idx])

    fpr_90 = fpr_at_tpr(0.90)
    fpr_95 = fpr_at_tpr(0.95)
    log.info(f"FPR@TPR=0.90: {fpr_90:.4f}")
    log.info(f"FPR@TPR=0.95: {fpr_95:.4f}")

    # FP/day (если есть timestamp)
    fp_per_day = None
    if "timestamp" in df.columns:
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            fp_mask = (y_true == 0) & (y_pred == 1)
            df_fp = df[fp_mask].copy()
            if not df_fp.empty:
                df_fp["date"] = df_fp["timestamp"].dt.date
                fp_by_day = df_fp.groupby("date").size()
                fp_per_day = float(fp_by_day.mean())
                log.info(f"Среднее количество FP в день: {fp_per_day:.2f}")
        except Exception as e:
            log.warning(f"Не удалось посчитать FP/day: {e}")

    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "roc_auc": float(roc_auc),
        "average_precision": float(ap),
        "precision_at_10": float(prec_at_10),
        "precision_at_50": float(prec_at_50),
        "precision_at_100": float(prec_at_100),
        "precision_at_N_anom": float(prec_at_n),
        "FPR_at_TPR_0_90": float(fpr_90),
        "FPR_at_TPR_0_95": float(fpr_95),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "fp_per_day": float(fp_per_day) if fp_per_day is not None else None,
        "predictions_file": str(predictions_file),
        "ground_truth_col": ground_truth_col,
        "prediction_col": pred_col,
        "timestamp": datetime.now().isoformat(),
    }
    with open(reports_dir / "evaluation_metrics.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(metrics, f, allow_unicode=True)

    # PR curve
    plt.figure(figsize=(8, 6))
    plt.step(recalls, precisions, where="post")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(reports_dir / "pr_curve.png", dpi=300)
    plt.close()

    # ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(reports_dir / "roc_curve.png", dpi=300)
    plt.close()

    log.info(f"Метрики и графики сохранены в {reports_dir}")


if __name__ == "__main__":
    app()
