#!/usr/bin/env python3
"""
UNSW-NB15 Anomaly Detection CLI
Командная строка для обнаружения аномалий в сетевом трафике
"""

import os
import sys
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import typer
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# ML библиотеки
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Визуализация
import matplotlib
matplotlib.use('Agg')  # Для работы без дисплея
import matplotlib.pyplot as plt
import seaborn as sns

app = typer.Typer(help="CLI для обнаружения аномалий в сетевом трафике UNSW-NB15")

# Глобальные переменные
logger = None

def setup_logging(config: Dict) -> logging.Logger:
    """Настройка системы логирования"""
    log_config = config.get('logging', {})

    # Создаем директорию для логов
    logs_dir = Path(config['paths']['logs_dir'])
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Настраиваем логгер
    logger = logging.getLogger('anomaly_detection')
    logger.setLevel(getattr(logging, log_config.get('level', 'INFO')))

    # Форматтер
    formatter = logging.Formatter(
        log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )

    # Файловый обработчик
    if log_config.get('file_enabled', True):
        file_handler = logging.FileHandler(
            logs_dir / f"anomaly_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Консольный обработчик
    if log_config.get('console_enabled', True):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

def load_config(config_path: str) -> Dict:
    """Загрузка конфигурационного файла"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def create_directories(config: Dict) -> None:
    """Создание необходимых директорий"""
    for path_key, path_value in config['paths'].items():
        Path(path_value).mkdir(parents=True, exist_ok=True)

def load_data(file_path: str, sample_size: Optional[int] = None) -> pd.DataFrame:
    """Загрузка данных с опциональной выборкой"""
    logger.info(f"Загрузка данных из {file_path}")

    if not Path(file_path).exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    df = pd.read_csv(file_path)
    logger.info(f"Загружено {len(df)} записей")

    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        logger.info(f"Выбрана случайная выборка из {sample_size} записей")

    return df

@app.command()
def eda(
    config: str = typer.Option("config/config.yaml", help="Путь к конфигурационному файлу"),
    sample_size: Optional[int] = typer.Option(None, help="Размер выборки для анализа")
):
    """Выполнить разведочный анализ данных (EDA)"""
    global logger

    # Загрузка конфигурации
    cfg = load_config(config)
    logger = setup_logging(cfg)
    create_directories(cfg)

    logger.info("=== Начало разведочного анализа данных ===")

    # Загрузка данных
    train_data = load_data(
        cfg['data']['train_csv'], 
        sample_size or cfg['eda'].get('sample_size')
    )

    # Настройка стиля графиков
    plt.style.use(cfg['eda'].get('plot_style', 'default'))

    # Создание отчета EDA
    eda_dir = Path(cfg['paths']['eda_dir'])

    # Базовая статистика
    logger.info("Генерация базовой статистики...")
    basic_stats = train_data.describe()
    basic_stats.to_csv(eda_dir / 'basic_statistics.csv')

    # Анализ пропусков
    missing_data = train_data.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)

    if not missing_data.empty:
        logger.warning(f"Найдены пропуски в {len(missing_data)} признаках")
        missing_data.to_csv(eda_dir / 'missing_values.csv')

    # Корреляционная матрица
    logger.info("Построение корреляционной матрицы...")
    numeric_cols = train_data.select_dtypes(include=[np.number]).columns
    correlation_matrix = train_data[numeric_cols].corr()

    plt.figure(figsize=cfg['eda']['figure_size'])
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title('Корреляционная матрица признаков')
    plt.tight_layout()
    plt.savefig(eda_dir / 'correlation_matrix.png', dpi=300)
    plt.close()

    # Распределения ключевых признаков
    logger.info("Анализ распределений признаков...")
    key_features = (cfg['features']['network_basic'] + 
                   cfg['features']['transport'][:3] + 
                   cfg['features']['statistical'][:3])

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    for i, feature in enumerate(key_features[:9]):
        if feature in train_data.columns:
            train_data[feature].hist(bins=50, ax=axes[i], alpha=0.7)
            axes[i].set_title(f'Распределение {feature}')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Частота')

    plt.tight_layout()
    plt.savefig(eda_dir / 'feature_distributions.png', dpi=300)
    plt.close()

    # Анализ выбросов (boxplot)
    logger.info("Анализ выбросов...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, feature in enumerate(key_features[:6]):
        if feature in train_data.columns:
            train_data.boxplot(column=feature, ax=axes[i])
            axes[i].set_title(f'Выбросы в {feature}')

    plt.tight_layout()
    plt.savefig(eda_dir / 'outliers_analysis.png', dpi=300)
    plt.close()

    # Создание HTML отчета
    html_report = f"""
    <html>
    <head><title>EDA Report - UNSW-NB15</title></head>
    <body>
    <h1>Разведочный анализ данных UNSW-NB15</h1>
    <h2>Основная статистика</h2>
    <p>Общее количество записей: {len(train_data)}</p>
    <p>Количество признаков: {len(train_data.columns)}</p>
    <p>Количество пропусков: {missing_data.sum()}</p>

    <h2>Графики</h2>
    <img src="correlation_matrix.png" alt="Корреляционная матрица">
    <img src="feature_distributions.png" alt="Распределения признаков">
    <img src="outliers_analysis.png" alt="Анализ выбросов">
    </body>
    </html>
    """

    with open(eda_dir / 'eda_report.html', 'w', encoding='utf-8') as f:
        f.write(html_report)

    logger.info(f"EDA завершен. Результаты сохранены в {eda_dir}")

@app.command()
def train(
    config: str = typer.Option("config/config.yaml", help="Путь к конфигурационному файлу"),
    model_type: str = typer.Option("isolation-forest", help="Тип модели: isolation-forest, lof, one-class-svm"),
    tune_params: bool = typer.Option(False, help="Выполнить подбор гиперпараметров")
):
    """Обучить модель обнаружения аномалий"""
    global logger

    cfg = load_config(config)
    logger = setup_logging(cfg)
    create_directories(cfg)

    logger.info(f"=== Начало обучения модели {model_type} ===")

    # Загрузка данных
    train_data = load_data(cfg['data']['train_csv'])

    # Подготовка признаков
    all_features = (cfg['features']['network_basic'] + 
                   cfg['features']['transport'] + 
                   cfg['features']['statistical'] + 
                   cfg['features']['contextual'] + 
                   cfg['features']['additional'])

    # Выбираем только существующие признаки
    available_features = [f for f in all_features if f in train_data.columns]

    # Исключаем целевую переменную и ID
    exclude_features = cfg['features'].get('exclude', ['id', 'label', 'attack_cat'])
    feature_columns = [f for f in available_features if f not in exclude_features]

    X = train_data[feature_columns]

    logger.info(f"Используется {len(available_features)} признаков для обучения")

    # Обработка пропусков
    median_dict = {}
    for col in X.columns:
        if X[col].dtype in ['int64', 'float64']:
            median_val = float(X[col].median())
            median_dict[col] = median_val
            X[col].fillna(median_val, inplace=True)

    # Создание модели
    model_config = cfg['models'][model_type.replace('-', '_')]

    if model_type == "isolation-forest":
        model = IsolationForest(**model_config)
    elif model_type == "lof":
        model = LocalOutlierFactor(**model_config)  
    elif model_type == "one-class-svm":
        model = OneClassSVM(**model_config)
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")

    # Создание pipeline с масштабированием
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])

    # Обучение
    logger.info("Обучение модели...")
    pipeline.fit(X)

    # Сохранение модели
    models_dir = Path(cfg['paths']['models_dir'])
    model_path = models_dir / f"{model_type}_model.joblib"
    joblib.dump(pipeline, model_path)

    # Сохранение метаданных
    metadata = {
        'model_type': model_type,
        'features': available_features,
        'training_samples': len(X),
        'model_params': model_config,
        'timestamp': datetime.now().isoformat(),
        'median': median_dict
    }

    metadata_path = models_dir / f"{model_type}_metadata.yaml"
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f)

    logger.info(f"Модель сохранена: {model_path}")
    logger.info(f"Метаданные сохранены: {metadata_path}")

@app.command()
def detect(
    config: str = typer.Option("config/config.yaml", help="Путь к конфигурационному файлу"),
    model_path: str = typer.Option(None, help="Путь к обученной модели"),
    input_file: str = typer.Option(None, help="Файл для детекции аномалий"),
    output_file: str = typer.Option("artifacts/predictions.csv", help="Файл для сохранения результатов")
):
    """Обнаружить аномалии в данных"""
    global logger

    cfg = load_config(config)
    logger = setup_logging(cfg)

    logger.info("=== Начало детекции аномалий ===")

    # Определение путей
    if not model_path:
        model_path = Path(cfg['paths']['models_dir']) / "isolation-forest_model.joblib"

    if not input_file:
        input_file = cfg['data']['test_csv']

    # Загрузка модели
    logger.info(f"Загрузка модели из {model_path}")
    model = joblib.load(model_path)

    # Загрузка данных
    test_data = load_data(input_file)

    # Загрузка метаданных модели
    metadata_path = Path(model_path).parent / f"{Path(model_path).stem.replace('_model', '_metadata.yaml')}"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = yaml.safe_load(f)
            features = metadata['features']
    else:
        logger.warning("Метаданные модели не найдены, используем все доступные признаки")
        features = test_data.select_dtypes(include=[np.number]).columns.tolist()

    # Подготовка данных
    X = test_data[features]
    median_dict = metadata['median']
    for col in X.columns:
        if X[col].dtype in ['int64', 'float64']:
            median_val = median_dict[col]
            X[col].fillna(median_val, inplace=True)

    # Предсказание
    logger.info("Выполнение детекции аномалий...")
    pred = model.predict(X)  # -1 для аномалий, 1 для нормальных
    predictions = (pred == 1).astype(int)  # Конвертируем в 0/1

    # Получение anomaly scores если возможно
    if hasattr(model.named_steps['model'], 'decision_function'):
        scores = model.decision_function(X)
    elif hasattr(model.named_steps['model'], 'score_samples'):
        scores = model.named_steps['model'].score_samples(X)
    else:
        scores = np.zeros(len(predictions))

    # Конвертация предсказаний (-1/1 -> 1/0)
    anomaly_labels = (pred == -1).astype(int)

    # Создание результатов
    results = test_data.copy()
    results['anomaly_score'] = scores
    results['is_anomaly'] = anomaly_labels
    results['prediction'] = predictions

    # Сохранение результатов
    results.to_csv(output_file, index=False)

    # Статистика
    n_anomalies = anomaly_labels.sum()
    anomaly_rate = n_anomalies / len(anomaly_labels) * 100

    logger.info(f"Найдено аномалий: {n_anomalies} из {len(anomaly_labels)} ({anomaly_rate:.2f}%)")
    logger.info(f"Результаты сохранены в {output_file}")

@app.command()
def evaluate(
    config: str = typer.Option("config/config.yaml", help="Путь к конфигурационному файлу"),
    predictions_file: str = typer.Option("artifacts/predictions.csv", help="Файл с предсказаниями"),
    ground_truth_col: str = typer.Option("label", help="Название колонки с истинными метками")
):
    """Оценить качество детекции аномалий"""
    global logger

    cfg = load_config(config)
    logger = setup_logging(cfg)

    logger.info("=== Начало оценки модели ===")

    # Загрузка результатов
    results = pd.read_csv(predictions_file)

    if ground_truth_col not in results.columns:
        logger.error(f"Колонка {ground_truth_col} не найдена в файле")
        return

    # Получение истинных меток и предсказаний
    y_true = results[ground_truth_col]
    y_pred = results['is_anomaly']
    y_scores = results.get('anomaly_score', np.zeros(len(y_pred)))

    # Расчет метрик
    from sklearn.metrics import precision_score, recall_score, f1_score

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    if len(np.unique(y_true)) > 1 and len(np.unique(y_scores)) > 1:
        roc_auc = roc_auc_score(y_true, y_scores)
    else:
        roc_auc = None

    # Вывод результатов
    logger.info("=== Метрики оценки ===")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1-Score: {f1:.4f}")
    if roc_auc:
        logger.info(f"ROC-AUC: {roc_auc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    logger.info("\nConfusion Matrix:")
    logger.info(f"TN: {cm[0,0]}, FP: {cm[0,1]}")
    logger.info(f"FN: {cm[1,0]}, TP: {cm[1,1]}")

    # Сохранение метрик
    reports_dir = Path(cfg['paths']['reports_dir'])
    reports_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc) if roc_auc else None,
        'confusion_matrix': cm.tolist(),
        'evaluation_timestamp': datetime.now().isoformat()
    }

    with open(reports_dir / 'evaluation_metrics.yaml', 'w') as f:
        yaml.dump(metrics, f)

    logger.info(f"Метрики сохранены в {reports_dir / 'evaluation_metrics.yaml'}")

if __name__ == "__main__":
    app()
