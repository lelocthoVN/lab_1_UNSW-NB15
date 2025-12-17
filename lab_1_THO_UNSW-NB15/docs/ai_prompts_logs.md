# Лог ключевых промптов

## (вариант Isolation Forest / drop / standard / precision-recall)

Документ фиксирует ключевые запросы к AI-ассистенту и то, как эти ответы были превращены в изменения в коде `scripts/anomaly_cli.py`.

---

### Промпт 1: Построить пайплайн строго по варианту “Isolation Forest; подготовка: drop, standard; метрики: precision/recall”. Составь структуру CLI-проекта и план пайплайна (EDA → train → detect → evaluate).

**Ответ AI:**

- оставить команды: `eda`, `train`, `detect`, `evaluate`;
- реализовать подготовку: `drop` (исключения), `standard` (StandardScaler);
- ориентация отчёта на precision/recall.

**Какие изменения внесены в код:**

- `train()`: pipeline `preprocess(StandardScaler) -> IsolationForest`;
- `evaluate()`: метрики фокус на `precision`, `recall`.

---

### Промпт 2: Как реализовать “drop + standard” технически правильно в sklearn Pipeline, чтобы одинаково работало и в train, и в detect?

**Ответ AI:**

- `ColumnTransformer`: numeric → `StandardScaler`, categorical → `OneHotEncoder(handle_unknown="ignore")`;
- один `Pipeline` сохраняется и используется повторно.

**Какие изменения внесены в код:**

- `train()`: добавлен `ColumnTransformer([...])` и `Pipeline([("preprocess", ...), ("model", IsolationForest(...))])`;
- модель сохраняется как один объект `.joblib` (preprocess + model вместе).

---

### Промпт 3: Как считать precision/recall для Isolation Forest, если модель возвращает -1/1?

**Ответ AI:**

- `is_anomaly = (raw_pred == -1).astype(int)`;
- precision/recall считать по `label` (если есть);
- дополнительно сохранять confusion matrix для интерпретации FP/FN.

**Какие изменения внесены в код:**

- `detect()`: добавлено преобразование `raw_pred -> is_anomaly`;
- `evaluate()`: рассчитывает `precision_score`, `recall_score`, `confusion_matrix`;
- `evaluate()`: результаты пишутся в `artifacts/reports/evaluation_metrics.yaml`.

---

### Промпт 4: Почему в `feature_distributions.png` рисуется 3–4 графика вместо 6?

**Ответ AI:**

- часть полей из списка может отсутствовать в датафрейме (после выборки/преобразований);
- нужно логировать список `key_features` и проверять пересечение с `df.columns`.

---

### Промпт 5: У меня в варианте подготовка **drop**. Подскажи, какие колонки точно надо исключить из обучения и почему (id/label/attack_cat)? Как убедиться, что я реально “drop” сделал?

**Ответ AI:**

Под “drop” в UNSW-NB15 логично исключать поля, которые не описывают поведение трафика или дают утечку ответа:

- **`id`** — просто идентификатор строки, не несёт сетевого смысла.
- **`label`** — истинная метка (0/1), её нельзя давать модели при unsupervised обучении.
- **`attack_cat`** — тип атаки (фактически “пояснение после”), тоже приводит к leakage.

**Как убедиться, что drop реально выполнен:**

1. вывести в лог `feature_cols` и проверить, что `id/label/attack_cat` там отсутствуют;
2. вывести “сколько было фич → сколько осталось после exclude”;
3. при желании — лог “Dropped columns: …”.

**Какие изменения внесены в код:**

- `train()`: `exclude = cfg["features"]["exclude"]`, формирование `feature_cols` и лог до/после drop.

---

### Промпт 6: В варианте **standard** — зачем StandardScaler, если Isolation Forest “может и без него”?

**Ответ AI:**
Isolation Forest не всегда требует scaling, но в варианте **standard** это обязательное условие и оно полезно:

- признаки в UNSW-NB15 имеют разные масштабы (байты/пакеты/время) → без стандартизации часть признаков доминирует;
- при наличии One-Hot (0/1) + числовых признаков стандартизация делает вклад числовых стабильнее;
- главное — **консистентность train/detect**: одинаковый preprocessing гарантируется только через Pipeline.

**Какие изменения внесены в код:**

- `train()`: `StandardScaler()` для numeric внутри `ColumnTransformer`;
- сохранение Pipeline целиком, чтобы `detect()` применял тот же scaling автоматически.

* `train()`: сформирован `exclude = cfg["features"]["exclude"]` и `feature_cols = [f for f in all_features if f not in exclude]`;
* добавлен лог: сколько фич до/после drop.

---

### Промпт 7: Как анализировать F1 и ROC-AUC для unsupervised моделей?

**Ответ AI:**

- использовать anomaly_score;
- строить PR и ROC кривые;
- интерпретировать FP/FN через confusion matrix;
- F1 — основная метрика по варианту.

---

### Промпт 8: У меня метрики precision/recall. Объясни, как интерпретировать confusion matrix именно для моей задачи (что такое FP/FN в anomaly detection)?

**Ответ AI:**

- `evaluate()`: добавлен расчёт `confusion_matrix`, сохранение `tn/fp/fn/tp` в `evaluation_metrics.yaml`;
- в логах выводится матрица (tn, fp, fn, tp).

---

Этот лог демонстрирует реальный итеративный процесс: уточнение требований варианта → устранение ошибок совместимости признаков → корректный подсчёт precision/recall → воспроизводимые артефакты для отчёта.
