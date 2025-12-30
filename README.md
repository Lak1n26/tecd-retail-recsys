# Нейросетевые подходы к разработке рекомендаций по повышению продаж в товарных корзинах

<p align="center">
  <a href="#about-project">About project</a> •
  <a href="#setup">Setup</a> •
  <a href="#train">Train</a> •
  <a href="#demo">Demo</a> •
  <a href="#project-structure">Project structure</a>
</p>


## About project

Рекомендательная система на корзинах для повышения продаж товаров. Данная система учитывает индивидуальные истории покупок, сезонность, зависимости между товарами и прочие особенности поведения пользователей.

**Метрики**: стандартные метрики для задачи рекомендаций: HitRate@K, Precision@K, Recall@K, NDCG@K. Так как планируется большое количество данных (и юзеров, и товаров), то прогнозное значение метрик составит около 0.02-0.03 при K=100.

**Валидация и тест**: использую временную отложенную валидацию (Global Temporal Split): train – первые N дней, валидация – следующие 2 дня, тест – последние 2 дня. Это обеспечит воспроизводимость валидации.

**Датасет**: используется датасет T-ECD ([хабр](https://habr.com/ru/companies/tbank/articles/950696/), [hugging face](https://huggingface.co/datasets/t-tech/T-ECD)). Это кросс-доменный датасет (Marketplace, Retail, Offers, Reviews, Payments), содержит данные за 227 дней. В датасете представлены данные о 44 млн пользователей и 30 млн айтемов, всего – 135 млрд взаимодействий. Есть уменьшенная версия датасета на миллиард взаимодействий, ее и использовал. Рекомендации осуществляются на домене Retail.

**Основная модель**: в качестве основной модели выступает MLP-Ranker — многослойный перцептрон для ранжирования пар user-item. Обучение с BPR (Bayesian Personalized Ranking) loss и негативным сэмплированием.

## Setup

```bash
# Клонирование репозитория
git clone https://github.com/yourusername/tecd-retail-recsys.git
cd tecd-retail-recsys

# Создание виртуального окружения с uv
uv venv
source .venv/bin/activate  # Linux/macOS
# или
.venv\Scripts\activate  # Windows

# Установка зависимостей
uv pip install -e ".[dev]"

# Установка pre-commit хуков
pre-commit install
```

Данные хранятся в публичном Yandex Object Storage.

```bash
# Скачайте данные через DVC
dvc pull
```

## Train

Препроцессинг данных

```bash
tecd-recsys preprocess

# С кастомными параметрами
tecd-recsys preprocess --overrides="data.min_user_interactions=10,data.min_item_interactions=10"
```

Обучение модели

```bash
# Запуск обучения с дефолтными параметрами
tecd-recsys train

# С изменением гиперпараметров
tecd-recsys train --overrides="train.epochs=30,train.learning_rate=0.0005,model.dropout=0.3"

# С указанием конкретного конфига
tecd-recsys train --config-path=/path/to/configs
```

Оценка модели

```bash
# Оценка на тестовом наборе
tecd-recsys evaluate --checkpoint=outputs/checkpoints/best.ckpt
```

Инференс

```bash
# Генерация рекомендаций
tecd-recsys infer --checkpoint=outputs/checkpoints/best.ckpt --output=recommendations.csv
```

## Demo

Пример работы с проектом представлен в [demo.ipynb](demo.ipynb)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qXixMZJHfTh2UN4gc5R9x4ZZS0wK77We?usp=sharing)


## Project structure

```
tecd-retail-recsys/
├── configs/                     # Hydra конфигурации
│   ├── config.yaml              # Основной конфиг
│   ├── data/
│   │   └── default.yaml         # Параметры данных
│   ├── model/
│   │   └── mlp.yaml             # Параметры модели
│   └── train/
│       └── default.yaml         # Параметры обучения
├── tecd_retail_recsys/          # Основной пакет
│   ├── __init__.py
│   ├── commands.py              # CLI команды
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py           # PyTorch Dataset/DataModule
│   │   └── preprocessing.py     # Препроцессинг данных
│   ├── models/
│   │   ├── __init__.py
│   │   └── mlp_ranker.py        # MLP модель
│   ├── train.py                 # Обучение
│   └── infer.py                 # Инференс
├── .dvc/                        # DVC конфигурация
│   └── config                   # Remote storage настройки
├── t_ecd_small_partial.dvc      # DVC tracking файл
├── .pre-commit-config.yaml      # Pre-commit хуки
├── pyproject.toml               # Зависимости и настройки
├── uv.lock                      # Lock-файл зависимостей
├── demo.ipynb                   # Демо ноутбук
└── README.md
```
