# Рекомендательная система для ЭБС на основе SASRec
Прототип рекомендательной системы для Электронной Библиотечной Системы (ЭБС) университета.
Модель SASRec (Self-Attentive Sequential Recommendation) с контентными эмбеддингами 
авторов и категорий предсказывает следующую книгу на основе истории чтения пользователя.

## Данные

- Датасет: Amazon Books Reviews (Kaggle)
- 
## Метрики

| Модель | NDCG@10 | Recall@10 |
|--------|---------|-----------|
| Базовая (dim=64) | 0.440 | 0.534 |
| Улучшенная (dim=128) | 0.459 | 0.548 |

## Структура проекта

| Файл                  | Назначение |
|-----------------------|------------|
| `model.py`            | Модель SASRec с контентными эмбеддингами и Negative Sampling |
| `eda.ipynb`           | Разведочный анализ данных |
| `preprocessing.ipynb` | Предобработка: нормализация, фильтрация, тензоры |
| `train_model.ipynb`   | Обучение базовой модели |
| `train_exp.ipynb`     | Эксперимент с hidden_dim=128 и num_negatives=300 |
| `tisasrec_exp.ipynb` | Эксперимент TiSASRec + RepeatNet |
| `contrastive_exp.ipynb` | Эксперимент Contrastive Learning + Item Dropout |
| `main.py`             | Демонстрационный GUI с карточками книг |
| `tests.ipynb`            | Тестирование (10 тестов) |


## Установка

### 1. Клонировать репозиторий
- git clone https://github.com/Katerina676/sasrec-ebs.git
- cd sasrec-ebs

### 2. Создать виртуальное окружение
- python -m venv venv

### 3. Активировать окружение

### Windows:
- venv\Scripts\activate
### Linux/Mac:
- source venv/bin/activate

### 4. Установить зависимости

- pip install -r requirements.txt

### Если у вас NVIDIA GPU:
- сначала удалить torch потом установить верный
- pip uninstall torch torchvision torchaudio -y

- pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
### в colab
- будет по умолчанию cuda если выбрать tesla t4

## Запуск GUI
- python main.py