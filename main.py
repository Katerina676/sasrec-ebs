import tkinter as tk
import torch
import numpy as np
import pandas as pd
import random
import os
import urllib.request
from PIL import Image, ImageTk
from io import BytesIO
import threading

from model import SASRecModel

# загружаем предобработанные данные и модель
print("Загрузка...")
device = torch.device('cpu')
data = torch.load('preprocessed_data.pt', map_location=device)
item_id_to_idx = data['item_id_to_idx']
idx_to_item = {v: k for k, v in item_id_to_idx.items()}

model = SASRecModel(cnt_item=data['cnt_item'], max_seq_len=30,
    hidden_dim=64, num_heads=2, num_layers=2, dropout=0.2,
    cnt_authors=data['cnt_author'], cnt_categories=data['cnt_category']).to(device)
model.load_state_dict(torch.load('checkpoints/sasrec_best.pth', map_location=device))
model.eval()

# метаданные книг
cache = os.path.expanduser('~/.cache/kagglehub/datasets/mohamedbakhet/amazon-books-reviews/versions/1')
books_df = pd.read_csv(os.path.join(cache, 'books_data.csv'), usecols=['Title', 'image'])
book_info = {r['Title']: r['image'] for _, r in books_df.iterrows() if pd.notna(r['Title'])}

# оставляем только книги из словаря модели и пользователей с >= 5 книг
ratings_df = pd.read_csv(os.path.join(cache, 'Books_rating.csv'), usecols=['User_id', 'Id', 'Title'])
ratings_df = ratings_df.rename(columns={'User_id': 'user_id', 'Id': 'book_id', 'Title': 'title'})
ratings_df = ratings_df[ratings_df['book_id'].isin(set(item_id_to_idx.keys()))]
ratings_df = ratings_df[ratings_df.groupby('user_id')['user_id'].transform('count') >= 5]

user_books = ratings_df.groupby('user_id')['book_id'].apply(list)
user_titles = ratings_df.groupby('user_id')['title'].apply(list)
user_ids = list(user_books.index)

# словарь: ID книги -> название
book_id_to_title = dict(zip(ratings_df['book_id'], ratings_df['title']))
print(f"Готово: {len(user_ids)} пользователей")

# gui
root = tk.Tk()
root.title("SASRec — Рекомендации книг")
root.geometry("950x750")

tk.Label(root, text="ID пользователя:").pack(pady=(10, 0))
user_entry = tk.Entry(root, width=45)
user_entry.pack()

tk.Label(root, text="Количество рекомендаций:").pack()
k_var = tk.IntVar(value=10)
tk.Spinbox(root, from_=5, to=50, textvariable=k_var, width=5).pack()

status_var = tk.StringVar(value="Готов")
tk.Label(root, textvariable=status_var, fg="gray").pack()

# кнопки
btn_frame = tk.Frame(root)
btn_frame.pack(pady=10)
tk.Button(btn_frame, text="Случайный пользователь",
    command=lambda: (user_entry.delete(0, tk.END), user_entry.insert(0, random.choice(user_ids)))
).pack(side=tk.LEFT, padx=5)

# запуск в отдельном потоке чтобы интерфейс не зависал
tk.Button(btn_frame, text="Получить рекомендации",
    command=lambda: threading.Thread(target=get_recommendations, daemon=True).start()
).pack(side=tk.LEFT, padx=5)

# блок истории (последние 10 книг)
history_frame = tk.LabelFrame(root, text="Последние 10 книг", padx=5, pady=5)
history_frame.pack(fill=tk.X, padx=10, pady=(10, 5))
history_text = tk.Text(history_frame, height=2, wrap=tk.WORD, font=("Arial", 9))
history_text.pack(fill=tk.X)

# блок рекомендаций (прокручиваемый)
rec_frame = tk.LabelFrame(root, text="Рекомендации", padx=5, pady=5)
rec_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

# Canvas с прокруткой для карточек книг
canvas = tk.Canvas(rec_frame, highlightthickness=0)
scrollbar = tk.Scrollbar(rec_frame, orient="vertical", command=canvas.yview)
cards_frame = tk.Frame(canvas)
cw = canvas.create_window((0, 0), window=cards_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# растягиваем cards_frame на ширину canvas при изменении размера окна
def on_canvas_configure(event):
    canvas.itemconfig(cw, width=event.width)
canvas.bind("<Configure>", on_canvas_configure)

cards_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

# параметры сетки карточек
COLS = 5        # карточек в ряд
CARD_W = 150    # ширина карточки


# глобальный список чтобы PhotoImage не удалялись сборщиком мусора
photos_ref = []

# функция загружает обложку книги по URL
def load_image(url):
    try:
        if url and str(url) != 'nan':
            with urllib.request.urlopen(str(url), timeout=5) as u:
                img = Image.open(BytesIO(u.read())).resize((100, 150), Image.Resampling.LANCZOS)
                return ImageTk.PhotoImage(img)
    except:
        pass
    return ImageTk.PhotoImage(Image.new('RGB', (100, 150), color='lightgray'))

# получение рекомендаций от модели
def get_recommendations():
    global photos_ref
    uid = user_entry.get().strip()
    if uid not in user_books:
        status_var.set("Пользователь не найден")
        return

    status_var.set("Загрузка...")
    books = user_books[uid][-30:]
    history_text.delete(1.0, tk.END)
    history_text.insert(1.0, " → ".join(user_titles[uid][-10:]))

    history_idx = [item_id_to_idx.get(b, 0) for b in books]
    indices, scores = model.predict_next(history_idx, top_k=k_var.get())
    # очищаем старые карточки
    for w in cards_frame.winfo_children():
        w.destroy()

    photos_ref = []
    try:
        for i, (idx, score) in enumerate(zip(indices, scores)):
            if not cards_frame.winfo_exists():
                break

            title = book_id_to_title.get(idx_to_item.get(idx, ''), f'Книга {idx}')
            prob = 1 / (1 + np.exp(-float(score)))
            # карточка книги
            card = tk.Frame(cards_frame, relief=tk.RIDGE, borderwidth=1, width=CARD_W)
            card.grid(row=i // COLS, column=i % COLS, padx=8, pady=8)
            card.grid_propagate(False)
            # обложка
            photo = load_image(book_info.get(title, ''))
            photos_ref.append(photo)
            tk.Label(card, image=photo).pack(pady=(5, 0))
            tk.Label(card, text=str(title)[:35], wraplength=130, font=("Arial", 8)).pack(pady=2)
            tk.Label(card, text=f"Скор: {float(score):.3f}", font=("Arial", 7), fg="blue").pack()
            tk.Label(card, text=f"{prob:.1%}", font=("Arial", 9, "bold"), fg="green").pack()

        status_var.set(f"Готово — {k_var.get()} рекомендаций")
    except tk.TclError:
        pass

# запуск приложения
root.mainloop()
