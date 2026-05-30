import torch
import torch.nn as nn
import numpy as np


class SASRecModel(nn.Module):
    def __init__(self, cnt_item, max_seq_len=30, hidden_dim=64,
                 num_heads=2, num_layers=2, dropout=0.2,
                 cnt_authors=0, cnt_categories=0):
        super().__init__()

        self.max_seq_len = max_seq_len


        # эмбеддинг книг
        self.item_emb = nn.Embedding(cnt_item + 1, hidden_dim, padding_idx=0)

        # позиционный эмбеддинг: чтобы модель знала порядок книг
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)

        # Контентные эмбеддинги (если есть авторы и категории)
        # чтобы модель понимала: "книги этого автора похожи"
        if cnt_authors > 0:
            self.author_emb = nn.Embedding(cnt_authors + 1, hidden_dim, padding_idx=0)
        if cnt_categories > 0:
            self.category_emb = nn.Embedding(cnt_categories + 1, hidden_dim, padding_idx=0)

        self.emb_norm = nn.LayerNorm(hidden_dim)  # нормализация перед Transformer
        self.dropout = nn.Dropout(dropout)  # отключение нейронов (регуляризация)

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,  # размерность векторов
            nhead=num_heads,  # число голов внимания
            dim_feedforward=4 * hidden_dim,  # размер скрытого слоя в feed-forward
            dropout=dropout,  # регуляризация
            batch_first=True,  # батчи идут первым измерением
            activation='gelu'  # функция активации
        )

        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(hidden_dim)  # финальная нормализация
        self.output = nn.Linear(hidden_dim, cnt_item + 1)  # линейный слой

        self._init_weights()

    def _init_weights(self):
        # инициализация весов методом Xavier
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, item_ids, author_ids=None, category_ids=None):
        batch_size, seq_len = item_ids.shape
        item_emb = self.item_emb(item_ids) # id книг в векторы
        positions = torch.arange(seq_len, device=item_ids.device).unsqueeze(0)
        pos_emb = self.pos_emb(positions)
        x = item_emb + pos_emb #  книга + её позиция
        # контентные фичи (авторы, категории)
        if hasattr(self, 'author_emb') and author_ids is not None:
            x = x + self.author_emb(author_ids)
        if hasattr(self, 'category_emb') and category_ids is not None:
            x = x + self.category_emb(category_ids)

        x = self.emb_norm(x)
        x = self.dropout(x)
        x = self.layer_norm(x)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=item_ids.device), diagonal=1
        ).bool() # Каузальная маска — запрещает смотреть в будущее

        x = self.transformer(x, mask=causal_mask, is_causal=False)

        logits = self.output(x)
        return logits


    def predict_next(self, history_user, top_k=10):
        # предсказание топ-K следующих книг
        self.eval()
        with torch.no_grad():
            hist = history_user[-self.max_seq_len:]
            padded = [0] * (self.max_seq_len - len(hist)) + hist
            tensor = torch.tensor([padded], dtype=torch.long,
                                    device=next(self.parameters()).device)
            all_scores = self.forward(tensor)
            scores = all_scores[0, -1, :].cpu().numpy()
            rec_ids = np.argsort(scores)[::-1][:top_k]
            rec_scores = scores[rec_ids]
        return rec_ids, rec_scores


def negative_sampling_loss(scores, positive_items, cnt_item, num_negatives=100):
    # Функция потерь с Negative Sampling
    batch_size = scores.shape[0]

    pos_scores = scores[torch.arange(batch_size), positive_items] # скоры правильных книг
    neg_ids = torch.randint(1, cnt_item, (batch_size, num_negatives),
                            device=scores.device) # случайно выбираем num_negatives неправильных книг
    neg_scores = scores.gather(1, neg_ids)

    pos_term = -torch.nn.functional.logsigmoid(pos_scores)
    neg_term = -torch.nn.functional.logsigmoid(-neg_scores).sum(dim=1)

    loss = pos_term + neg_term
    return loss.mean()

