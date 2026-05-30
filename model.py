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
