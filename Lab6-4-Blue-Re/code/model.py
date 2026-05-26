import paddle
from paddle.nn import Linear, Embedding, LSTM
import os
import config as cfg
from pre_data import load_vocab

vocab = load_vocab(os.path.join(cfg.data_root_path, 'dict.txt'))

class RNN(paddle.nn.Layer):
    def __init__(self):
        super(RNN, self).__init__()
        self.dict_dim = len(vocab)      # 字典大小
        self.emb_dim = 128              # 词向量维度
        self.hid_dim = 128              # LSTM 隐藏层维度
        self.class_dim = 3              # 三分类
        
        # 嵌入层
        self.embedding = Embedding(
            num_embeddings=self.dict_dim,
            embedding_dim=self.emb_dim,
            sparse=False)
        
        # 线性层 + LSTM + 分类层
        self.fc1 = Linear(self.emb_dim, self.hid_dim)
        self.lstm = LSTM(self.hid_dim, self.hid_dim)
        self.fc2 = Linear(self.hid_dim, self.class_dim)

    def forward(self, inputs):
        emb = self.embedding(inputs)         # [batch_size, seq_len, emb_dim]
        fc1 = self.fc1(emb)                  # [batch_size, seq_len, hid_dim]
        lstm_out, (hidden, _) = self.lstm(fc1)   # hidden: [num_layers, batch, hid_dim]
        x = hidden[-1]                       # 取最后一层的隐藏状态 [batch_size, hid_dim]
        x = self.fc2(x)                      # [batch_size, class_dim]
        return x

if __name__ == '__main__':
    rnn = RNN()
    paddle.summary(rnn, (32, 150), "int64")
