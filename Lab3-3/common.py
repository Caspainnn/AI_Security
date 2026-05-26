

captcha_array=list("0123456789abcdefghijklmnopqrstuvwxyz")
captcha_size=4

# one_hot编码类
# class one_hot:
#     @staticmethod
#     def text2vec(text):
#         import torch
#         # 使用预计算的索引映射以提高性能
#         index_map = {char: idx for idx, char in enumerate(captcha_array)}
#         vectors = torch.zeros((captcha_size, len(captcha_array)))
#         for i, char in enumerate(text):
#             if char in index_map:
#                 vectors[i, index_map[char]] = 1
#             else:
#                 raise ValueError(f"character '{char}' not found in captcha array")
#         return vectors