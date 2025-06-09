from preprocess import VideoDataset  # 自作の動画データセットクラスをインポート
from vivit_model import ViViTClassifier # 自作のViViT風モデルをインポート
from torch.utils.data import DataLoader # PyTorchデータローダー
import torch  # PyTorch本体
import torch.nn as nn  # ニューラルネットワーク用のサブモジュール
import torch.optim as optim  # 最適化アルゴリズム
import tqdm as tqdm # プログレスバー表示用ライブラリ

# 使用可能なデバイスを選択（GPUがあれば使う）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 動画データセットを読み込み（フォルダ構成に依存）
dataset = VideoDataset("dataset", num_frames = 16)  # 動画データセットを読み込み（フォルダ構成に依存）

# データローダーの作成（バッチサイズ2、シャッフル有り）
dataloader = DataLoader(dataset, batch_size = 2, shuffle = True)

# 分類クラス数に応じたViViT風モデルのインスタンス化（例：2クラス）
model = ViViTClassifier(num_classes= 2).to(device)

# 損失関数：クロスエントロピー（分類問題の定番）
criterion = nn.CrossEntyopyLoss()

# 最適化アルゴリズム：Adam（学習率 1e-4）
optimizer = optim.Adam(model.parameters(), Lr = 1e-4)

"""
エポック数分の学習ループ ここでは5エポック
"""
for epoch in range(5):
    model.train() #モデルを訓練モードに設定
    

