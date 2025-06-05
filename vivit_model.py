import torch
import torch.nn as nn #ニューラルネットワーク用のモジュール

class ViViTClassifier(nn.Module):
    def __init__(self ,num_classes, num_frames=16, image_size=224, patch_size=16, dim=256):
        """
        動画の分類を行うViViT風モデルの初期化
        :param num_classes: 分類するクラスの数
        :param num_frames: 1動画あたりのフレーム数
        :param image_size: 入力画像サイズ（高さ・幅）
        :param patch_size: パッチの1辺のサイズ
        :param dim: Transformer内部で使う特徴量の次元数
        """
        super().__init__()

        self.num_patches = (image_size // patch_size) ** 2  # 各フレームあたりのパッチ数
        self.patch_dim = 3 * patch_size * patch_size
        self.num_frames = num_frames

        

