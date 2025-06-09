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

        # 各パッチをTransformer用の埋め込みベクトルに変換
        self.to_patch_embedding = nn.Linar(self.patch_dim, dim)  
        # 位置エンコーディング（時空間の順番をモデルに教える）
        self.pos_embedding = nn.parameter(
            torch.randn(1 , num_frames * self.num_patches + 1,dim)
        )
        # CLSトークン：分類結果を取り出すための特別なトークン
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # Transformerエンコーダー（自己注意機構）
        self.transformer = nn.transformerEncoder(
            nn.TransformerEncoderLayer(d_modeL = dim, nhead=8),  # ヘッド数8
            num_layers=4  # 4層構成
        )
        # 分類用MLP（CLSトークンの出力を分類スコアに変換）
        self.mlp_head = nn.sequential(
            nn.LayerNorm(dim),  # 正規化
            nn.Linear(dim, num_classes)  # 出力クラス数に変換
        )

    #forwardメソッド
    def forward(self, x):
        """
        モデルの順伝播処理
        :param x: 入力テンソル [B, T, C, H, W]
        :return: 分類スコア [B, num_classes]
        """

        # バッチ・フレーム単位で並べ替え → [B*T, C, H, W]        
        B,T,C,H,W = x.shape   # バッチサイズ、フレーム数、チャネル、高さ、幅
        x = x.reshape(B * T,C,H,W)

        # フレームを16×16パッチに分割 → パッチ数 × パッチサイズ に整形
        patches = x.unfold(2,16,16).unfold(3,16,16) # パッチ分割
        patches = patches.contiguous().view(B, T * self.num_patches, -1)  # [B, T*patch数, patch_dim] 

        # パッチをベクトルに埋め込む（Transformer用の入力に変換）
        tokens = self.to_patch_embedding(patches) # [B, sequence_length, dim]

        # CLSトークンを追加
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat((cls_tokens,tokens), dim = 1) # [B, sequence_length + 1, dim]
        
        # 位置エンコーディングを追加
        tokens += self.pos_embedding[:, :tokens.size(1), :] # [B, sequence_length + 1, dim]

        # Transformerエンコーダーに入力
        x = self.transformer(tokens)  # [B, sequence_length + 1, dim]

        # CLSトークンの出力を取り出す
        return self.mlp_head(x[:, 0]) # CLSトークンの出力を分類スコアに変換 [B, num_classes]