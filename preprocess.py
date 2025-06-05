#動画をTensorに変換し、ラベルとともに返す。
import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset

class VideoDataset(Dataset):
    def __init__(self, root_dir, num_frames = 16):

        """
        :param root_dir: クラス別フォルダが格納されたルートディレクトリ
        :param num_frames: 1動画あたり取り出すフレーム数
        """

        self.root_dir = root_dir
        self.num_frames = num_frames
        self.video_paths, self.labels = [], []
        self.class_map = {name: i for i, name in enumerate(sorted(os.listdir(root_dir)))}

        for class_name in sorted(os.listdir(root_dir)):
            class_dir = os.path.join(root_dir, class_name)
            for video in os.listdir(class_dir):
                self.video_paths.append(os.apath.join(class_dir, video))
                self.labels.append(self.class_map[class_name])

def __len__(self):
    return len(self.video_paths)

def __getitem__(self, idx):
    """
    インデックスに対応する動画ファイルを読み込み、指定フレーム数を返す
    """

    cap = cv2.VideoCapture(self.video_paths[idx])
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total // self.num_frames, 1)

    frames = [] # 抽出したフレームを格納するリスト


    for i in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))# sizeをViViTの入力に合わせてリサイズ
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)#RGBに変換
        frames.append(frame)
        if len(frames) == self.num_frames:
            break
        cap.release()

    frames = np.array(frames).transpose(0, 3, 1, 2)

    frames = torch.tensor(frames /255.0, dtype=torch.float32)
    label = self.labels[idx]
    return frames, label

