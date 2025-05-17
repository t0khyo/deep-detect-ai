import cv2
import torch
import numpy as np
from torch import nn
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data.dataset import Dataset

class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional=bidirectional, bias=False)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))

class VideoDataset(Dataset):
    def __init__(self, video_path, sequence_length=60, transform=None, output_video_path='cropped_output_video.mp4'):
        self.video_path = video_path
        self.transform = transform
        self.count = sequence_length
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.output_video_path = output_video_path
        self.out_writer = None
        self.last_bbox = None
        self.padding = 50

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        frames = []
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        a = max(1, total_frames // self.count)
        first_frame = np.random.randint(0, a)
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count < first_frame:
                frame_count += 1
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) > 0:
                x, y, w, h = faces[0]
                self.last_bbox = (x, y, w, h)
            elif self.last_bbox is not None:
                x, y, w, h = self.last_bbox
            else:
                continue

            height, width = frame.shape[:2]
            x = max(0, x - self.padding)
            y = max(0, y - self.padding)
            w = min(width - x, w + 2 * self.padding)
            h = min(height - y, h + 2 * self.padding)

            cropped_frame = frame[y:y+h, x:x+w, :]
            if cropped_frame.size == 0:
                continue

            if self.out_writer is None:
                out_height, out_width = cropped_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.out_writer = cv2.VideoWriter(self.output_video_path, fourcc, fps, (out_width, out_height))

            self.out_writer.write(cropped_frame)

            transformed_frame = self.transform(cropped_frame)
            frames.append(transformed_frame)

            if len(frames) == self.count:
                break

        cap.release()
        if self.out_writer is not None:
            self.out_writer.release()

        if not frames:
            raise ValueError(f'No valid frames found in video: {self.video_path}')

        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames.unsqueeze(0)

class VideoService:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.transforms = self._get_transforms()
        self.sm = nn.Softmax(dim=1)

    def _load_model(self, model_path):
        model = Model(2).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model

    def _get_transforms(self):
        return T.Compose([
            T.ToPILImage(),
            T.Resize((112, 112)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, video_path, sequence_length=100):
        try:
            video_dataset = VideoDataset(
                video_path,
                sequence_length=sequence_length,
                transform=self.transforms,
                output_video_path='cropped_output_video.mp4'
            )

            frames = video_dataset[0]
            fmap, logits = self.model(frames.to(self.device))
            logits = self.sm(logits)
            _, prediction = torch.max(logits, 1)
            confidence = logits[:, int(prediction.item())].item() * 100

            return {
                "is_real": bool(prediction.item()),
                "confidence": float(confidence),
                "prediction": "Real" if prediction.item() == 1 else "Fake"
            }
        except Exception as e:
            raise Exception(f"Error processing video: {str(e)}") 
