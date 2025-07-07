import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import models
import torch.onnx 
import cv2
import time
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import importlib.resources as resources




class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )

class DetectionModel(nn.Module):
    def __init__(self, weights=None, device=None, return_decoded=False):
        super().__init__()
        self.return_decoded = return_decoded
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = models.efficientnet_b0(weights=None).features.to(self.device)
        self.neck = nn.Sequential(
            ConvBNAct(1280, 800, 3, padding=1),
            ConvBNAct(800, 400, 3, padding=1),
            ConvBNAct(400, 240, 3, padding=1)
        ).to(self.device)
        self.bbox_head = nn.Sequential(
            nn.Conv2d(240, 128, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 4, 1)
        ).to(self.device)
        self.obj_head = nn.Sequential(
            nn.Conv2d(240, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 1, 1)
        ).to(self.device)
        if weights:
            self.load_weights(weights)
        self.to(self.device)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        bbox = self.bbox_head(x)
        obj = self.obj_head(x)
        x = torch.cat([bbox, obj], dim=1)
        if self.return_decoded:
            x = self.decode_predictions(x)
        return x

    def decode_predictions(self, preds, conf_thresh=0.05):
        B, _, H, W = preds.shape
        preds = preds.permute(0, 2, 3, 1).contiguous()
        stride_x = 256 / W
        stride_y = 256 / H
        all_boxes = []
        for b in range(B):
            pred = preds[b]
            conf = torch.sigmoid(pred[..., 4])
            conf_mask = conf > conf_thresh
            boxes = []
            ys, xs = conf_mask.nonzero(as_tuple=True)
            for y, x in zip(ys, xs):
                px, py, pw, ph = pred[y, x, :4]
                pconf = conf[y, x].item()
                cx = (x + px.item()) * stride_x
                cy = (y + py.item()) * stride_y
                w = pw.item() * stride_x
                h = ph.item() * stride_y
                boxes.append([cx, cy, w, h, pconf])
            all_boxes.append(boxes)
        return all_boxes

    def predict_tensor(self, img_tensor):
        self.eval()
        with torch.inference_mode():
            img_tensor = img_tensor.to(self.device)
            preds = self.forward(img_tensor)
            decoded = self.decode_predictions(preds)
        return decoded
    
    def predict_pil_img(self, img):
        self.eval()
        transform = T.Compose([
            T.Resize((256,256)),
            T.ToTensor()
        ])
        img = transform(img)
        with torch.inference_mode():
            pred = self.forward(img)
            decoded = self.decode_predictions(pred)
            return decoded

    def load_weights(self, pth):
        state_dict = torch.load(pth, map_location=self.device)
        self.load_state_dict(state_dict)

    def save_weights(self, save_pth):
        torch.save(self.state_dict(), save_pth)

    def live_test(self, conf_thresh=0.5, frame_skip=0):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return
        import torch.nn.functional as F
        prev_time = time.time()
        frame_count = 0
        last_boxes = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                orig_h, orig_w = frame.shape[:2]
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_tensor = torch.from_numpy(frame_rgb).permute(2,0,1).unsqueeze(0).float() / 255.
                frame_tensor = frame_tensor.to(self.device)
                frame_resized = F.interpolate(frame_tensor, size=(256,256), mode='bilinear', align_corners=False)
                if frame_skip == 0 or frame_count % frame_skip == 0:
                    start = time.time()
                    last_boxes = self.predict_tensor(frame_resized)[0]
                    inference_time = (time.time() - start)*1000
                    fps = 1/(time.time() - prev_time)
                    prev_time = time.time()
                else:
                    inference_time = 0
                    fps = None
                frame_count += 1
                scale_x = orig_w / 256
                scale_y = orig_h / 256
                for (cx, cy, w, h, conf) in last_boxes:
                    if conf < conf_thresh:
                        continue
                    x1 = int((cx - w / 2) * scale_x)
                    y1 = int((cy - h / 2) * scale_y)
                    x2 = int((cx + w / 2) * scale_x)
                    y2 = int((cy + h / 2) * scale_y)
                    x1 = max(0, min(orig_w - 1, x1))
                    y1 = max(0, min(orig_h - 1, y1))
                    x2 = max(0, min(orig_w - 1, x2))
                    y2 = max(0, min(orig_h - 1, y2))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
                if inference_time > 0:
                    cv2.putText(frame, f"Inference: {inference_time:.1f} ms", (orig_w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
                    cv2.putText(frame, f"Device: {self.device}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
                cv2.imshow("Live Face Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def face_and_plot(self, img, conf_thresh=0.5):
        if isinstance(img, np.ndarray):
            if img.dtype == np.uint8:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
        transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor()
        ])
        img_tensor = transform(img).unsqueeze(0).to(self.device)
        self.eval()
        with torch.inference_mode():
            start = time.time()
            preds = self.forward(img_tensor)
            boxes = self.decode_predictions(preds, conf_thresh=conf_thresh)[0]
            end = time.time()
        img_np = np.array(img)
        orig_h, orig_w = img_np.shape[:2]
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(img_np)
        for (cx, cy, w, h, conf) in boxes:
            if conf < conf_thresh:
                continue
            x1 = (cx - w / 2) * orig_w / 256
            y1 = (cy - h / 2) * orig_h / 256
            width = w * orig_w / 256
            height = h * orig_h / 256
            rect = plt.Rectangle((x1, y1), width, height, edgecolor='lime', facecolor='none', linewidth=2)
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, f"{conf:.2f}", color='red', fontsize=10, weight='bold')
        ax.axis('off')
        plt.show()


class ArcFace(nn.Module):
    def __init__(self, emb_dim, num_classes, s=30.0, m=0.50):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, emb_dim))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m
    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, dim=1)
        weights = F.normalize(self.weight, dim=1)
        cosine = F.linear(embeddings, weights)
        cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cosine)
        target_logits = torch.cos(theta + self.m)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        output = self.s * (one_hot * target_logits + (1 - one_hot) * cosine)
        return output

class RecognitionModel(nn.Module):
    def __init__(self, emb_dim=384, num_classes=None, s=30.0, m=0.50, weights="DEFAULT", device="cpu"):
        super().__init__()
        self.device = device
        self.backbone = models.efficientnet_b0(weights="DEFAULT").features.to(self.device)
        self.neck = ConvBNAct(1280, emb_dim, kernel_size=1).to(self.device)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(emb_dim),
            nn.Dropout(0.5),
        ).to(self.device)
        self.arcface = ArcFace(emb_dim, num_classes, s=s, m=m).to(self.device) if num_classes is not None else None
        if weights != "DEFAULT" and weights != None:
            self.load_weights_partial(weights)
        else:
            if weights != None:
                self.load_weights_partial("approved_weights/best_recognition.pth")
        self.to(self.device)

    def forward(self, x, labels=None):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        x = F.normalize(x, dim=1)
        if labels is not None and self.arcface is not None:
            logits = self.arcface(x, labels)
            return logits
        return x

    def generate_emb(self, img, transform="DEFAULT", device=None):
        if device is None:
            device = self.device
        if transform == "DEFAULT":
            transform = T.Compose([
                T.Resize((256, 256)),
                T.ToTensor()
            ])
        self.eval()
        with torch.inference_mode():
            x = transform(img).unsqueeze(0).to(device)
            emb = self.forward(x).cpu()
        return emb.squeeze(0)

    def load_weights_partial(self, pth):
        if pth == "approved_weights/best_recognition.pth":
            with resources.open_binary("Memento", "approved_weights/best_recognition.pth") as f:
                checkpoint = torch.load(f, map_location=self.device)
        else:
            checkpoint = torch.load(pth, map_location=self.device)
        own_state = self.state_dict()
        for name, param in checkpoint.items():
            if name in own_state:
                if own_state[name].size() == param.size():
                    own_state[name].copy_(param)
                else:
                    print(f"Skipping loading parameter: {name}, size mismatch {own_state[name].size()} vs {param.size()}")
            else:
                print(f"Ignoring unexpected parameter: {name}")
    def save_weights(self, save_pth):
        torch.save(self.state_dict(), save_pth)
    
    def compare_faces(self, img1, img2, transform="DEFAULT", device=None):
        def to_pil(img):
            if isinstance(img, np.ndarray):
                return Image.fromarray(img)
            return img

        img1 = to_pil(img1)
        img2 = to_pil(img2)

        if transform == "DEFAULT":
            transform = T.Compose([
                T.Resize((256, 256)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
            ])

        emb1 = self.generate_emb(img1, transform=transform, device=device)
        emb2 = self.generate_emb(img2, transform=transform, device=device)

        emb1 = F.normalize(emb1, dim=0)  # Extra L2 normalization for safety
        emb2 = F.normalize(emb2, dim=0)

        similarity = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
        return similarity


if __name__ == "__main__":

    weights = "approved_weights/best_detection.pth"


    det = DetectionModel(weights=weights)
    print("Starting Test")
    det.live_test()