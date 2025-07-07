import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
import datetime
import time 
import torchvision.models as models
import torchvision.transforms as T
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset

class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )

class DetectionModel(nn.Module):
    def __init__(self, weights=None, device=None, return_decoded=False, img_size=256):
        super().__init__()
        self.return_decoded = return_decoded
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = img_size
        
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
        B, C, H, W = preds.shape
        preds = preds.permute(0, 2, 3, 1).contiguous()
        
        stride_from_model_output = self.img_size / H
        
        all_boxes = []
        for b in range(B):
            pred = preds[b]
            conf = torch.sigmoid(pred[..., 4])
            conf_mask = conf > conf_thresh
            
            boxes = []
            ys, xs = conf_mask.nonzero(as_tuple=True) 
            
            for y_grid, x_grid in zip(ys, xs):
                px, py, pw, ph = pred[y_grid, x_grid, :4] 
                pconf = conf[y_grid, x_grid].item()

                cx = (x_grid.item() + px.item()) * stride_from_model_output 
                cy = (y_grid.item() + py.item()) * stride_from_model_output
                
                w = pw.item() * stride_from_model_output 
                h = ph.item() * stride_from_model_output 
                
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
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor()
        ])
        img = transform(img)
        with torch.inference_mode():
            pred = self.forward(img.unsqueeze(0))
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
            print("Error: Could not open webcam.")
            return
        
        import torch.nn.functional as F
        prev_time = time.time()
        frame_count = 0
        last_boxes = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame from webcam.")
                    break
                orig_h, orig_w = frame.shape[:2]
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_tensor = torch.from_numpy(frame_rgb).permute(2,0,1).unsqueeze(0).float() / 255.
                frame_tensor = frame_tensor * 2 - 1
                
                frame_tensor = frame_tensor.to(self.device)
                
                frame_resized = F.interpolate(frame_tensor, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
                
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
                
                scale_x = orig_w / self.img_size
                scale_y = orig_h / self.img_size
                
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
            img_pil = Image.fromarray(img)
        else:
            img_pil = img

        transform = T.Compose([
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
        img_tensor = transform(img_pil).unsqueeze(0).to(self.device)
        
        self.eval()
        with torch.inference_mode():
            start = time.time()
            preds = self.forward(img_tensor)
            boxes = self.decode_predictions(preds, conf_thresh=conf_thresh)[0]
            end = time.time()
        
        img_np_plot = np.array(img_pil)
        orig_h, orig_w = img_np_plot.shape[:2]
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(img_np_plot)
        
        for (cx, cy, w, h, conf) in boxes:
            if conf < conf_thresh:
                continue
            
            x1 = (cx - w / 2) * orig_w / self.img_size
            y1 = (cy - h / 2) * orig_h / self.img_size
            width = w * orig_w / self.img_size
            height = h * orig_h / self.img_size
            
            rect = plt.Rectangle((x1, y1), width, height, edgecolor='lime', facecolor='none', linewidth=2)
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, f"{conf:.2f}", color='red', fontsize=10, weight='bold')
        
        ax.set_title(f"Detection Result (Inference Time: {(end-start)*1000:.2f} ms)")
        ax.axis('off')
        plt.show()


def compute_batch_iou(preds, targets):
    preds_reshaped = preds.permute(0, 2, 3, 1).reshape(-1, 4)
    targets_reshaped = targets.permute(0, 2, 3, 1).reshape(-1, 4)

    pred_x1 = preds_reshaped[:, 0] - preds_reshaped[:, 2] / 2
    pred_y1 = preds_reshaped[:, 1] - preds_reshaped[:, 3] / 2
    pred_x2 = preds_reshaped[:, 0] + preds_reshaped[:, 2] / 2
    pred_y2 = preds_reshaped[:, 1] + preds_reshaped[:, 3] / 2

    targ_x1 = targets_reshaped[:, 0] - targets_reshaped[:, 2] / 2
    targ_y1 = targets_reshaped[:, 1] - targets_reshaped[:, 3] / 2
    targ_x2 = targets_reshaped[:, 0] + targets_reshaped[:, 2] / 2
    targ_y2 = targets_reshaped[:, 1] + targets_reshaped[:, 3] / 2

    inter_x1 = torch.max(pred_x1, targ_x1)
    inter_y1 = torch.max(pred_y1, targ_y1)
    inter_x2 = torch.min(pred_x2, targ_x2)
    inter_y2 = torch.min(pred_y2, targ_y2)

    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    
    pred_area = (pred_x2 - pred_x1).clamp(0) * (pred_y2 - pred_y1).clamp(0)
    targ_area = (targ_x2 - targ_x1).clamp(0) * (targ_y2 - targ_y1).clamp(0)
    union_area = pred_area + targ_area - inter_area + 1e-6
    iou = inter_area / union_area
    return iou.mean()

class DetectionDataset(Dataset):
    def __init__(self, img_folder, lbl_folder, transform=None, grid_size=8, img_size=256):
        self.img_size = img_size
        self.img_pths = sorted([os.path.join(img_folder, f) for f in os.listdir(img_folder)])
        self.lbl_pths = sorted([os.path.join(lbl_folder, f) for f in os.listdir(lbl_folder)])
        self.transform = transform
        self.grid_size = grid_size
        self.stride = self.img_size / self.grid_size 
        
        self.lbls = []
        for pth in self.lbl_pths:
            _lbls = []
            with open(pth, 'r') as f:
                for line in f:
                    lbl = line.strip().split()
                    if len(lbl) >= 5:
                        _lbls.append([float(x) for x in lbl[1:5]])
            self.lbls.append(_lbls)

    def __len__(self):
        return len(self.img_pths)

    def __getitem__(self, idx):
        img = Image.open(self.img_pths[idx]).convert("RGB")
        labels = self.lbls[idx]
        
        if self.transform:
            img = self.transform(img)

        bbox_grid = torch.zeros((self.grid_size, self.grid_size, 4), dtype=torch.float32)
        obj_grid = torch.zeros((self.grid_size, self.grid_size, 1), dtype=torch.float32)
        
        for lbl in labels:
            cx_norm, cy_norm, w_norm, h_norm = lbl
            
            cx_pixel = cx_norm * self.img_size
            cy_pixel = cy_norm * self.img_size
            w_pixel = w_norm * self.img_size
            h_pixel = h_norm * self.img_size

            grid_x = int(cx_pixel / self.stride)
            grid_y = int(cy_pixel / self.stride)
            
            if grid_x < self.grid_size and grid_y < self.grid_size:
                offset_x = (cx_pixel % self.stride) / self.stride
                offset_y = (cy_pixel % self.stride) / self.stride
                
                w_scaled = w_pixel / self.stride
                h_scaled = h_pixel / self.stride
                
                bbox_grid[grid_y, grid_x] = torch.tensor([offset_x, offset_y, w_scaled, h_scaled])
                obj_grid[grid_y, grid_x] = 1.0

        return img, bbox_grid, obj_grid

class DetectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bbox_loss = nn.SmoothL1Loss(reduction='sum')
        self.obj_loss = nn.BCEWithLogitsLoss(reduction='sum')

    def forward(self, preds, bbox_targets, obj_targets):
        bbox_pred = preds[:, :4, :, :]
        obj_pred = preds[:, 4:5, :, :]
        
        bbox_targets = bbox_targets.permute(0, 3, 1, 2)
        obj_targets = obj_targets.permute(0, 3, 1, 2)

        bbox_loss = self.bbox_loss(bbox_pred * obj_targets, bbox_targets * obj_targets)
        
        obj_loss = self.obj_loss(obj_pred, obj_targets)
        
        iou = compute_batch_iou(bbox_pred, bbox_targets) 
        
        valid_cells = obj_targets.sum()
        num_cells = bbox_pred.shape[2] * bbox_pred.shape[3]
        
        if valid_cells == 0:
            return obj_loss / (bbox_pred.shape[0] * num_cells)
        
        return (bbox_loss / valid_cells + obj_loss / (bbox_pred.shape[0] * num_cells)) - 0.1 * iou

def train_detection(model, train_dl, loss_fn, optimizer, save_folder, val_dl=None, scheduler=None, epochs=10, device='cpu'):
    model = model.to(device)
    log_path = os.path.join(save_folder, "training_log.json")
    os.makedirs(save_folder, exist_ok=True)

    len_train_dl = len(train_dl)

    def format_time(seconds):
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    for epoch in range(epochs):
        model.train()
        train_total_loss = 0
        train_total_count = 0
        epoch_successful = True
        start_time_epoch = time.time()

        print(f"\nStarting Training for Epoch: {epoch + 1}")
        try:
            for i, (img, bbox_target, obj_target) in enumerate(train_dl):
                img = img.to(device)
                bbox_target = bbox_target.to(device)
                obj_target = obj_target.to(device)

                optimizer.zero_grad()
                pred = model(img)
                loss = loss_fn(pred, bbox_target, obj_target)

                train_total_loss += loss.item()
                train_total_count += 1

                loss.backward()
                optimizer.step()

                elapsed_time = time.time() - start_time_epoch
                avg_time_per_batch = elapsed_time / train_total_count
                
                batches_remaining = len_train_dl - (i + 1)
                eta_seconds = batches_remaining * avg_time_per_batch
                
                avg_loss = train_total_loss / train_total_count
                
                print(f"Epoch: {epoch+1} | Loss: {avg_loss:.4f} | Progress: {i+1}/{len_train_dl} | Elapsed: {format_time(elapsed_time)} | ETA: {format_time(eta_seconds)} |", end='\r')

            avg_train_loss = train_total_loss / train_total_count
            print(f"\nTrain Loss: {avg_train_loss:.4f}")

            val_metrics = {}
            if val_dl:
                model.eval()
                val_total_loss = 0
                val_total_count = 0
                correct_cells = 0
                total_cells = 0

                print(f"\nStarting Evaluation for Epoch: {epoch+1}")
                with torch.inference_mode():
                    for img, bbox_target, obj_target in val_dl:
                        img = img.to(device)
                        bbox_target = bbox_target.to(device)
                        obj_target = obj_target.to(device)

                        pred = model(img)
                        loss = loss_fn(pred, bbox_target, obj_target)

                        val_total_loss += loss.item() 
                        val_total_count += 1

                        bbox_pred = pred[:, :4, :, :]
                        
                        valid_obj_mask = (obj_target == 1).squeeze(-1).flatten()
                        
                        if valid_obj_mask.sum() > 0:
                            pred_for_iou = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)[valid_obj_mask]
                            target_for_iou = bbox_target.reshape(-1, 4)[valid_obj_mask]

                            iou_scores = compute_batch_iou_per_box(pred_for_iou, target_for_iou)
                            correct_cells += (iou_scores > 0.5).float().sum().item()
                            total_cells += valid_obj_mask.sum().item()

                avg_val_loss = val_total_loss / val_total_count
                accuracy = correct_cells / max(1, total_cells)
                val_metrics['val_loss'] = avg_val_loss
                val_metrics['accuracy'] = accuracy

                print(f"Validation Loss: {avg_val_loss:.4f} | Accuracy: {accuracy:.2%}")
                if scheduler:
                    scheduler.step(avg_val_loss)

            save_path = f"{save_folder}/epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

        except Exception as e:
            print(f"Error in epoch {epoch+1}: {e}")
            epoch_successful = False
            avg_train_loss = float('nan')
            val_metrics = {}

        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "save_folder": save_folder,
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": val_metrics.get("val_loss", None),
            "accuracy": val_metrics.get("accuracy", None),
            "epoch_successful": epoch_successful,
            "device": device,
            "batch_size": train_dl.batch_size
        }

        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                logs = json.load(f)
        else:
            logs = []

        logs.append(log_entry)
        with open(log_path, "w") as f:
            json.dump(logs, f, indent=4)

    final_path = os.path.join(save_folder, "fully_trained.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved to {final_path}")

def compute_batch_iou_per_box(preds, targets):
    pred_x1 = preds[:, 0] - preds[:, 2] / 2
    pred_y1 = preds[:, 1] - preds[:, 3] / 2
    pred_x2 = preds[:, 0] + preds[:, 2] / 2
    pred_y2 = preds[:, 1] + preds[:, 3] / 2

    targ_x1 = targets[:, 0] - targets[:, 2] / 2
    targ_y1 = targets[:, 1] - targets[:, 3] / 2
    targ_x2 = targets[:, 0] + targets[:, 2] / 2
    targ_y2 = targets[:, 1] + targets[:, 3] / 2

    inter_x1 = torch.max(pred_x1, targ_x1)
    inter_y1 = torch.max(pred_y1, targ_y1)
    inter_x2 = torch.min(pred_x2, targ_x2)
    inter_y2 = torch.min(pred_y2, targ_y2)

    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    
    pred_area = (pred_x2 - pred_x1).clamp(0) * (pred_y2 - pred_y1).clamp(0)
    targ_area = (targ_x2 - targ_x1).clamp(0) * (targ_y2 - targ_y1).clamp(0)
    union_area = pred_area + targ_area - inter_area + 1e-6
    iou = inter_area / union_area
    return iou


def detection_collate_fn(batch):
    images = []
    bbox_labels = []
    obj_labels = []

    for img, bbox, obj in batch:
        images.append(img)
        bbox_labels.append(bbox)
        obj_labels.append(obj)

    return torch.stack(images, dim=0), torch.stack(bbox_labels, dim=0), torch.stack(obj_labels, dim=0)

def plot_dataset_sample(dataset, index):
    img_tensor, bbox_grid, obj_grid = dataset[index]

    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 0.5 + 0.5)
    img_np = np.clip(img_np, 0, 1)

    img_h, img_w = img_np.shape[0], img_np.shape[1]

    decoded_boxes = []
    grid_size = dataset.grid_size
    stride = dataset.stride

    for gy in range(grid_size):
        for gx in range(grid_size):
            if obj_grid[gy, gx].item() == 1:
                offset_x, offset_y, w_scaled, h_scaled = bbox_grid[gy, gx].tolist()

                cx = (gx + offset_x) * stride
                cy = (gy + offset_y) * stride
                
                w_pixel = w_scaled * stride 
                h_pixel = h_scaled * stride
                
                x1 = cx - w_pixel / 2
                y1 = cy - h_pixel / 2
                
                decoded_boxes.append([x1, y1, w_pixel, h_pixel])

    plt.figure(figsize=(8, 8))
    plt.imshow(img_np)
    ax = plt.gca()

    for x1, y1, width, height in decoded_boxes:
        rect = plt.Rectangle((x1, y1), width, height,
                             fill=False, edgecolor='lime', linewidth=2)
        ax.add_patch(rect)
    
    plt.title(f"Dataset Sample - Index: {index} (IMG_SIZE: {img_w}x{img_h}, Grid: {grid_size}x{grid_size})")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    print("Creating Datasets.")

    TRAIN_IMAGE_PATH = "Detection_Train_Data/train/images"
    TRAIN_LABEL_PATH = "Detection_Train_Data/train/labels"
    VAL_IMAGE_PATH = "Detection_Train_Data/val/images"
    VAL_LABEL_PATH = "Detection_Train_Data/val/labels"


    SAVE_FOLDER = "Detection_Logs/Run_2"
    EPOCHS = 50
    LR = 1e-3
    
    IMG_SIZE = 256
    GRID_SIZE = IMG_SIZE // 32
    BATCH_SIZE = 32

    os.makedirs(SAVE_FOLDER, exist_ok=True)

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomAutocontrast(p=0.2), 
        transforms.RandomEqualize(p=0.2), 
        transforms.RandomSolarize(threshold=128, p=0.1), 
        transforms.RandomPosterize(bits=4, p=0.1),
        transforms.RandomAdjustSharpness(.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    train_data = DetectionDataset(TRAIN_IMAGE_PATH, TRAIN_LABEL_PATH, transform=train_transform, grid_size=GRID_SIZE, img_size=IMG_SIZE)

    val_data = DetectionDataset(VAL_IMAGE_PATH, VAL_LABEL_PATH, transform=val_transform, grid_size=GRID_SIZE, img_size=IMG_SIZE)

    print(f"Train dataset size: {len(train_data)}")
    print(f"Validation dataset size: {len(val_data)}")

    print("\nPlotting a sample from the training dataset (index 0):")
    if len(train_data) > 0:
        plot_dataset_sample(train_data, 1)
    else:
        print("Training dataset is empty, cannot plot sample.")
        
    if len(val_data) > 5:
        plot_dataset_sample(val_data, 69)
    elif len(val_data) > 0:
        plot_dataset_sample(val_data, 69)
    else:
        print("Validation dataset is empty, cannot plot sample.")

    train_dl = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=detection_collate_fn, num_workers=os.cpu_count()//2 or 1, pin_memory=True)
    val_dl = DataLoader(val_data, batch_size=BATCH_SIZE, collate_fn=detection_collate_fn, num_workers=os.cpu_count()//2 or 1, pin_memory=True) 
    
    model = DetectionModel(img_size=IMG_SIZE, device=DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10)
    loss_fn = DetectionLoss()

    
    
    train_detection(model, train_dl, loss_fn, optimizer, SAVE_FOLDER, scheduler=scheduler, val_dl=val_dl, epochs=EPOCHS, device=DEVICE)

    print("\nTraining complete. Testing with a sample image:")
    if len(val_data) > 0:
        test_image_path = val_data.img_pths[0]
        test_img = Image.open(test_image_path).convert("RGB")
        model.face_and_plot(test_img, conf_thresh=0.5)
    else:
        print("No validation data to test with face_and_plot.")


# $env:KMP_DUPLICATE_LIB_OK="TRUE"