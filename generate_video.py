import torchvision.transforms as T
from PIL import Image
import time
from memento.models import RecognitionModel, DetectionModel
import torch
import cv2
import os
import torch.nn.functional as F

def process_video(model, detector, video_path, ref_image_folder=None, output_path="output_video.mp4", 
                 det_conf_thresh=0.85, rec_conf_thresh=0.5, device='cuda', output_fps=30, img_size=256):
    """
    Process a video for face detection and recognition, annotating with bounding boxes and detailed labels.
    Handles multiple faces per frame, displays processing live with detection and recognition confidences,
    and saves output video. Reference images are optional.
    
    Args:
        model: Face recognition model
        detector: Face detection model
        video_path (str): Path to input video
        ref_image_folder (str, optional): Path to folder with reference face images
        output_path (str): Path to save output video
        det_conf_thresh (float): Detection confidence threshold
        rec_conf_thresh (float): Recognition confidence threshold
        device (str): Device to run models ('cuda' or 'cpu')
        output_fps (int): Frames per second for output video
    """
    model.eval().to(device)
    detector.eval()

    
    known_faces = {} 
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if ref_image_folder:
        print(f"[INFO] Loading reference images from {ref_image_folder}")
        try:
            for filename in os.listdir(ref_image_folder):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    name = os.path.splitext(filename)[0]
                    img_path = os.path.join(ref_image_folder, filename)
                    try:
                        img = Image.open(img_path).convert('RGB')
                        img_tensor = transform(img).unsqueeze(0).to(device)
                        with torch.no_grad():
                            embedding = model(img_tensor).cpu().squeeze(0)
                        known_faces[name] = embedding
                        print(f"[INFO] Loaded reference face for: {name}")
                    except Exception as e:
                        print(f"[WARNING] Failed to load {filename}: {e}")
        except Exception as e:
            print(f"[WARNING] Could not access reference folder: {e}")

    if not known_faces:
        print("[INFO] No reference images loaded. Faces will be labeled as 'Unknown'.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) if output_fps is None else output_fps

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (orig_w, orig_h))

    print(f"[INFO] Processing video: {video_path}")
    frame_count = 0

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        orig = frame.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.
        frame_tensor = frame_tensor.to(device)
        frame_resized = F.interpolate(frame_tensor, size=(256, 256), mode='bilinear', align_corners=False)

        with torch.inference_mode():
            boxes = detector.predict_tensor(frame_resized)[0]

        scale_x = orig_w / 256
        scale_y = orig_h / 256
        detected_faces = []

        for (cx, cy, w, h, conf) in boxes:
            if conf < det_conf_thresh:
                continue
            x1 = int((cx - w / 2) * scale_x)
            y1 = int((cy - h / 2) * scale_y)
            x2 = int((cx + w / 2) * scale_x)
            y2 = int((cy + h / 2) * scale_y)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_w - 1, x2), min(orig_h - 1, y2)

            if x2 <= x1 or y2 <= y1:
                print(f"[WARNING] Invalid bounding box at frame {frame_count}: ({x1}, {y1}, {x2}, {y2})")
                continue

            face_img = orig[y1:y2, x1:x2]
            if face_img.size == 0:
                print(f"[WARNING] Empty face image at frame {frame_count}: ({x1}, {y1}, {x2}, {y2})")
                continue

            try:
                face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                face_tensor = transform(face_pil).unsqueeze(0).to(device)
            except Exception as e:
                print(f"[WARNING] Failed to process face image at frame {frame_count}: {e}")
                continue

            name = "Unknown"
            rec_conf = "N/A"
            if known_faces:  
                with torch.no_grad():
                    embedding = model(face_tensor).cpu().squeeze(0)
                best_sim = 0.0
                for known_name, known_emb in known_faces.items():
                    sim = F.cosine_similarity(embedding.unsqueeze(0), known_emb.unsqueeze(0)).item()
                    if sim > rec_conf_thresh and sim > best_sim:
                        best_sim = sim
                        name = known_name
                        rec_conf = f"{sim:.2f}"

            detected_faces.append(((x1, y1, x2, y2), name, conf, rec_conf))

        header_text = f"Frame: {frame_count} | Faces Detected: {len(detected_faces)} | Status: Running"
        cv2.rectangle(frame, (0, 0), (orig_w, 40), (0, 0, 0), -1) 
        cv2.putText(frame, header_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        for i, ((x1, y1, x2, y2), name, det_conf, rec_conf) in enumerate(detected_faces):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Name: {name}, Det: {det_conf:.2f}, Rec: {rec_conf}"
            label_y = y1 - 35 - (i * 30) if y1 - 35 > 40 else y2 + 30 + (i * 30)
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, label_y - text_h - 5), (x1 + text_w, label_y + 5), (0, 0, 0), -1)
            cv2.putText(frame, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        inf_time = (time.time() - start_time) * 1000
        cv2.putText(frame, f"Inference: {inf_time:.1f} ms", (10, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow("Live Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Processing interrupted by user")
            break

        out.write(frame)
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"[INFO] Processed {frame_count} frames, {len(detected_faces)} faces detected in this frame")

    print(f"[INFO] Processed {frame_count} frames in total")
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Output video saved to: {output_path}")



if __name__ == "__main__":

    # Hyperparameters
    DEVICE = 'cuda'
    RECOGNITION_WEIGHTS = "C:/VSCODE/FaceTrainingV3/memento/approved_weights/best_recognition.pth"
    DETECTION_WEIGHTS = "C:/VSCODE/FaceTrainingV3/Detection_Logs/Run_2/epoch_44.pt"
    EMBEDDING_DIM = 1024
    DETECTION_CONF_THRESH = 0.8
    RECOGNITION_CONF_THRESH = 0.5
    REF_IMAGE_FOLDER = "test_media/video_generator_media"
    VIDEO_PATH = "test_media/saymyname30sec.mp4"
    OUTPUT_VIDEO_PATH = "test_media/output_saymyname.mp4"
    OUTPUT_FPS = 24
    IMG_SIZE = 256


    
    det = DetectionModel(weights=DETECTION_WEIGHTS, device=DEVICE)
    print(f"Loaded Detection Weights: {DETECTION_WEIGHTS}")

    rec = RecognitionModel(emb_dim=EMBEDDING_DIM, weights=RECOGNITION_WEIGHTS, device=DEVICE)
    print(f"Loaded Recognition Weights: {RECOGNITION_WEIGHTS}")


    print(f"Test Device: {DEVICE}")


    process_video(
        model=rec,
        detector=det,
        video_path=VIDEO_PATH,
        ref_image_folder=REF_IMAGE_FOLDER,
        output_path=OUTPUT_VIDEO_PATH,
        det_conf_thresh=DETECTION_CONF_THRESH,
        rec_conf_thresh=RECOGNITION_CONF_THRESH,
        device=DEVICE,
        output_fps=OUTPUT_FPS,
        img_size=IMG_SIZE
    )





