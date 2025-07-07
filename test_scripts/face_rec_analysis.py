

import torch
import torchvision.transforms as T
import torch.nn.functional as F
from Memento.models import RecognitionModel
import time
from PIL import Image

def rec_analysis(model, device, img_1, img_2, img_size=256):

    model.eval()

    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor()
    ])

    img_1 = transform(Image.open(img_1)).to(device)
    img_2 = transform(Image.open(img_2)).to(device)

    with torch.inference_mode():

        model.to(device)

        start_img_1 = time.time()
        emb_1 = model(img_1)
        end_img_1 = time.time()

        start_img_2 = time.time()
        emb_2 = model(img_2)
        end_img_2 = time.time()

        start_cos_sim = time.time()
        sim = F.cosine_similarity(emb_1, emb_2)
        end_cos_sim = time.time()

        print(f"Total Processing Time: {(end_cos_sim-start_img_1)*1000:.4f}s")
        print(f"Image 1 Inference Time: {(end_img_1-start_img_1)*1000:.4f}s")
        print(f"Image 2 Inference Time: {(end_img_2-start_img_2)*1000:.4f}s")
        print(f"Cosine Similarity Time: {(end_cos_sim-start_cos_sim)*1000:4f}s")
        print(f"Similarity Score: {sim*100:2f}%")


if __name__ == "__main__":

    DEVICE = 'cuda'
    REC_WEIGHTS = ""
    
    image_1 = ""
    image_2 = ""    
    
    rec = RecognitionModel(weights=REC_WEIGHTS, device=DEVICE)

    rec_analysis(
        model=rec,
        device=DEVICE,
        img_1=image_1,
        img_2=image_2,
        img_size=256    
    )





