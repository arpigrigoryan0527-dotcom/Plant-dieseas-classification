 Plant Disease Classification

A deep learning system that identifies plant diseases from leaf images. The model predicts the disease type regardless of the host plant species, classifying images into one of 40 disease categories.

 What it does

- Takes a plant leaf image as input
- Predicts the disease type (e.g. "late blight", "rust", "powdery mildew")
- Returns confidence score with the prediction
- Returns "Uncertain" if confidence is too low (possibly healthy plant)
- Works for 40 different plant diseases across multiple plant species

 Results

| Metric | Value |
|--------|-------|
| mAP | 0.8645 |
| F1 Macro | 0.7889 |
| Precision | 0.8085 |
| Recall | 0.7900 |

Links

API: https://arpigmm-plant-disease-api.hf.space
Swagger Docs: https://arpigmm-plant-disease-api.hf.space/docs
Model Weights: https://huggingface.co/arpigmm/plant-disease-model

 Project Structure

Plant
  ├── Disease_Classification.ipynb  
  ├── app.py                        
  ├── requirements.txt              
  ├── Dockerfile                    
  └── report.pdf                    


Model Architecture

Backbone: ResNet50 pretrained on ImageNet
Head: Dropout(0.4) → Linear(2048→512) → ReLU → Dropout(0.3) → Linear(512→40)
Classes: 40 plant diseases
Input: 224×224 RGB image

 Training

Strategy: Two-stage (frozen backbone  full fine-tune)
Loss: Focal Loss (gamma=2.0) for class imbalance
Sampler: WeightedRandomSampler with sqrt balancing
Augmentation: Conservative  RandomResizedCrop, HorizontalFlip, Rotation(15°), mild ColorJitter, MixUp(alpha=0.2)
Scheduler: CosineAnnealingLR
Dataset: ~8,000 images, 40 classes, 32:1 imbalance ratio

API Usage

Send a POST request to `/predict` with a plant leaf image:

```bash
curl -X POST "https://arpigmm-plant-disease-api.hf.space/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_image.jpg"
```

Response:
```json
{
  "disease": "late blight",
  "confidence": 0.923,
  "reliable": true
}
```

