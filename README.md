Plant Disease Classification
A deep learning system that identifies plant diseases from leaf images. The model predicts the disease type regardless of the host plant species,
classifying images into one of 40 disease categories.

What it does
1.Takes a plant leaf image as input
2.Predicts the disease type (e.g. "late blight", "rust", "powdery mildew")
3.Returns confidence score with the prediction
4.Returns "Uncertain" if confidence is too low (possibly healthy plant)
5.Works for 40 different plant diseases across multiple plant species

Results
  LR:            1.74e-05
  Train Loss:    0.5555
  Val Loss:      0.4687
  mAP:           0.8645   
  F1 Macro:      0.7889  
  F1 Weighted:   0.7876 
  Precision:     0.8085
  Recall:        0.7900
  Mean cls acc:  0.7900
  Worst cls acc: 0.2000 (class 31)
  Worst 3 AP:    [(33, '0.40'), (31, '0.44'), (1, '0.59')]

Links

 API: https://arpigmm-plant-disease-api.hf.space
 Swagger Docs: https://arpigmm-plant-disease-api.hf.space/docs
 Model Weights: https://huggingface.co/arpigmm/plant-disease-model


Project Structure
├── Disease(3).ipynb  
├── app_py.py                        
├── requirements(1).txt            
├── Dockerfile                   
└── report.pdf                  

Model Architecture

Backbone: ResNet50 pretrained on ImageNet
Head: Dropout(0.4) → Linear(2048→512) → ReLU → Dropout(0.3) → Linear(512→40)
Classes: 40 plant diseases
Input: 224×224 RGB image

Training

Strategy: Two-stage (frozen backbone → full fine-tune)
Loss: Focal Loss (gamma=2.0) for class imbalance
Sampler: WeightedRandomSampler with sqrt balancing
Augmentation: Conservative — RandomResizedCrop, HorizontalFlip, Rotation(15°), mild ColorJitter, MixUp(alpha=0.2)
Scheduler: CosineAnnealingLR
Dataset: ~8,000 images, 40 classes, 32:1 imbalance ratio

API Usage
Send a POST request to /predict with a plant leaf image:
"Response:
json{
  "disease": "late blight",
  "confidence": 0.923,
  "reliable": true
}
