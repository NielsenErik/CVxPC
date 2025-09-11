# Crack Detection

## Introduction

## Data

https://docs.ultralytics.com/it/datasets/segment/crack-seg/

How to download CrackSeg dataset:
```python
from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="API_KEY"
)

result = CLIENT.infer(your_image.jpg, model_id="crack-bphdr/2")
```

https://www.kaggle.com/datasets/lakshaymiddha/crack-segmentation-dataset
Or from kaggle hub:
```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("lakshaymiddha/crack-segmentation-dataset")

print("Path to dataset files:", path)
```

Fine-tuning data (?)
Property data of images on cracks

## Satrting point
https://www.kaggle.com/code/muzammalnawaz/crack-segmentation