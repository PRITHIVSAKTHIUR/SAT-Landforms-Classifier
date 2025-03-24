![3.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/6rEQLpCqxSb1JECmzNCKz.png)

# **SAT-Landforms-Classifier**

> **SAT-Landforms-Classifier** is an image classification vision-language encoder model fine-tuned from **google/siglip2-base-patch16-224** for a single-label classification task. It is designed to classify satellite images into different landform categories using the **SiglipForImageClassification** architecture.  

```py
Accuracy: 0.9863
F1 Score: 0.9858

Classification Report:
                       precision    recall  f1-score   support

          Annual Crop     0.9866    0.9810    0.9838      3000
               Forest     0.9927    0.9957    0.9942      3000
Herbaceous Vegetation     0.9697    0.9800    0.9748      3000
              Highway     0.9826    0.9928    0.9877      2500
           Industrial     0.9964    0.9916    0.9940      2500
              Pasture     0.9882    0.9610    0.9744      2000
       Permanent Crop     0.9690    0.9760    0.9725      2500
          Residential     0.9940    0.9970    0.9955      3000
                River     0.9864    0.9872    0.9868      2500
             Sea Lake     0.9963    0.9923    0.9943      3000

             accuracy                         0.9863     27000
            macro avg     0.9862    0.9855    0.9858     27000
         weighted avg     0.9863    0.9863    0.9863     27000
```

![download.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/Vt95rKi7pcP_6mV9fkIkS.png)

The model categorizes images into ten classes:
- **Class 0:** "Annual Crop"
- **Class 1:** "Forest"
- **Class 2:** "Herbaceous Vegetation"
- **Class 3:** "Highway"
- **Class 4:** "Industrial"
- **Class 5:** "Pasture"
- **Class 6:** "Permanent Crop"
- **Class 7:** "Residential"
- **Class 8:** "River"
- **Class 9:** "Sea Lake"

# **Run with Transformers🤗**

```python
!pip install -q transformers torch pillow gradio
```

```python
import gradio as gr
from transformers import AutoImageProcessor
from transformers import SiglipForImageClassification
from transformers.image_utils import load_image
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/SAT-Landforms-Classifier"
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

def landform_classification(image):
    """Predicts landform category for a satellite image."""
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
    
    labels = {
        "0": "Annual Crop", "1": "Forest", "2": "Herbaceous Vegetation", "3": "Highway", "4": "Industrial", 
        "5": "Pasture", "6": "Permanent Crop", "7": "Residential", "8": "River", "9": "Sea Lake"
    }
    predictions = {labels[str(i)]: round(probs[i], 3) for i in range(len(probs))}
    
    return predictions

# Create Gradio interface
iface = gr.Interface(
    fn=landform_classification,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(label="Prediction Scores"),
    title="SAT Landforms Classification",
    description="Upload a satellite image to classify its landform type."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
```

# **Intended Use:**  

The **SAT-Landforms-Classifier** model is designed to classify satellite images into various landform types. Potential use cases include:  

- **Land Use Monitoring:** Identifying different land use patterns from satellite imagery.
- **Environmental Studies:** Supporting researchers in tracking changes in vegetation and water bodies.
- **Urban Planning:** Assisting planners in analyzing residential, industrial, and infrastructure distributions.
- **Agricultural Analysis:** Helping assess crop distribution and pastureland areas.
- **Disaster Management:** Providing insights into land coverage for emergency response and recovery planning.
