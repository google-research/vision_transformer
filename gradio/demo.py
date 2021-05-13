from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests
import gradio as gr
import ast
import torch

torch.hub.download_url_to_file('http://images.cocodataset.org/val2017/000000039769.jpg', 'cat.jpg')
torch.hub.download_url_to_file('https://cdn.pixabay.com/photo/2016/03/27/22/22/fox-1284512_1280.jpg', 'fox.jpg')


feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

def vitinf(image):
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    dout = ast.literal_eval("{'Predicted class' : "+ f'"{model.config.id2label[predicted_class_idx]}"'+"}")
    return dout['Predicted class']

inputs = gr.inputs.Image(type='pil', label="Original Image")


outputs = gr.outputs.Label(type="auto", label="Predicted class")

title = "VIT"
description = "demo for Google VIT. To use it, simply upload your image, or click one of the examples to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2010.11929'>An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale</a> | <a href='https://github.com/google-research/vision_transformer'>Github Repo</a></p>"

examples = [
    ['cat.jpg'], 
    ['fox.jpg']
]


gr.Interface(vitinf, inputs, outputs, title=title, description=description, article=article, examples=examples).launch()