from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests
import gradio as gr
import ast

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



gr.Interface(vitinf, inputs, outputs, title=title, description=description, article=article).launch(debug=True)