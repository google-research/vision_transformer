from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests
import gradio as gr

feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

def vitinf(image):
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    # model predicts one of the 1000 ImageNet classes
    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]

inputs = gr.inputs.Image(type='pil', label="Original Image")

outputs = gr.outputs.Textbox(label="Output Text")

title = "VIT"
description = "demo for Google VIT. To use it, simply upload your image, or click one of the examples to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2010.11929'>An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale</a> | <a href='https://github.com/google-research/vision_transformer'>Github Repo</a></p>"



gr.Interface(vitinf, inputs, outputs, title=title, description=description, article=article).launch(debug=True)