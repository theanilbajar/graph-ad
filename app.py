import streamlit as st
from PIL import Image

# load classification model
from torchvision import models, transforms
import torch

def predict(image_path, classes):
    image = Image.open(image_path)

    resnet = models.resnet101(pretrained=True)

    transform = transforms.Compose([
        # transforms.R(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        )])


    # pre-process image and make predictions
    batch_t = torch.unsqueeze(transform(image), 0)

    resnet.eval()
    out = resnet(batch_t)

    # get top 5 predictions
    prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
    _, indices = torch.sort(out, descending=True)
    return [(classes[idx], prob[idx].item()) for idx in indices[0][:5]]


file_up = st.file_uploader('Upload an image', type='jpg')


# display image
if file_up is not None:
    image = Image.open(file_up)
    st.image(image, caption='Uploaded iamge', use_column_width=True)
    
    with open('imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]

    labels = predict(file_up, classes)

    for i in labels:
        st.write(f'Prediction (index, name): {i[0]} - Score  {i[1]}')
