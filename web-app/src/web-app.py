"""
Web app built on Streamlit
"""
import streamlit as st
from pil_utils import *
from request_api import *

st.write("""
# Open set classification app
## Image classification with open-set image filtering.
**by Andrew Wang**
Open-set classification is critical for letting image classifiers work in the real world. Source: https://github.com/Andrewwango/open-set-resnet

Use the machine learning API directly here: http://open-set-resnet-api.herokuapp.com/ 

""")

file = None

file_cols = st.columns(2)
with file_cols[0]:
    file = file if file is not None else st.file_uploader("Please upload an image file", type=["jpg", "png"])
with file_cols[1]:
    paths,path_to_name = get_sample_images()
    path = st.selectbox(label="Or select an image...", options=paths, format_func=path_to_name)
    if st.button("Choose"):
        file = path

param_cols = st.columns(2)
with param_cols[0]:
    model = st.selectbox(label="Select model", options=get_model_options())
with param_cols[1]:
    classif = st.select_slider(label="Open set or closed set", options=["Open-set", "Closed-set"])
    open_set = classif=='Open-set'

if file is None:
    st.sidebar.write("## Results")
else:
    st.sidebar.write("## Results")
    image = pil_open_image(file)
    with st.sidebar.columns(2)[0]:
        st.image(image, use_column_width=True)

    with st.spinner('Wait for it...'):
        results = post_image_for_inference(bytes=pil_to_bytes(image), 
                                    model=str(model), 
                                    open_set=open_set)
    pred = results["pred"]
    conf = results["conf"]
    #st.write(f"Prediction: {pred}, Confidence: {conf}")
 
    #diagrams = split_diagram(get_diagram(model_name=model, open_set=open_set))
    diagram = get_diagram(pred, int(results["level"]), model_name=model, open_set=open_set)

    st.sidebar.image(diagram, use_column_width=True)

st.write("""
## Introduction
Currently, a normal image classifier will assign a random image a category despite it not belonging to any specific category. These closed-set classifiers often do this with high confidence. An open-set classifier should detect images that do not belong in any of the classes. For example, a spaniels classifier should filter images of non-dogs and of non-spaniels; a car-model classifier should filter images of other makes or non-cars. 

This repo contains an example classifier which takes a spaniel/dog/Mercedes model classifier and adds open-set filtering capabilities. The classifier structure is as follows:

1. Classify image according to original ImageNet and reject if not car/dog-like (based on ImageNet labels).
2. Classify image according to 2-class "species" classifier trained on spaniels vs. non-spaniels/Mercedes vs. non-Mercedes, and reject if not spaniel/Mercedes.
3. Classify image according to original closed-set classifier (spaniel-breeds/Mercedes models).

All the models are based on the ResNet architecture and use PyTorch for training and inference:
1. Model 1: resnet18 with pretrained weights on ImageNet
2. Model 2: resnet18 pretrained, and then retrained to 2-class dataset (correct make/species vs. incorrect) using transfer learning.
3. Model 3: original pretrained and retrained resnet50 closed-set classifier. 

This is the equivalent of first asking a friend what a car is, then asking a friend what Mercedes is, then what the individual models are.

## Deployment
The open-set inference is developed as an API using FastAPI and uvicorn. This can be accessed using `requests.post`. You can test out different models on the Streamlit web-app. We deploy this repo as 2 separate apps on Heroku.

## Training
Model training can be done in the training folder. To create a different open-set classifier, two models are needed:

1. Your original closed-set classifier.

2. Train another model with all the closed-set classes in one class, and images of different species but same thing in the other (e.g. non-Mercedes cars, or non-cow animals).

To set up another model,

1. Put images in training folder/AWS S3 bucket.

2. Run training notebook with correct training folder location.

3. Move models over to `api/src/models`

4. Add classifier to `api/src/classification.config` including model locations and label names.

## Getting started locally

1.

        git clone https://github.com/Andrewwango/open-set-resnet.git
        cd open-set-resnet
        pip install -r requirements.txt
        cd python

2. Start API: `uvicorn api.src.main:app --reload`

3. Start Web app: `streamlit run web-app/src/web-app.py`
**OR**
3. Query the API using Swagger UI at `http://localhost:8000/docs`
**OR**
3. Call the inference function in Python (see ![demo](python/demo.ipynb))

        from api import open_set_inference as osi 
        osi.classify_open_set(image='test-images/animal.jpg')

## Datasets
Cars: [Stanford](http://ai.stanford.edu/~jkrause/cars/car_dataset.html )
Dogs: [Kaggle](https://www.kaggle.com/gpiosenka/70-dog-breedsimage-data-set)

""")