"""
Web app built on Streamlit
"""
import streamlit as st
from pil_utils import *
from request_api import *

st.write("""
# Open set classification app
""")
st.write("Classification of Mercedes models with open-set image filtering.")
st.write("by Andrew Wang")

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
    st.sidebar.text(" ")
else:
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
