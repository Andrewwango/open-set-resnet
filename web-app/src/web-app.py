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

options = get_model_options()

paths = get_sample_images()
class File:
    file = None

def int_gen():
    a=-1
    while 1: a+=1; yield a

col1,col2,col3,col4,col5,col6=st.columns(6)
g = int_gen()
def change_file(p):
    print("Changing to ", p)
    File.file = p
    print("Changed to ", File.file)


with col1:
    i=next(g); st.image(paths[i], use_column_width=True)
    st.button("Use", key=str(i), on_click=lambda:change_file(paths[i]))

#with col2: i=next(g); st.image(paths[i], use_column_width=True); file = paths[i] if st.button("Use", key=str(i)) else file
#with col3: i=next(g); st.image(paths[i], use_column_width=True); file = paths[i] if st.button("Use", key=str(i)) else file
#with col4: i=next(g); st.image(paths[i], use_column_width=True); file = paths[i] if st.button("Use", key=str(i)) else file
#with col5: i=next(g); st.image(paths[i], use_column_width=True); file = paths[i] if st.button("Use", key=str(i)) else file
#with col6: i=next(g); st.image(paths[i], use_column_width=True); file = paths[i] if st.button("Use", key=str(i)) else file


col1, col2 = st.columns(2)
with col1:
    print("Checking col", File.file)
    File.file = File.file if File.file is not None else st.file_uploader("Please upload an image file", type=["jpg", "png"])
with col2:
    model = st.selectbox(label="Select model", options=options)
    classif = st.select_slider(label="Open set or closed set", options=["Open-set", "Closed-set"])
    open_set = classif=='Open-set'

with open(full_path("image_path.txt"), 'r') as f:
    p = f.read()
    File.file = p if p!="" else None

if File.file is None:
    print("file is None", File.file)
    st.sidebar.text(" ")
else:
    print("file is", File.file)
    image = pil_open_image(File.file)
    col1,_ = st.sidebar.columns(2)
    with col1:
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



