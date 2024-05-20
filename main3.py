import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np

from util import classify, set_background


set_background('./bgs/chicken.jpg')

# set title
st.title(':blue[Poulty Disease Detector Through Chicken Fecal Images]')

# set header
st.header('Please upload an image')

# upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# load classifier
model = load_model('./model/chickenDiseaseV2.h5')

# load class names
with open('./model/labels.txt', 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    f.close()
# Ensure that class names are aligned with the expected classes
expected_class_names = ["Coccidiosis", "Healthy", "Salmonella"]
class_names = [name if name in expected_class_names else "Unknown" for name in expected_class_names ]

# display class names for verification
print(class_names)

# display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=False)

    # classify image
    class_name, conf_score = classify(image, model, class_names)

    # write classification
    st.write("## {}".format(class_name))
    st.write("### score: {}%".format(int(conf_score * 1000) / 10))
