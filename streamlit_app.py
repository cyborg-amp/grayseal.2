import streamlit as st
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.measure import label,regionprops
from skimage import io
from mpl_toolkits.mplot3d import Axes3D
import cv2 as cv2
import numpy as np
from skimage import measure
from stl import mesh
from PIL import Image

import streamlit as st
from PIL import Image
import cv2
import numpy as np

import streamlit as st
from PIL import Image
import cv2
import numpy as np

st.title("📄 Document question answering")
st.write("Upload an image below and ask a question about it – GPT will answer!")

# Let the user upload an image file via `st.file_uploader`.
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="image_uploader")
if not uploaded_file:
    st.info("Please upload an image to continue.", icon="🖼️")
else:
    # Display the uploaded image.
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Convert the image to grayscale using OpenCV.
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    # Quantize the grayscale image.
    num_colors = 5
    gray_levels = [255 * (i + 0.5) / num_colors for i in range(num_colors)]
    gray = gray.astype(np.float32) / 255
    quantized = 255 * np.floor(gray * num_colors + 0.5) / num_colors
    quantized = quantized.astype(np.uint8)

    # Display the quantized grayscale image.
    st.image(quantized, caption='Quantized Grayscale Image.', use_column_width=True)

    # Ask the user for a question via `st.text_area`.
    question = st.text_area(
        "Now ask a question about the image!",
        placeholder="Can you describe the image?",
        disabled=not uploaded_file,
    )

    if question:
        # Process the uploaded image and question.
        image_bytes = uploaded_file.read()
        messages = [
            {
                "role": "user",
                "content": f"Here's an image: {image_bytes} \n\n---\n\n {question}",
            }
        ]

        # Generate an answer using the OpenAI API.
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            stream=True,
        )

        # Stream the response to the app using `st.write_stream`.
        st.write_stream(stream)
