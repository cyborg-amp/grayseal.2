import select
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage import io
from mpl_toolkits.mplot3d import Axes3D
import cv2 as cv2
import numpy as np
from stl import mesh
import streamlit as st
from PIL import Image
import cv2
import numpy as np
from skimage.measure import label, regionprops
from skimage import measure
from pythreejs import *
import ipywidgets as widgets
import io

st.set_page_config(layout="wide")
st.title("Image to Stl Printer")
st.write("Upload an image below and convert it to a black and gray 3D printable stl file.")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="image_uploader")


col1, col2, = st.columns(2)


# Let the user upload an image file via `st.file_uploader`.

if not uploaded_file:
    with col1:
        st.info("Please upload an image to continue.", icon="🖼️")
else:
    on = st.toggle("Display original image")
             
    if on:
        with col1:
            # Display the uploaded image.
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            st.write(f"Original image dimensions:{0.1*image.size[0]}mm x {0.1*image.size[1]}mm")
        
        # Slider to control the number of quantization levels.
        num_colors = st.slider("Select number of quantization levels", min_value=1, max_value=10, value=5)
        # Slider to control the blurring effect.
        blur_ksize = st.slider("Select blurring kernel size", min_value=1, max_value=20, value=5, step=1)

        # Convert the image to grayscale using OpenCV.
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

        # Quantize the grayscale image.
        gray_levels = [255 * (i + 0.5) / num_colors for i in range(num_colors)]
        gray = gray.astype(np.float32) / 255
        quantized = 255 * np.floor(gray * num_colors + 0.5) / num_colors
        quantized = quantized.astype(np.uint8)

        # Apply blurring to the quantized image.
        blur1 = cv2.blur(quantized, (blur_ksize, blur_ksize))
        blur = blur1.astype(np.float32) / 255

        # Re-quantize the blurred image.
        result = 255 * np.floor(blur * num_colors + 0.5) / num_colors
        result = result.clip(0, 255).astype(np.uint8)

        # Display the quantized and blurred grayscale image.
        
        # Generate bounds
        lower_bounds = [gray_levels[i] for i in range(num_colors)]
        upper_bounds = [gray_levels[i] for i in range(num_colors)]

        with col2:
            st.image(result, caption='Quantized and Blurred Grayscale Image.', use_column_width=True)
            


            
    else:
        image = Image.open(uploaded_file)
        with st.sidebar:
            num_colors = st.slider("Select number of quantization levels", min_value=1, max_value=10, value=5)
            # Slider to control the blurring effect.
            blur_ksize = st.slider("Select blurring kernel size", min_value=1, max_value=15, value=5, step=1)

        # Convert the image to grayscale using OpenCV.
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

        # Quantize the grayscale image.
        gray_levels = [255 * (i + 0.5) / num_colors for i in range(num_colors)]
        gray = gray.astype(np.float32) / 255
        quantized = 255 * np.floor(gray * num_colors + 0.5) / num_colors
        quantized = quantized.astype(np.uint8)

        # Apply blurring to the quantized image.
        blur1 = cv2.blur(quantized, (blur_ksize, blur_ksize))
        blur = blur1.astype(np.float32) / 255

        # Re-quantize the blurred image.
        result = 255 * np.floor(blur * num_colors + 0.5) / num_colors
        result = result.clip(0, 255).astype(np.uint8)
        lower_bounds = [gray_levels[i] for i in range(num_colors)]
        upper_bounds = [gray_levels[i] for i in range(num_colors)]
        st.image(result, caption='Quantized and Blurred Grayscale Image.', use_column_width=True)
        st.write(f"Original image dimensions: {0.1*image.size[0]}mm x {0.1*image.size[1]}mm")
    col1, col2, col3 = st.columns(3)


# Add a selectbox for choosing options
    with col1:
        option = st.selectbox(
        'Select Size:', 
        ('Freesize', 'Choose 3D printer'),
        help="Freesize allows you to scale the image freely. If you choose a 3D printer, you can scale according to the the 3D printer's build volume."
        )

        if option =='Choose 3D printer':
            with col2:
                    checkbox_option = st.radio(
                    "Choose an option:",
                ('Anker', 'Anycubic', 'Bambu Lab', 'Creality', 'Elegoo', 'Prusa', 'Ultimaker', 'QIDI')
              )
        
            with col3:
                if checkbox_option == 'Bambu Lab':
                   select_option = st.selectbox(
                      'Bambu Lab:',
                       ('Bambu lab A1 mini', 'Bambu Lab A1', 'Bambu Lab P1S', 'Bambu Lab P1P', 'Bambu Lab X1E', 'Bambu Lab X1', 'Bambu Lab X1 Carbon',),
                       index=None, placeholder="Select Model"
                   )
                   if select_option in ['Bambu lab A1 mini', 'Bambu Lab A1', 'Bambu Lab P1S', 'Bambu Lab P1P', 'Bambu Lab X1E', 'Bambu Lab X1', 'Bambu Lab X1 Carbon',]:
                         if select_option == 'Bambu lab A1 mini':
                              volume = (180, 180)
                              
                         elif select_option == 'Bambu Lab A1':
                              volume = (256, 256)
                         elif select_option == 'Bambu Lab P1S':
                              volume = (256, 256)
                         elif select_option == 'Bambu Lab P1P':
                              volume = (256, 256)
                         elif select_option == 'Bambu Lab X1E':
                              volume = (256, 256)
                         elif select_option == 'Bambu Lab X1':
                              volume = (256, 256)
                         elif select_option == 'Bambu Lab X1 Carbon':
                              volume = (256, 256)
                         
                         scale = st.slider("Scale", 1, 100, value=100, step=1, format="%d%%", help="100% fills the build plate.")
                         larger = max(image.size[0], image.size[1])
                         largerv = max(volume[0], volume[1])
                         width = scale * 0.01 * largerv / larger * image.size[0]
                         height = scale * 0.01 * largerv / larger * image.size[1]
                         st.write("Modified Dimensions:")
                         st.write(f"{width:.1f}mm x {height:.1f}mm")

            
                elif checkbox_option =='Prusa':
                    select_option = st.selectbox(
                       'Prusa:',
                       ('Prusa MK3S', 'Prusa MK4', 'Prusa MINI+', 'Prusa i3 MK3S+', 'Prusa XL', 'Prusa Pro',),
                       index=None, placeholder="Select Model"
                   )
                    if select_option in ['Prusa MK3S', 'Prusa MK4', 'Prusa MINI+', 'Prusa i3 MK3S+', 'Prusa XL', 'Prusa Pro',]:
                         if select_option == 'Prusa MK3S':
                              volume = (250, 210)                              
                         elif select_option == 'Prusa MK4':
                              volume = (250, 210)
                         elif select_option == 'Prusa MINI+':
                              volume = (180, 180)
                         elif select_option == 'Prusa i3 MK3S+':
                              volume = (250, 210)
                         elif select_option == 'Prusa XL':
                              volume = (360, 360)
                         elif select_option == 'Prusa Pro':
                              volume = (250, 210)
                         
                         scale = st.slider("Scale", 1, 100, value=100, step=1, format="%d%%",help="100% fills the build plate.")
                         larger = max(image.size[0], image.size[1])
                         largerv = max(volume[0], volume[1])
                         width = scale * 0.01 * largerv / larger * image.size[0]
                         height = scale * 0.01 * largerv / larger * image.size[1]
                         st.write("Modified Dimensions:")
                         st.write(f"{width:.1f}mm x {height:.1f}mm")
                    
                elif checkbox_option == 'Anker':
                    select_option = st.selectbox(
                      'Anker:',
                      ('Anker M5', 'Anker M5C',),
                      index=None, placeholder="Select Model"
                  )
                elif checkbox_option == 'Anycubic':
                    select_option = st.selectbox(
                        'Anycubic:',
                     ('Anycubic Kossel Linear Plus', 'Anycubic Kossel Pully', 'Anycubic Mega Zero', 'Anycubic Predator', 'Anycubic i3 Mega', 'Anycubic i3 Mega S', 'Anycubic Chiron', 'Anycubic Vyper', 'Anycubic Pro', 'Anycubic  Pro 2', 'Anycubic Kobra 2', 'Anycubic Kobra 2 Neo', 'Anycubic Kobra 2 Max', 'Anycubic Kobra 2 Pro', 'Anycubic Kobra 2 Plus', 'Anycubic Kobra 3',),
                     index=None, placeholder="Select Model"
                   )
                elif checkbox_option == 'Creality':
                    select_option = st.selectbox(
                       'Creality:',
                       ('Ender 3 V3 Plus','Ender 3 V3','Ender 3 V3 KE','Ender 3 V3 SE','Ender 3 S1', 'Ender 3 S1 Pro','Ender 3 S1 Plus', 'Ender 3 V2 Neo','Ender 3 Max Neo' ,'Ender 3 Neo', 'Ender 3', 'Ender 3 Pro', 'Ender 3 V2', 'Ender 3 Max','K1', 'K1C','K1 Max', 'K1 SE'),
                       index=None, placeholder="Select Model"
                   )
                elif checkbox_option == 'Ultimaker':
                    select_option = st.selectbox(
                        'Ultimaker:',
                        ('Ultimaker S5','Ultimaker S7','Ultimaker S3', 'Ultimaker Method','Ultimaker Method X','Ultimaker Method XL'),
                        index=None, placeholder="Select Model"
                    )
                elif checkbox_option == 'Elegoo':
                    select_option = st.selectbox(
                        'Elegoo:',
                        ('Neptune 4 Max','Neptune 4 Plus','Neptune 4 Pro', 'Neptune 4','Neptune 3 Max', 'Neptune 3 Plus', 'Neptune 3 Pro', 'Neptune 3'),
                        index=None, placeholder="Select Model")
                
                elif checkbox_option == 'QIDI':
                    select_option = st.selectbox(
                       'QIDI:',
                       ('QIDI X-PLUS 3', 'QIDI Tech X-CF Pro', 'QIDI Tech I-Fast',),
                       index=None, placeholder="Select Model")

        else:
             with col2:
                scale = st.slider("Scale", 1, 400, value=100, step=1,format="%d%%")
                width = scale * 0.001 * image.size[0]
                height = scale * 0.01 * image.size[1]
                st.write(f"Modified Dimensions:")
                st.write(f"{width:.1f}mm x {height:.1f}mm")



        # Display the selected option
        st.write('You selected:', option)
            # Generate masks
        masks = []
        for i in range(num_colors):
            ret, mask = cv2.threshold(result, lower_bounds[i], lower_bounds[i], cv2.THRESH_BINARY)
            masks.append(mask > (lower_bounds[i] - 1))

        # Label masks and get region properties
        labeled_masks = [label(mask) for mask in masks]
        region_props = [regionprops(labeled_mask) for labeled_mask in labeled_masks]

        # Extrude
        b = np.repeat([bool(1)], img_array.shape[1], axis=0)  # size
        ba = np.repeat([b], img_array.shape[0], axis=0)
        base = np.repeat([ba], 10, axis=0)
        e = np.repeat([bool(0)], img_array.shape[1], axis=0)
        en = np.repeat([e], img_array.shape[0], axis=0)
        end = np.repeat([en], 1, axis=0)
        base1 = np.repeat([en], 1, axis=0)
        mask_layers = [np.repeat([mask], 1, axis=0) for mask in masks]
        comb = np.concatenate((base1, base, *mask_layers, end))
        comb[-1, :, :] = 0
        comb[:, -1, :] = 0
        comb[:, :, -1] = 0
        comb[:, :, 0] = 0
        comb[:, 0, :] = 0
        # Button to ask if they want to generate the STL file
        if st.button("Generate STL file"):
            # Marching cubes
            verts, faces, normals, values = measure.marching_cubes(comb)
            obj_3d = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
            for i, f in enumerate(faces):
                obj_3d.vectors[i] = verts[f]



            st.write_stream(stream)