import select
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from networkx import sigma
from skimage import io
from mpl_toolkits.mplot3d import Axes3D
import cv2 as cv2
import numpy as np
from stl import mesh
import streamlit as st
from PIL import Image
import cv2
import numpy as np
from skimage.measure import label, regionprops, marching_cubes
from skimage import measure, morphology, color
from skimage.segmentation import find_boundaries
from pythreejs import *
import ipywidgets as widgets
import io
from io import BytesIO
import time
from collections import deque

st.set_page_config(layout="wide")

rotation_y = np.array([
        [np.cos(np.radians(90)), 0, np.sin(np.radians(90))],
        [0, 1, 0],
        [-np.sin(np.radians(90)), 0, np.cos(np.radians(90))]
    ])

rotation_z = np.array([
        [np.cos(np.radians(180)), -np.sin(np.radians(180)), 0],
        [np.sin(np.radians(180)), np.cos(np.radians(180)), 0],
        [0, 0, 1]
    ])

def quantize_image(_image, num_colors):
    # Convert the image to grayscale using OpenCV.
    img_array = np.array(_image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    # Quantize the grayscale image.
    gray_levels = [255 * (i + 0.5) / num_colors for i in range(num_colors)]
    gray = gray.astype(np.float32) / 255
    quantized = 255 * np.floor(gray * num_colors + 0.5) / num_colors
    quantized = quantized.astype(np.uint8)

    return quantized, gray_levels

#@st.cache_data
def blur_image(_quantized_image, blur_ksize, num_colors):
    sigmaColor = 50
    sigmaSpace = 50
    # Apply blurring to the quantized image.
    blur1 = cv2.bilateralFilter(_quantized_image, blur_ksize,sigmaColor,sigmaSpace)
    blur = blur1.astype(np.float32) / 255

    # Re-quantize the blurred image.
    result = 255 * np.floor(blur * num_colors + 0.5) / num_colors
    result = result.clip(0, 255).astype(np.uint8)

    return result



def replace_small_regions(_image, min_area):
    # Convert the image to a NumPy array if it is not already.
    if not isinstance(_image, np.ndarray):
        _image = np.array(_image)

    # Label the connected regions
    labeled_image, num_labels = measure.label(_image, connectivity=2, return_num=True)

    # Get region properties
    regions = measure.regionprops(labeled_image)

    # Create a mask for small regions
    small_regions_mask = np.zeros_like(labeled_image, dtype=bool)

    # Mark small regions in the mask
    for region in regions:
        if region.area < min_area:
            for coord in region.coords:
                small_regions_mask[coord[0], coord[1]] = True

    # Replace small regions with the color of the nearest non-small region pixel
    result_image = _image.copy()
    for region in regions:
        if region.area < min_area:
            for coord in region.coords:
                x, y = coord
                # Use BFS to find the nearest non-small region pixel
                queue = deque([(x, y)])
                visited = set()
                found = False
                while queue and not found:
                    cx, cy = queue.popleft()
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < _image.shape[0] and 0 <= ny < _image.shape[1] and (nx, ny) not in visited:
                            if not small_regions_mask[nx, ny]:
                                result_image[x, y] = _image[nx, ny]
                                found = True
                                break
                            queue.append((nx, ny))
                            visited.add((nx, ny))

    return result_image

def crop_image(image, width_percentage, height_percentage):
    # Calculate crop dimensions
    width, height = image.size
    new_width = int(width * width_percentage / 100)
    new_height = int(height * height_percentage / 100)
    
    # Calculate the crop box (center crop)
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    
    # Crop the image
    cropped_image = image.crop((left, top, right, bottom))
    
    return cropped_image

def calculate_and_display_dimensions(result, volume):
    scale = st.slider("Scale", 1, 100, value=100, step=1, format="%d%%", help="100% fills the build plate.")
    larger = max(result.shape[0], result.shape[1])
    largerv = max(volume[0], volume[1])
    width = scale * 0.01 * largerv / larger * result.shape[1]
    height = scale * 0.01 * largerv / larger * result.shape[0]
    st.write("Modified Dimensions:")
    st.write(f"{width:.1f}mm x {height:.1f}mm")
    a=width/result.shape[1]
    b=height/result.shape[0]
    return a,b

import numpy as np
import cv2
from skimage.measure import label, regionprops

def process_image(result, gray_levels, num_colors, base_layers=10):
    masks = []
    
    # Create masks for each gray level
    for i in range(num_colors):
        lower_bound = gray_levels[i]
        _, mask = cv2.threshold(result, lower_bound, 255, cv2.THRESH_BINARY)
        masks.append(mask > (lower_bound - 1))

    # Label masks and get region properties
    labeled_masks = [label(mask) for mask in masks]
    region_props = [regionprops(labeled_mask) for labeled_mask in labeled_masks]

    # Extrude
    img_array = np.array(result)
    base = np.ones((base_layers, img_array.shape[0], img_array.shape[1]), dtype=bool)
    end = np.zeros((1, img_array.shape[0], img_array.shape[1]), dtype=bool)
    
    mask_layers = [np.repeat(mask[np.newaxis, :, :], 1, axis=0) for mask in masks]
    if mask_layers:
        mask_layers[-1] = np.repeat(mask_layers[-1], 3, axis=0)
    
    comb = np.concatenate((end, base, *mask_layers, end), axis=0)
    comb[-1, :, :] = 0
    comb[:, -1, :] = 0
    comb[:, :, -1] = 0
    comb[:, :, 0] = 0
    comb[:, 0, :] = 0
    z=len(masks)
    return comb, labeled_masks, region_props, z

# Example usage

st.title("Image to Stl Printer")
st.write("Upload an image below and convert it to a black and gray 3D printable stl file.")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="image_uploader")




col1, col2 = st.columns(2)


# Let the user upload an image file via `st.file_uploader`.

if not uploaded_file:
    with col1:
        st.info("Please upload an image to continue.", icon="🖼️")
# Remove the unnecessary else statement
    pass
else:
    #on = st.toggle("Display original image", help="Display original image side by side")
    image = Image.open(uploaded_file)
    with st.sidebar:
    #st.image(image, caption='Uploaded Image.', use_column_width=True)
        on = st.toggle("Display original image", help="Display original image side by side")

        num_colors = st.slider("Select number of quantization levels", min_value=1, max_value=10, value=5)
    # Slider to control the blurring effect.
        blur_ksize = st.slider("Select blurring kernel size", min_value=1, max_value=15, value=5, step=1)    
        pixel_value = st.slider("Select pixel value", min_value=1, max_value=10, value=1, step=1)
        width_percentage = st.slider("Select width percentage", min_value=1, max_value=100, value=100)
        height_percentage = st.slider("Select height percentage", min_value=1, max_value=100, value=100)
        #min_area = st.slider("Select minimum area for small regions", min_value=1, max_value=100, value=10)

    #if on:
        #with st.sidebar:
        #with col1:
            # Display the uploaded image.
            #original_width, original_height = image.size

        result= crop_image(image,width_percentage, height_percentage)
        result= np.array(result)

        quantized_image, gray_levels = quantize_image(result, num_colors)
        result = blur_image(quantized_image, blur_ksize, num_colors)
        result = replace_small_regions(result, 2.5**(pixel_value)+10)
        #result = Image.fromarray(result)

        #result = crop_image(result, width_percentage,height_percentage)
        #result.resize((original_width, original_height),Image.ANTIALIAS)
        #result = np.array(result)

            # Display the quantized and blurred grayscale image
                
        # Generate bounds
        #lower_bounds = [gray_levels[i] for i in range(num_colors)]
        #upper_bounds = [gray_levels[i] for i in range(num_colors)]
        st.write(f"Original image dimensions:{0.1*result.shape[1]:.1f}mm x {0.1*result.shape[0]:.1f}mm")

        #on = st.toggle("Display original image", help="Display original image side by side")
    if on:
        with col1:
            st.image(result, caption='Quantized and Blurred Grayscale Image.', use_column_width=True)
        with col2:
         st.image(image, caption='Uploaded Image.', use_column_width=True)
        
    else:
        st.image(result, caption='Quantized and Blurred Grayscale Image.', use_column_width=True)


    #on = st.toggle("size")            
    #if on:
    with st.expander("Size"):
        col1, col2, col3 = st.columns(3)

    # Add a selectbox for choosing options
        with col1:
            option = st.selectbox(
            'Select Size:', 
            ('Freesize', 'Choose 3D printer'),
            help="Freesize allows you to scale the image freely. If you choose a 3D printer, you can scale according to the the 3D printer's build volume."
            )


            if option == 'Choose 3D printer':
                with col2:
                    checkbox_option = st.radio(
                        "Choose an option:",
                        ('Anker', 'Anycubic', 'Bambu Lab', 'Creality', 'Elegoo', 'Prusa', 'Ultimaker', 'QIDI')
                    )
                    with col3:
                        if checkbox_option == 'Bambu Lab':
                            select_option = st.selectbox(
                                'Bambu Lab:',
                                ('Bambu lab A1 mini', 'Bambu Lab A1', 'Bambu Lab P1S', 'Bambu Lab P1P', 'Bambu Lab X1E', 'Bambu Lab X1', 'Bambu Lab X1 Carbon'),
                                index=None, placeholder="Select Model"
                            )
                            bambu_lab_volumes = {
                                'Bambu lab A1 mini': (180, 180),
                                'Bambu Lab A1': (256, 256),
                                'Bambu Lab P1S': (256, 256),
                                'Bambu Lab P1P': (256, 256),
                                'Bambu Lab X1E': (256, 256),
                                'Bambu Lab X1': (256, 256),
                                'Bambu Lab X1 Carbon': (256, 256)
                            }
                            if select_option in bambu_lab_volumes:
                                volume = bambu_lab_volumes[select_option]
                                a,b=calculate_and_display_dimensions(result, volume)

                        elif checkbox_option == 'Prusa':
                            select_option = st.selectbox(
                                'Prusa:',
                                ('Prusa MK3S', 'Prusa MK4', 'Prusa MINI+', 'Prusa i3 MK3S+', 'Prusa XL', 'Prusa Pro'),
                                index=None, placeholder="Select Model"
                            )
                            prusa_volumes = {
                                'Prusa MK3S': (250, 210),
                                'Prusa MK4': (250, 210),
                                'Prusa MINI+': (180, 180),
                                'Prusa i3 MK3S+': (250, 210),
                                'Prusa XL': (360, 360),
                                'Prusa Pro': (250, 210)
                            }
                            if select_option in prusa_volumes:
                                volume = prusa_volumes[select_option]
                                a,b=calculate_and_display_dimensions(result, volume)
                                            
                        if checkbox_option == 'Anker':
                            select_option = st.selectbox(
                                'Anker:',
                                ('Anker M5', 'Anker M5C'),
                                index=None, placeholder="Select Model"
                            )
                            anker_volumes = {
                                'Anker M5': (235, 235),
                                'Anker M5C': (235, 235)
                            }
                            if select_option in anker_volumes:
                                volume = anker_volumes[select_option]
                                a,b=calculate_and_display_dimensions(result, volume)



                        elif checkbox_option == 'Anycubic':
                            select_option = st.selectbox(
                                'Anycubic:',
                                ('Anycubic Kossel Linear Plus', 'Anycubic Kossel Pully', 'Anycubic Mega Zero', 'Anycubic Predator', 'Anycubic i3 Mega', 'Anycubic i3 Mega S', 'Anycubic Chiron', 'Anycubic Vyper', 'Anycubic Pro', 'Anycubic Pro 2', 'Anycubic Kobra 2', 'Anycubic Kobra 2 Neo', 'Anycubic Kobra 2 Max', 'Anycubic Kobra 2 Pro', 'Anycubic Kobra 2 Plus', 'Anycubic Kobra 3'),
                                index=None, placeholder="Select Model"
                            )
                            anycubic_volumes = {
                                'Anycubic Kossel Linear Plus': (230, 230),
                                'Anycubic Kossel Pully': (230, 230),
                                'Anycubic Mega Zero': (220, 220),
                                'Anycubic Predator': (370, 370),
                                'Anycubic i3 Mega': (210, 210),
                                'Anycubic i3 Mega S': (210, 210),
                                'Anycubic Chiron': (400, 400),
                                'Anycubic Vyper': (245, 245),
                                'Anycubic Pro': (210, 210),
                                'Anycubic Pro 2': (210, 210),
                                'Anycubic Kobra 2': (220, 220),
                                'Anycubic Kobra 2 Neo': (220, 220),
                                'Anycubic Kobra 2 Max': (400, 400),
                                'Anycubic Kobra 2 Pro': (220, 220),
                                'Anycubic Kobra 2 Plus': (300, 300),
                                'Anycubic Kobra 3': (220, 220)
                            }
                            if select_option in anycubic_volumes:
                                volume = anycubic_volumes[select_option]
                                a,b=calculate_and_display_dimensions(result, volume)

                        elif checkbox_option == 'Creality':
                            select_option = st.selectbox(
                                'Creality:',
                                ('Ender 3 V3 Plus', 'Ender 3 V3', 'Ender 3 V3 KE', 'Ender 3 V3 SE', 'Ender 3 S1', 'Ender 3 S1 Pro', 'Ender 3 S1 Plus', 'Ender 3 V2 Neo', 'Ender 3 Max Neo', 'Ender 3 Neo', 'Ender 3', 'Ender 3 Pro', 'Ender 3 V2', 'Ender 3 Max', 'K1', 'K1C', 'K1 Max', 'K1 SE'),
                                index=None, placeholder="Select Model"
                            )
                            creality_volumes = {
                                'Ender 3 V3 Plus': (220, 220),
                                'Ender 3 V3': (220, 220),
                                'Ender 3 V3 KE': (220, 220),
                                'Ender 3 V3 SE': (220, 220),
                                'Ender 3 S1': (220, 220),
                                'Ender 3 S1 Pro': (220, 220),
                                'Ender 3 S1 Plus': (300, 300),
                                'Ender 3 V2 Neo': (220, 220),
                                'Ender 3 Max Neo': (300, 300),
                                'Ender 3 Neo': (220, 220),
                                'Ender 3': (220, 220),
                                'Ender 3 Pro': (220, 220),
                                'Ender 3 V2': (220, 220),
                                'Ender 3 Max': (300, 300),
                                'K1': (220, 220),
                                'K1C': (220, 220),
                                'K1 Max': (300, 300),
                                'K1 SE': (220, 220)
                            }
                            if select_option in creality_volumes:
                                volume = creality_volumes[select_option]
                                a,b=calculate_and_display_dimensions(result, volume)

                        elif checkbox_option == 'Ultimaker':
                            select_option = st.selectbox(
                                'Ultimaker:',
                                ('Ultimaker S5', 'Ultimaker S7', 'Ultimaker S3', 'Ultimaker Method', 'Ultimaker Method X', 'Ultimaker Method XL'),
                                index=None, placeholder="Select Model"
                            )
                            ultimaker_volumes = {
                                'Ultimaker S5': (330, 240),
                                'Ultimaker S7': (330, 240),
                                'Ultimaker S3': (230, 190),
                                'Ultimaker Method': (190, 190),
                                'Ultimaker Method X': (190, 190),
                                'Ultimaker Method XL': (305, 305)
                            }
                            if select_option in ultimaker_volumes:
                                volume = ultimaker_volumes[select_option]
                                a,b=calculate_and_display_dimensions(result, volume)

                        elif checkbox_option == 'Elegoo':
                            select_option = st.selectbox(
                                'Elegoo:',
                                ('Neptune 4 Max', 'Neptune 4 Plus', 'Neptune 4 Pro', 'Neptune 4', 'Neptune 3 Max', 'Neptune 3 Plus', 'Neptune 3 Pro', 'Neptune 3'),
                                index=None, placeholder="Select Model"
                            )
                            elegoo_volumes = {
                                'Neptune 4 Max': (420, 420),
                                'Neptune 4 Plus': (320, 320),
                                'Neptune 4 Pro': (225, 225),
                                'Neptune 4': (225, 225),
                                'Neptune 3 Max': (420, 420),
                                'Neptune 3 Plus': (320, 320),
                                'Neptune 3 Pro': (225, 225),
                                'Neptune 3': (220, 220)
                            }
                            if select_option in elegoo_volumes:
                                volume = elegoo_volumes[select_option]
                                a,b=calculate_and_display_dimensions(result, volume)

                        elif checkbox_option == 'QIDI':
                            select_option = st.selectbox(
                                'QIDI:',
                                ('QIDI X-PLUS 3', 'QIDI Tech X-CF Pro', 'QIDI Tech I-Fast'),
                                index=None, placeholder="Select Model"
                            )
                            qidi_volumes = {
                                'QIDI X-PLUS 3': (270, 200),
                                'QIDI Tech X-CF Pro': (250, 250),
                                'QIDI Tech I-Fast': (330, 250)
                            }
                            if select_option in qidi_volumes:
                                volume = qidi_volumes[select_option]
                                #calculate_and_display_dimensions(result, volume)
                                a,b=calculate_and_display_dimensions(result, volume)

            else:
                with col2:
                    scale = st.slider("Scale", 1, 400, value=100, step=1,format="%d%%")
                    larger = 10
                    largerv=1
                    width = scale * 0.001 * result.shape[1]
                    height = scale * 0.001 * result.shape[0]
                    st.write(f"Modified Dimensions:")
                    st.write(f"{width:.1f}mm x {height:.1f}mm")



            #st.write(comb.shape)
            col1, col2= st.columns(2)
            comb, labeled_masks, region_props,z = process_image(result, gray_levels, num_colors)
                # Button to ask if they want to generate the STL file
            if st.button("Generate STL file"):
                # Marching cubes
                import tempfile

                # Generate the 3D mesh using marching cubes
                verts, faces, normals, values = marching_cubes(comb)
                obj_3d = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
                for i, f in enumerate(faces):
                    for j in range(3):
                        obj_3d.vectors[i][j] = verts[f[j], :]

                # Apply rotations
                for i in range(len(obj_3d.vectors)):
                    obj_3d.vectors[i] = np.dot(obj_3d.vectors[i], rotation_y)
                    obj_3d.vectors[i] = np.dot(obj_3d.vectors[i], rotation_z)

                # Define scaling factors for x and y
                num_colour = z  # Example definition
                z_value = (num_colour + 12) * 0.08
                #width = scale * 0.01 * largerv / larger
                #height = scale * 0.01 * largerv / larger

                # Apply scaling to x and y coordinates
                for i in range(len(obj_3d.vectors)):
                    for j in range(3):
                        obj_3d.vectors[i][j][0] *= a
                        obj_3d.vectors[i][j][1] *= b
                        obj_3d.vectors[i][j][2] = (obj_3d.vectors[i][j][2] - 2) * 0.08  # Subtract 2 and then multiply by 0.08

                st.write(a, b, z_value)

                # Save the STL file to a temporary file
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    obj_3d.save(tmp_file.name)
                    tmp_file.seek(0)
                    stl_data = tmp_file.read()
                    file_size = os.path.getsize(tmp_file.name)
                    st.success(f"STL file generated: {tmp_file.name}")
                    st.write(f"File size: {file_size} bytes")

                # Create a BytesIO object from the temporary file data
                stl_io = BytesIO(stl_data)

                # Create a download button
                st.download_button(
                    label="Download STL file",
                    data=stl_io,
                    file_name="output.stl",
                    mime="application/octet-stream"
                )

                # Save the STL file locally
                obj_3d.save('output.stl')
                st.success("STL file generated successfully!")