# Image-to-STL Converter

A web-based application for converting images into STL files, ready for 3D printing in black and white. This app provides flexible control over image processing settings, enabling customized prints with variable shading and depth.

Here is the site : [Grayseal](https://github.com/cyborg-amp/grayseal.2/branch2/C:/Users/lexie/OneDrive/Documents/GitHub/grayseal.2/grayseal.streamlit.app)
"

## Table of Contents
- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Description

This application allows users to upload an image, which is then processed into a grayscale 3D model that can be exported as an STL file for 3D printing. The application runs on [Streamlit](https://streamlit.io/), a web framework that makes it easy to set up and deploy. The generated STL file reflects the input image in layered shades of gray, where each gray level corresponds to a distinct layer height in the print.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/cyborg-amp/grayseal.2.git
    ```
2. Navigate into the project directory:
    ```bash
    cd grayseal.2
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

## Usage

1. Open the app in your browser after running the `streamlit run` command.
2. Upload an image file.
3. Adjust settings:
   - **Size**: Modify the output dimensions of the STL file.
   - **Blurring**: Apply a blur effect to smoothen details.
   - **Noise Reduction**: Remove unwanted noise for cleaner layers.
   - **Grayscale Levels**: Control the number of grayscale shades, which will directly determine the number of distinct layers in the STL file.
   
4. Download the generated STL file for 3D printing.

## Features

- **Image to STL Conversion**: Turn any black and white image into an STL file.
- **Layer Control via Grayscale**: Specify the number of grayscale shades to control the depth and layering in the STL output.
- **Adjustable Settings**: Customize image size, blur, and noise reduction to achieve desired print results.
- **Simple Web Interface**: Easy-to-use Streamlit interface for quick setup and processing.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or suggestions, feel free to reach out via [GitHub](https://github.com/yourusername).
