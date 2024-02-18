#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import streamlit as st
from PIL import Image
import cv2  # Import OpenCV for image processing

def create_skin_mask(image, kernel_size=5):
    """Creates a skin mask using HSV color space and morphological operations.

    Args:
        image (np.ndarray): The input image in BGR format.
        kernel_size (int, optional): Size of the morphological structuring element. Defaults to 5.

    Returns:
        np.ndarray: The skin mask (values of 0 for non-skin, 255 for skin).
    """

    # Convert to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define HSV range for skin color (adjust these values as needed)
    lower_skin = np.array([0, 48, 80], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Threshold the HSV image
    mask = cv2.inRange(hsv_image, lower_skin, upper_skin)

    # Perform morphological operations to refine the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Blur the mask to reduce noise
    mask = cv2.GaussianBlur(mask, (3, 3), 0)

    return mask

def main():
    st.title("Landmark Recognition (Improved)")

    # Image upload and display
    uploaded_image = st.file_uploader("Choose your image (PNG, JPG):", type=['png', 'jpg'])
    if uploaded_image is not None:
        # Use PIL for image preprocessing (robustly handle alpha channels)
        image = Image.open(uploaded_image)
        image = image.convert('RGB')  # Ensure the image is in RGB format

        # Convert to NumPy array and BGR for OpenCV
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # User-adjustable kernel size
        kernel_size = st.sidebar.slider('Kernel size:', 1, 20, 5, 2)

        # Create and display the skin mask
        mask = create_skin_mask(image, kernel_size)
        st.image(mask, channels='L', width=400)  # Display as grayscale image

if __name__ == '__main__':
    main()


# In[ ]:




