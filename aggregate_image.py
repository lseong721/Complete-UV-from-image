import cv2
import os
from glob import glob
import numpy as np

def dilate_image(image_path, dilation_size=5):
    """
    Dilate an image with a specified dilation size.
    
    Parameters:
    - image_path: Path to the input image.
    - dilation_size: Size of the dilation kernel. Default is 5.
    
    Returns:
    - The dilated image.
    """
    # Load the image
    image = cv2.imread(image_path)
    
    # Define the kernel size
    kernel = np.ones((dilation_size, dilation_size), np.uint8)
    
    # Apply dilation
    dilated_image = cv2.dilate(image, kernel, iterations=1)
    
    return dilated_image

def main():
    dilation_size = 2

    img_list = sorted(glob('results/image_*.png'))

    # images = [dilate_image(i, dilation_size=3) for i in img_list]
    images = [cv2.imread(i) for i in img_list]

    image_stack = np.stack(images, axis=0)
    # image_stack = image_stack / 127.5 - 1

    # Create a masked array where zeros are ignored
    masked_stack = np.ma.masked_equal(image_stack, 128)

    # Compute the median along the stack axis, ignoring the masked (zero) values
    median_image = np.ma.median(masked_stack, axis=0).filled(128)  # .filled(0) fills masked values with 0 in the output

    # cv2.imwrite("median_image.png", (median_image + 1) * 127.5)
    cv2.imwrite("median_image.png", median_image)

if __name__ == "__main__":
    main()