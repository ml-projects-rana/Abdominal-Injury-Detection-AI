
from components.modules import *
import pandas as pd

# -------- converting .dcm (images) to png - converting instead of white color to grayscale image also using 512 image size ---------
def test_preprocessing(test_images_dir, final_train_images_png_dir):
    convert_dcm_and_restructure_image(test_images_dir, final_train_images_png_dir)
   
