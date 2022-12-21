import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings
import glob

cv2.ocl.setUseOpenCL(False)
warnings.filterwarnings('ignore')
feature_extraction_algo = "sift"
feature_to_match = 'bf'

# 读入图片
image_path = glob.glob('imgs/src_2/*.jpg')
images = ()
for image in image_path:
    img = cv2.imread(image)
    img = cv2.resize(img, (1280, 720))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    images.append(img)

# 展示图片
fig, images = plt.subplots(nrows=1, ncols=images.count(), constrained_layout= False, figsize=(16, 9))
for image in

    https: // www.youtube.com / watch?v = uMABRY8QPe0