import os, os.path
from PIL import Image
import numpy as np

size = 224, 224
path = "/home/shyam.nandan/WGAN/LearningToSeeThroughTurbulrntWater/TurbulentWater-Enssemble/results/DTD_GANEn"
valid_images = [".JPEG"]
for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    #print(f, ext, os.path.join(path,f))
    im = Image.open(os.path.join(path,f))
    img = np.array(im)
    #if(len(img.shape)==2):
    im = im.convert("RGB")
    im = im.resize((224, 224), Image.ANTIALIAS)
    im.save(os.path.join(path,f), "JPEG")
    print(f)
