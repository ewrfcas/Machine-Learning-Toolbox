from threadpool import ThreadPool,makeRequests
import cv2
import numpy as np
from tqdm import tqdm
# Load images and define data_generator
# images=glob.glob(os.path.join(data_dir,'*.png'))
data=[]
label=[]

def preprocess(image):
    image=np.transpose(image,(2,0,1))
    image=(image-127.5)*0.0078125
    image=image.astype(np.float32)
    return image

def load_image(path):
    sr_img=cv2.imread(path)
    lr_img=cv2.resize(sr_img,(sr_img.shape[1]//8,sr_img.shape[0]//8))
    sr_img=preprocess(sr_img)
    lr_img=preprocess(lr_img)
    return lr_img,sr_img

with tqdm(total=len(images),desc='Loading images') as pbar:
    def callback(req,x):
        data.append(x[0])
        label.append(x[1])
        pbar.update()

    t_pool=ThreadPool(16)
    requests=makeRequests(load_image,images,callback=callback)
    for req in requests:
        t_pool.putRequest(req)
    t_pool.wait()
