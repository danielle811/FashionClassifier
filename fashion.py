import os
import gzip
import numpy as np
from PIL import Image

# load data
# offset = 8 for lables
# offset = 16 for images 
def read_data(path, offset_no):
    with gzip.open(path, 'rb') as file:
        data = np.frombuffer(file.read(), dtype=np.uint8, offset=offset_no)
    return data

# Convert array to images 
def arr_to_img(arr):
    mat = np.reshape(arr, (28, 28))
    img = Image.fromarray(np.uint8(mat) , 'L')
    img.show()

def getCategory(x):
    cat = {
        0: 'T-shirt/top',
        1: 'Trousers',
        2: 'Pullover',
        3: 'Dress', 
        4: 'Coat',
        5: 'Sandal',
        6: 'Shirt',
        7: 'Sneaker',
        8: 'Bag',
        9: 'Ankle boot'
    }
    return cat[x]

train_labels = read_data('data/train-labels-idx1-ubyte.gz', 8)
train_images = read_data('data/train-images-idx3-ubyte.gz', 16).reshape(len(train_labels), 784)

test_labels = read_data('data/t10k-labels-idx1-ubyte.gz', 8)
test_images = read_data('data/t10k-images-idx3-ubyte.gz', 16).reshape(len(test_labels), 784)


x = 100
arr_to_img(test_images[x])
print(getCategory(test_labels[x]))