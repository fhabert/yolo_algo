import numpy as np
import math
from bs4 import BeautifulSoup
import requests
import random
from PIL import Image
from io import BytesIO
import csv
import cv2 as cv
import time

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

class Yolo(object):
    def __init__(self):
        pass

    def edges_mask(self, mask, I):
        intensity = []
        for i in range(1, len(I)-1):
            inner_list = []
            for j in range(1, len(I[0])-1):
                new_value = I[i-1][j-1]*mask[0][0] + I[i-1][j]*mask[0][1] + I[i-1][j+1]*mask[0][2] \
                             + I[i][j-1]*mask[1][0] + I[i][j]*mask[1][1] + I[i][j+1]*mask[1][2] \
                             + I[i+1][j-1]*mask[2][0] + I[i+1][j]*mask[2][1] + I[i+1][j+1]*mask[2][2]
                inner_list.append(abs(new_value))
            intensity.append(inner_list)
        return np.array(intensity)

    def convolution_layer(self, img_pix_resize=None, relu=None):
        if type(relu) == type(np.array([])) or type(relu) == list:
            img_grey_scale = relu
        else:
            img_grey_scale = [[np.uint8(sum(x)/3) for x in inner_list] for inner_list in list(img_pix_resize)]
        sobel_filter_x = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        sobel_filter_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_x_mat = np.sqrt(np.square(self.edges_mask(sobel_filter_x, img_grey_scale)))
        sobel_y_mat = np.sqrt(np.square(self.edges_mask(sobel_filter_y, img_grey_scale)))
        combined_mat = np.sqrt(np.add(np.square(sobel_x_mat), np.square(sobel_y_mat)))
        median_mat = self.median_value(combined_mat)
        return median_mat
    
    def relu_layer(self, conv):
        relu_conv = []
        for i in range(len(conv)):
            inner_list = []
            for j in range(len(conv[0])):
                pixel_value = conv[i][j]
                new_value = func.Relu(pixel_value)
                inner_list.append(abs(new_value))
            relu_conv.append(inner_list)
        return np.array(relu_conv)
    
    def pooling_layer(self, relu):
        pool = []
        for i in range(0, len(relu)-2, 2):
            inner_list = []
            for j in range(0, len(relu[0])-2, 2):
                # array = [relu[i-1][j-1], relu[i-1][j], relu[i-1][j+1], relu[i][j-1], relu[i][j], relu[i][j+1],
                #         relu[i+1][j-1], relu[i+1][j], relu[i+1][j+1]]
                array = [relu[i-1][j-1], relu[i-1][j],relu[i][j-1], relu[i][j],]
                inner_list.append(max(array))
            pool.append(inner_list)
        return np.array(pool)

    def median_value(self, I):
        # Applying the median filter and creating a new intensity matrix of pixels
        intensity = []
        for i in range(1, len(I)-1):
            inner_list = []
            for j in range(1, len(I)-1):
                new_value = np.median([I[i-1][j-1], I[i-1][j], I[i-1][j+1], I[i][j-1], \
                    I[i][j], I[i][j+1], I[i+1][j-1], I[i+1][j], I[i+1][j+1]])
                inner_list.append(new_value)
            intensity.append(inner_list)
        return np.array(intensity)

    def execute_yolo(self, img):
        conv_yolo = self.convolution_layer(img)
        relu_yolo = self.relu_layer(conv_yolo)
        conv_yolo = self.convolution_layer(relu=relu_yolo)
        relu_yolo = self.relu_layer(conv_yolo)
        pool_yolo = self.pooling_layer(relu_yolo)
        return pool_yolo
        

class NN(object):
    def __init__(self):
        self.layers = []
        pass

    def create_weights_nodes(self, structure):
        weights = []
        for i in range(len(structure)-1):
            node_weights = []
            for _ in range(structure[i]):
                layer_weights = []
                for _ in range(structure[i+1]):
                    layer_weights.append(random.uniform(0,1))
                node_weights.append(layer_weights)
            weights.append(node_weights)
        return weights
    
    def set_layers(self, layers):
        self.layers = layers
        pass
    
    def forward_pass(self):
        pass


class Layer(object):
    def __init__(self, nodes, num):
        self.nodes = nodes
        self.num = num
        pass

class Node(object):
    def __init__(self, weights, activation_func, input_value=None):
        self.weights = weights
        self.activation_func = activation_func
        self.input_value = input_value
        pass

class Functions():
    def __init__(self):
        pass

    def Relu(self, value):
        if value < 0:
            return 0
        else:
            return value

    def Leaky_Relu(self, value):
        if value < 0:
            return 0.01
        else:
            return value

    def sigmoid(self, value):
        den = 1 + math.exp(-value)
        return 1/den

    def tanh(self, value):
        return math.tanh(value)

    def linear(self, value):
        return value

class Derivatives():
    def __init__(self):
        pass

    def Relu(self, value):
        if value < 0:
            return 0
        else:
            return 1

    def Leaky_Relu(self, value):
        if value < 0:
            return 0.01
        else:
            return 1

    def sigmoid(self, value):
        den = 1 + math.exp(-value)
        sig = 1/den
        return (1 - sig) * sig

    def tanh(self, value):
        return 1 - (math.tanh(value))**2

    def linear(self, value):
        return 1     

def get_images(num_images, nb_page):
    global num_image
    url = f"https://www.freepik.com/photos/tables-open/{nb_page}" 
    HEADERS = {'User-Agent': 'Mozilla/5.0 (iPad; CPU OS 12_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148'}
    webpage_source = requests.get(url=url, headers=HEADERS)
    webpage = webpage_source.content
    tables = BeautifulSoup(webpage, "html.parser")
    images = tables.find_all("figure", class_="showcase__item", limit=num_images)
    img_src = [img['data-image'] for img in images]
    for i in range(len(img_src)):
        r = requests.get(url=img_src[i], headers=HEADERS)
        if r.status_code == 200:
            # img_content = Image.open(BytesIO(r.content))
            # image_resize = img_content.resize((256,256))
            # img_pixels_resize = np.asarray(image_resize)
            # pool = yolo.execute_yolo(img_pixels_resize)
            # flatten_pool = list(np.array([[255 if x > 255 else np.uint(x) for x in inner_list] for inner_list in pool]).flatten())
            # if i == 1:
            #     img = Image.fromarray(pool)
            #     img.show()
            # add_line(flatten_pool)
            # print("Add a photo:", i)
            with open(f"images3/table{num_image}.png", "ab") as f:
                f.write(r.content)
                print("New picture:", i)
            f.close()
        num_image += 1
    pass

def initialise_neural():
    structure = [5, 15, 4, 1]
    neural = NN()
    # biases = [[random.uniform(0,1) for _ in range(x)] for i in range(1, len(structure))]
    weights = neural.create_weights_nodes(structure)
    layers_all_nodes = []
    for i in range(len(weights)):
        one_layer = []
        for w in weights[i]:
            if i == 0:
                node = Node(w, func.Relu, 0.5)
            else:
                node = Node(w, func.Relu, None)
            one_layer.append(node)
        layer = Layer(one_layer, i)
        layers_all_nodes.append(layer)
    one_layer = []
    for i in range(structure[-1]):
        node = Node(None, None, None)
        one_layer.append(node)
    layer = Layer(one_layer, len(structure)-1)
    layers_all_nodes.append(layer)
    neural.set_layers(layers_all_nodes)
    return neural

def add_line(ar):
    with open("dataset/pixels.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(ar) 
        f.close()
    pass

def erase_dataset():
    with open("dataset/pixels.csv", "w") as f:
        f.close()
    pass

func = Functions()
deriv = Derivatives()
yolo = Yolo()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Can't receive frame (stream end?). Exiting ...")
#         break
#     if cv.waitKey(1) == ord('q'):
#         break
#     if cv.waitKey(1) == ord('m'):
#         img = Image.fromarray(frame)
#         image_resize = img.resize((256,256))
#         img_pixels_resize = np.asarray(image_resize)
#         pool = yolo.execute_yolo(img_pixels_resize)
#         cv.imshow('frame', pool)
#         time.sleep(2)
#     cv.imshow('frame', frame)

# cap.release()
# cv.destroyAllWindows()

n_images = 50
num_image = 1
# nn = initialise_neural()
# for i in range(4):
#     print(nn.layers[i].nodes)
for i in range(3, 18):
    get_images(n_images, i)

