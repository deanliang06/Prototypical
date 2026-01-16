import random


class Model:
    
    def __init__(self):
        self.conv1 = []
        self.conv2 = []
        self.fc1 = []
        self.fc2 = []
        self.initialize()

    def init_fully_connected(self, array, input_dim, output_dim):
        for i in range(output_dim):
            array.append([random.gauss(0, 0.5) for _ in range(input_dim)])

    def init_conv(self, conv, input)

    def initialize(self):
        self.init_fully_connected(self.fc1, 784, 256)
        self.init_fully_connected(self.fc2, 256, 64)

    
