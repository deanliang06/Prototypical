import random


class Model:
    def __init__(self):
        self.conv1 = []
        self.conv2 = []
        self.fc1 = []
        self.fc1_bias = []
        self.fc2 = []
        self.fc1_bias = []

        self.initialize()

    def init_fully_connected(self, array, input_dim, output_dim):
        for i in range(output_dim):
            array.append([random.gauss(0, 0.5) for _ in range(input_dim)])

    def init_conv(self, conv, input_chan, output_chan):
        for i in range(output_chan):
            filter = []
            for j in range(input_chan):
                self.init_fully_connected(filter, 3, 3)
            conv.append({"filter": filter, "bias": random.gauss(0, 0.5)})

    def initialize(self):
        self.init_fully_connected(self.fc1, 784, 256)
        self.init_fully_connected(self.fc2, 256, 64)
        self.fc1_bias = [random.gauss(0, 0.5) for _ in range(256)]
        self.fc2_bias = [random.gauss(0, 0.5) for _ in range(64)]
        self.init_conv(self.conv1, 1, 8)
        self.init_conv(self.conv2, 8, 16)


    
