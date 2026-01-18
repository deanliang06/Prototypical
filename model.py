import random
import math

def kernalMulti(x, y, kernal, bias, input):
    displacement = (len(kernal) - 1) // 2 #assuming square and odd *rolling eyes
    sum = 0
    for i in range(len(kernal)):
        for j in range(len(kernal)):
            #custom padding i guess (i am going to not do positive things to myself)
            if x - displacement + i < 0 or x - displacement + i >= len(input[0]):
                continue
            if y - displacement + j < 0 or y - displacement + j >= len(input):
                continue

            sum+=kernal[i][j]*input[x - displacement + i][y - displacement + j]
    return sum + bias

def kernalMultiWInputChannelNot1(x, y, kernal, bias, input):
    displacement = (len(kernal) - 1) // 2 #assuming square and odd *rolling eyes
    sum = 0
    for input_chan in input:
        for i in range(len(kernal)):
            for j in range(len(kernal)):
                #custom padding i guess (i am going to not do positive things to myself)
                if x - displacement + i < 0 or x - displacement + i >= len(input_chan[0]):
                    continue
                if y - displacement + j < 0 or y - displacement + j >= len(input_chan):
                    continue

                sum+=kernal[i][j]*input_chan[x - displacement + i][y - displacement + j]
    return sum + bias

def maxPoolKernal(x, y, len, input):
    initialDisplacement = (len - 1) / 2
    max = float('-inf')
    for i in range(len):
        for j in range(len):
            max = math.max(input[x + i - initialDisplacement][y + j - initialDisplacement])
    return max

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

    #def train(self, loss): 
    def evaluateConv(self, conv, input):
        output = []
        for filterIsh in conv:
            kernal = filterIsh["filter"]
            bias = filterIsh["bias"]
            filterOutput = []
            print(len(input))
            #one input dim (very simple) kinda hacky as well because we know if len is 28 then 1 dim but
            if len(input) == 28:
                for i, row in enumerate(input):
                    newRow = []
                    for j in range(len(row)):
                        newRow.append(kernalMulti(i, j, kernal, bias, input))
                    filterOutput.append(newRow)
            else:
                print(input)
                for i, row in enumerate(input[0]):
                    newRow = []
                    for j in range(len(row)):
                        newRow.append(kernalMultiWInputChannelNot1(i, j, kernal, bias, input))
                    filterOutput.append(newRow)
            output.append(filterOutput)

    def relu(self, input):
        for input_chan in range(len(input)):
            for row in range (len(input_chan)):
                for el in range(len(row)):
                    val = input[input_chan][row][el]
                    if val < 0: 
                        input[input_chan][row][el] = 0
        return input
    
    #assuming square
    def max_pool(self, len, input):  
        output = []          
        for input_chan in range(len(input)):
            matrix = []
            for row in range (len(input_chan) / 2):#implementing stride 2
                row = []
                for el in range(len(row) / 2):
                    row.append(maxPoolKernal(row*2, el*2, len, input[input_chan]))
                matrix.append(row)
            output.append(matrix)
        return output


    def evaluate(self, input):
        #ik its not the most robust but...
        #conv1
        conv1_output = self.evaluateConv(self.conv1, input)
        relu_output = self.relu(conv1_output)
        max_pool_output = self.max_pool(relu_output)
        return max_pool_output #to test

    
        