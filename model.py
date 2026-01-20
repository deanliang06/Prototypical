import random
import math

def kernalMultiWInputChannelNot1(x, y, kernal, bias, input):
    displacement = (len(kernal) - 1) // 2 #assuming square and odd *rolling eyes
    sum = 0
    for k, input_chan in enumerate(input):
        for i in range(len(kernal[0])):
            for j in range(len(kernal[0][0])):
                #custom padding i guess (i am going to not do positive things to myself)
                if x - displacement + i < 0 or x - displacement + i >= len(input_chan[0]):
                    continue
                if y - displacement + j < 0 or y - displacement + j >= len(input_chan):
                    continue

                sum+=kernal[k][i][j]*input_chan[x - displacement + i][y - displacement + j]
    return sum + bias

def maxPoolKernal(x, y, length, input):
    initialDisplacement = (length - 1) // 2
    maxIn = float('-inf')
    for i in range(length):
        for j in range(length):
            maxIn = max(input[int(x + i - initialDisplacement)][int(y + j - initialDisplacement)], maxIn)
    return maxIn

def sub_component_wise(vec1, vec2):
    if len(vec2) == 0: return vec1
    new = []
    for i in range(len(vec1)):
        new.append(vec1[i] - vec2[i])
    return new

def euclidean_distance(vec1, vec2):
    sum_of_squares = 0
    for i in range(len(vec1)):
        sum_of_squares+=(vec1[i]-vec2[i])**2
    return sum_of_squares**(1/2)

def scalar_to_vec(scalar, vec):
    for i in range(len(vec)):
        vec[i]*=scalar
    return vec

def add_component_wise(vec1, vec2):
    if len(vec2) == 0: return vec1
    new = []
    for i in range(len(vec1)):
        new.append(vec1[i] + vec2[i])
    return new


class Model:
    
    def __init__(self, learning_rate):
        self.conv1 = []
        self.conv1_input = []
        self.conv2 = []
        self.conv2_input = []
        self.fc1_input = []
        self.fc1 = []
        self.fc1_bias = []
        self.fc2_input = []
        self.fc2 = []
        self.fc2_bias = []


        self.fc1_output_grad = []


        self.img_loss_contribution = 0
        self.LEARNING_RATE = learning_rate

        self.initialize()

    def init_fully_connected(self, input_dim, output_dim):
        newArray = []
        for i in range(output_dim):
            newArray.append([random.gauss(0, 0.005) for _ in range(input_dim)])
        return newArray

    def init_conv(self, conv, input_chan, output_chan):
        for i in range(output_chan):
            filter = []
            for j in range(input_chan):
                filter.append(self.init_fully_connected(3, 3))
            conv.append({"filter": filter, "bias": random.gauss(0, 0.005)})

    def initialize(self):
        self.fc1 = self.init_fully_connected(784, 256)
        self.fc2 = self.init_fully_connected(256, 64)
        self.fc1_bias = [random.gauss(0, 0.005) for _ in range(256)]
        self.fc2_bias = [random.gauss(0, 0.005) for _ in range(64)]
        self.init_conv(self.conv1, 1, 8)
        self.init_conv(self.conv2, 8, 16)

    #def train(self, loss): 
    def evaluateConv(self, conv, input):
        output = []
        if len(input) == 28: input = [input]
        for filterIsh in conv:
            kernal = filterIsh["filter"]
            bias = filterIsh["bias"]
            filterOutput = []
            for i, row in enumerate(input[0]):
                newRow = []
                for j in range(len(row)):
                    newRow.append(kernalMultiWInputChannelNot1(i, j, kernal, bias, input))
                filterOutput.append(newRow)
            output.append(filterOutput)
        return output

    def relu_conv(self, input):
        for input_chan in range(len(input)):
            for row in range (len(input[0])):
                for el in range(len(input[0][0])):
                    val = input[input_chan][row][el]
                    if val < 0: 
                        input[input_chan][row][el] = 0
        return input
    
    def relu_vec(self, vector):
        for i in range(len(vector)):
            if vector[i] < 0:
                vector[i] = 0
        return vector
    
    #assuming square
    def max_pool(self, length, input):  
        output = []    
        for input_chan in range(len(input)):
            matrix = []
            for rowNum in range (len(input[0]) // 2):#implementing stride 2
                row = []
                for el in range(len(input[0][0]) // 2):
                    row.append(maxPoolKernal(rowNum*2, el*2, length, input[input_chan]))
                matrix.append(row)
            output.append(matrix)
        return output
    
    def evaluate_fc(self, layer, bias, vector):
        def dot_product(vec1, vec2):
            dot_prod_res = 0
            for i in range(len(vec1)):
                dot_prod_res += vec1[i] * vec2[i]
            return dot_prod_res
        

        final = []

        for i, row in enumerate(layer):
            num = dot_product(row, vector)
            final.append(bias[i] + num)
        return final
    
    def flatten(self, array):
        final = []
        for filter in array:
            for row in filter:
                for el in row:
                    final.append(el)
        return final
    
    def compute_loss(self, prediction, target, img_object, means):
        d_target = euclidean_distance(target, prediction)

        dists = []
        for value in means.values():
            dists.append(-1* euclidean_distance(value, prediction))

        max_val = max(dists)   
        log_sum = max_val + math.log(sum(math.exp(v - max_val) for v in dists))

        self.img_loss_contribution = (-d_target - log_sum) / 5
        return self.img_loss_contribution

    def apply_fc2_grad(self, dLdv):
        self.fc1_output_grad = []
        for k in range(len(self.fc2[0])):
            sum = 0.0
            for i,el in enumerate(dLdv):
                sum+=el*self.fc2[i][k]
            self.fc1_output_grad.append(sum)


        for i in range(len(self.fc2_bias)):
            self.fc2_bias[i]-=self.LEARNING_RATE*dLdv[i]
        
        for i,row in enumerate(self.fc2):
            for j, el in enumerate(row):
                self.fc2[i][j]-=self.LEARNING_RATE*dLdv[i]*self.fc2_input[j]



    def output_grad(self, means, prediction, target_class):
        dLdv = []
        sum = 0
        for value in means.values():
            sum+=math.exp(-1* euclidean_distance(value, prediction))

        for key,value in means.items():
            if key == target_class:
                if len(dLdv) == 0: dLdv = add_component_wise(scalar_to_vec(2*(math.exp(-1* euclidean_distance(value, prediction))/sum - 1),(sub_component_wise(prediction, value))), dLdv)
                dLdv= add_component_wise(scalar_to_vec(2*(math.exp(-1* euclidean_distance(value, prediction))/sum - 1),(sub_component_wise(prediction, value))), dLdv)
            else:
                if len(dLdv) == 0: dLdv = add_component_wise(scalar_to_vec(2*(math.exp(-1* euclidean_distance(value, prediction))/sum),(sub_component_wise(prediction, value))), dLdv)
                dLdv= add_component_wise(scalar_to_vec(2*(math.exp(-1* euclidean_distance(value, prediction))/sum),(sub_component_wise(prediction, value))), dLdv)
        return dLdv

    def apply_gradients(self, means, prediction, target_class):
        dLdv = self.output_grad(means, prediction, target_class)
        self.apply_fc2_grad(dLdv)


    def evaluate(self, input):
        #ik its not the most robust but... I have to hand calcualte gradients
        self.conv1_input = input
        conv1_output = self.evaluateConv(self.conv1, input)
        relu_output = self.relu_conv(conv1_output)
        max_pool_output = self.max_pool(2, relu_output)
        self.conv2_input = max_pool_output
        conv2_output = self.evaluateConv(self.conv2, max_pool_output)
        relu2_output = self.relu_conv(conv2_output)
        max_pool2_output = self.max_pool(2, relu2_output)
        result = self.flatten(max_pool2_output)
        self.fc1_input = result
        fc1_result = self.evaluate_fc(self.fc1, self.fc1_bias, result)
        fc1_relu = self.relu_vec(fc1_result)
        self.fc2_input = fc1_relu
        fc2_result = self.evaluate_fc(self.fc2, self.fc2_bias, fc1_relu)
        return fc2_result

        

    
        