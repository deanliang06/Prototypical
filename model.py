import random
import math
import json
import sys

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

def maxPoolKernalGrad(x, y, length, input, grad, matrix):
    maxIn = float('-inf')
    for i in range(length):
        for j in range(length):
            if x+i >= len(input) or y+j >=len(input[0]): continue
            maxIn = max(input[x + i][y + j], maxIn)
    
    for i in range(length):
        for j in range(length):
            if x+i >= len(input) or y+j >=len(input[0]): continue
            if input[x + i][y + j] == maxIn:
                matrix[x+i][y+j] = grad
            else:
                matrix[x+i][y+j] = 0


    
    


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
def sq_euclidean_distance(vec1, vec2):
    return euclidean_distance(vec1, vec2)**2

def scalar_to_vec(scalar, vec):
    newVec = []
    for i in range(len(vec)):
        newVec.append(vec[i] * scalar)
    return newVec

def add_component_wise(vec1, vec2):
    if len(vec2) == 0: return vec1
    new = []
    for i in range(len(vec1)):
        new.append(vec1[i] + vec2[i])
    return new

def regularize_embedding(vec):
    len = math.sqrt(sum(x*x for x in vec))
    return [x/len for x in vec]

class Model:
    
    def __init__(self, learning_rate):
        self.conv1 = []
        self.conv1_input = []
        self.conv2 = []
        self.conv2_input = []
        self.conv2_output = []
        self.fc1_input = []
        self.fc1 = []
        self.fc1_bias = []
        self.fc1_output = []
        self.fc2_input = []
        self.fc2 = []
        self.fc2_bias = []

        self.conv1_output_grad = []
        self.fc1_output_grad = []
        self.max_pool_2_output_grad = []

        self.img_loss_contribution = 0
        self.LEARNING_RATE = learning_rate

        self.initialize()

    def init_fully_connected(self, input_dim, output_dim):
        newArray = []
        for i in range(output_dim):
            newArray.append([random.gauss(0, 0.05) for _ in range(input_dim)])
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
        self.fc1_bias = [random.gauss(0, 0.05) for _ in range(256)]
        self.fc2_bias = [random.gauss(0, 0.05) for _ in range(64)]
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
        newConv = []
        for input_chan in range(len(input)):
            chanel = []
            for row in range (len(input[0])):
                newRow = []
                for el in range(len(input[0][0])):
                    val = input[input_chan][row][el]
                    if val < 0: newRow.append(0)
                    else: newRow.append(val)
                chanel.append(newRow)
            newConv.append(chanel)
        return newConv
    
    def relu_vec(self, vector):
        newVector = []
        for i in range(len(vector)):
            if vector[i] < 0:
                newVector.append(0)
            else:
                newVector.append(vector[i])
        return newVector
    
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
    
    def compute_loss(self, prediction, target, means):
        d_target = sq_euclidean_distance(target, prediction)

        dists = []
        for value in means.values():
            dists.append(-1* sq_euclidean_distance(value, prediction))

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
        temperature = 500
        sum = 0
        for value in means.values():
            sum+=math.exp(-1/temperature* euclidean_distance(value, prediction))

        for key,value in means.items():
            if key == target_class:
                if len(dLdv) == 0: dLdv = add_component_wise(scalar_to_vec(2*(math.exp((-1/temperature* euclidean_distance(value, prediction))/sum - 1)),(sub_component_wise(prediction, value))), dLdv)
                else: dLdv= add_component_wise(scalar_to_vec(2*(math.exp((-1/temperature* euclidean_distance(value, prediction))/sum - 1)),(sub_component_wise(prediction, value))), dLdv)
            else:
                if len(dLdv) == 0: dLdv = add_component_wise(scalar_to_vec(2*(math.exp(-1/temperature* euclidean_distance(value, prediction))/sum),(sub_component_wise(prediction, value))), dLdv)
                else: dLdv= add_component_wise(scalar_to_vec(2*(math.exp(-1/temperature* euclidean_distance(value, prediction))/sum),(sub_component_wise(prediction, value))), dLdv)
        return dLdv
    
    def apply_relu3_grad(self, relu_output_grad):
        for k, i in enumerate(self.fc1_output):
            if i < 0:
                relu_output_grad[k] = 0
            
    def apply_fc1_grad(self, dLdv):
        self.flatten_output_grad = []
        for k in range(len(self.fc1[0])):
            sum = 0.0
            for i,el in enumerate(dLdv):
                sum+=el*self.fc1[i][k]
            self.flatten_output_grad.append(sum)


        for i in range(len(self.fc1_bias)):
            self.fc1_bias[i]-=self.LEARNING_RATE*dLdv[i]
        
        for i,row in enumerate(self.fc1):
            for j, el in enumerate(row):
                self.fc1[i][j]-=self.LEARNING_RATE*dLdv[i]*self.fc1_input[j]

    def apply_reverse_flatten(self, channel, dim, flattened):
        output = []
        for i in range(channel):
            matrix = []
            for j in range(dim):
                row = []
                for k in range(dim):
                    row.append(flattened[(i)*dim*dim +(j)*dim + (k)])
                matrix.append(row)
            output.append(matrix)
        self.max_pool_2_output_grad = output


    def apply_max_pool2_grad(self, output_grad):
        #assumed 2x2 maxpool with stride 2
        input = self.max_pool_2_input
        output = []
        for input_chan in range(len(input)):
            matrix = self.init_fully_connected(len(input[0]), len(input[0]))
            for rowNum in range (math.ceil(len(input[0]) / 2)):#implementing stride 2
                for el in range(math.ceil(len(input[0][0]) / 2)):
                    maxPoolKernalGrad(rowNum* 2, el* 2, 2, input[input_chan], output_grad[input_chan][rowNum][el], matrix)
            output.append(matrix)
        self.conv2_output_grad = output

    def apply_relu_grad(self, output, input):
        for i in range(len(output)):
            for j in range(len(output[0])):
                for k in range(len(output[0][0])):
                    if input[i][j][k] < 0:
                        output[i][j][k] = 0
    def calculate_input_grad(self, kernal, filter,x, y, output_grad):
        sum = 0
        for k, filt in enumerate(kernal):
            act_kernal = filt["filter"]
            correct_kernal = act_kernal[filter]
            correct_grad_matrix = output_grad[k]
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if x+i < 0 or x+i >= len(correct_grad_matrix) or y+j < 0 or y+j >= len(correct_grad_matrix[0]): continue
                    sum+=correct_grad_matrix[x+i][y+j]*correct_kernal[1-i][1-j]
        return sum

    def calculate_kernal_grad(self, output_num, x, y, input, output_grad):
        output_grad_simplied = output_grad[output_num]
        sum = 0
        for i in range(len(input)):#because if x == 2 then we don't do the first row and other extrapolated facts
            for j in range(len(input[0])):
                if i + x - 1 < 0 or i + x - 1 >= len(input) or j + y - 1 < 0 or j + y - 1 >= len(input[0]): continue
                sum+=output_grad_simplied[i][j]*input[i + x - 1][j + y - 1]
        return sum
                
    def apply_conv2_grad(self, input, output_grad):
        #calcualte the ouput_grad of last layer first
        for f, filter in enumerate(input):
            filter_grad = []
            for i, row in enumerate(filter):
                newRow = []
                for j in range(len(row)):
                    newRow.append(self.calculate_input_grad(self.conv2, f, i, j, output_grad))
                filter_grad.append(newRow)
            self.conv1_output_grad.append(filter_grad)


        for o, filter in enumerate(self.conv2):
            kernal = filter["filter"]

            #update bias
            bias_grad = 0
            for row in output_grad[o]:
                for el in row:
                    bias_grad+=el
            filter["bias"]-=self.LEARNING_RATE*bias_grad

            #update kernal vals
            for f, input_filter in enumerate(kernal):
                for i, row in enumerate(input_filter):
                    for j, el in enumerate(row):
                        filter["filter"][f][i][j]-=self.LEARNING_RATE* self.calculate_kernal_grad(o, i, j, input[f], output_grad)



    def apply_max_pool1_grad(self, output_grad):
        #assumed 2x2 maxpool with stride 2
        input = self.max_pool1_input
        output = []
        for input_chan in range(len(input)):
            matrix = self.init_fully_connected(len(input[0]), len(input[0]))
            for rowNum in range (math.ceil(len(input[0]) / 2)):#implementing stride 2
                for el in range(math.ceil(len(input[0][0]) / 2)):
                    maxPoolKernalGrad(rowNum* 2, el* 2, 2, input[input_chan], output_grad[input_chan][rowNum][el], matrix)
            output.append(matrix)
        self.conv1_output_grad = output

    def apply_conv1_grad(self, input, output_grad):
        for o, filter in enumerate(self.conv1):
            kernal = filter["filter"]

            #update bias
            bias_grad = 0
            for row in output_grad[o]:
                for el in row:
                    bias_grad+=el
            filter["bias"]-=self.LEARNING_RATE*bias_grad

            #update kernal vals
            for f, input_filter in enumerate(kernal):
                for i, row in enumerate(input_filter):
                    for j, el in enumerate(row):
                        filter["filter"][f][i][j]-=self.LEARNING_RATE* self.calculate_kernal_grad(o, i, j, input, output_grad)

    def zero_grad(self):
        self.conv2_output_grad = []
        self.conv1_output_grad = []
        self.max_pool_2_output_grad = []
        self.fc1_output_grad = []

    def save(self):
        output = {
            "conv1": self.conv1,
            "conv2": self.conv2,
            "fc1": self.fc1,
            "fc1_bias": self.fc1_bias,
            "fc2": self.fc2,
            "fc2_bias": self.fc2_bias,
        }
        
        with open("model.txt", 'w') as f:
            json.dump(output, f)

    def apply_gradients(self, means, prediction, target_class):
        dLdv = self.output_grad(means, prediction, target_class)
        print(self.fc2[0][0])
        self.apply_fc2_grad(dLdv)
        print(self.fc2[0][0])
        self.apply_relu3_grad(self.fc1_output_grad)
        self.apply_fc1_grad(self.fc1_output_grad)
        self.apply_reverse_flatten(16, 7, self.flatten_output_grad)
        self.apply_max_pool2_grad(self.max_pool_2_output_grad)
        self.apply_relu_grad(self.conv2_output_grad, self.conv2_output)
        self.apply_conv2_grad(self.conv2_input, self.conv2_output_grad)
        self.apply_max_pool1_grad(self.conv1_output_grad)
        self.apply_relu_grad(self.conv1_output_grad, self.conv1_output)
        self.apply_conv1_grad(self.conv1_input,self.conv1_output_grad)
        self.zero_grad()
        self.save()

    def evaluate(self, input):
        #ik its not the most robust but... I have to hand calcualte gradients
        #I don't really know how to do that but okay vro
        self.conv1_input = input
        conv1_output = self.evaluateConv(self.conv1, input)
        self.conv1_output = conv1_output
        relu_output = self.relu_conv(conv1_output)
        self.max_pool1_input = relu_output
        max_pool_output = self.max_pool(2, relu_output)
        self.conv2_input = max_pool_output
        self.conv2_output = self.evaluateConv(self.conv2, max_pool_output)
        relu2_output = self.relu_conv(self.conv2_output)
        self.max_pool_2_input = relu2_output
        max_pool2_output = self.max_pool(2, relu2_output)
        result = self.flatten(max_pool2_output)
        self.fc1_input = result
        fc1_result = self.evaluate_fc(self.fc1, self.fc1_bias, result)
        self.fc1_output = fc1_result
        fc1_relu = self.relu_vec(fc1_result)
        self.fc2_input = fc1_relu
        fc2_result = self.evaluate_fc(self.fc2, self.fc2_bias, fc1_relu)
        return fc2_result

        

    
        