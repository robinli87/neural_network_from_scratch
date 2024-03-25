#main neural network, barebone, no optimisation, importable.
import random
import cmath
import math
import multiprocessing as mp

class NN:
    def __init__(self, structure):
        self.structure = structure    #wanted to keep a global storage of the self.structure but seems unecessary for now
        self.stop = False
        # self.train_inputs = train_inputs
        # self.train_outputs = train_outputs

    def pause(self):
        self.stop = True

    def run(self, input_vector, weights, biases):
        #runs the network with the given weights, biases and inputs. Doesn't care about outside or big picture
        #firstly build / clear the nodes
        z, a = self.build_nodes()

        #firstly set 0th z layer and a layer to be the inputs; inputs can be either the training data or actual exam data
        #sanity check: we expect the input vector to have the same size as the input layer
        if self.structure[0] != len(input_vector):
            print("input vector is of the wrong size. Expecting length ", self.structure[0])

        for k in range(0, self.structure[0]):
            z[0][k] = input_vector[k]
            a[0][k] = input_vector[k]
            #k is for specifying the vertical location within this current layer; j is for the previous layer

        #now we propagate forward with our given weights and biases
        for l in range(1, len(self.structure)):
            #we start at one because layer 0 was the inputs
            #self.structure[l] gives the size of this layer
            for k in range(0, self.structure[l]):
                total = biases[l-1][k]
                for j in range(0, self.structure[l-1]):
                    #j considers the weights connecting from the previous layer, so we run to l-1
                    total = total + weights[l-1][j][k] * a[l-1][j]

                #now update z according to this total.
                z[l][k] = total
                a[l][k] = self.activate(total, self.activation_type)

        #we have filled out all of the nodes. Extract the last layer and return it as output
        #print(z)
        #print(a)
        return(z[-1]) #we are returning a vector.

    def train(self, training_inputs, correct_outputs, activation_type="leakyRELU", cost_type=2):
        self.training_inputs = training_inputs
        self.correct_outputs = correct_outputs
        self.activation_type = activation_type
        self.cost_type = cost_type
        self.learning_rate = 0.000001
        #dry run once and evaluate loss
        self.w, self.b = self.build_parameters()
        #print(w)

        self.w_modifiers = self.build_weights(0)
        self.b_modifiers = self.build_biases()

        self.layer_number = len(self.structure)-1

        benchmark = self.loss(self.w, self.b)
        print("loss: ", benchmark)

        tolerance = 0.1
        self.dw = 0.00000001
        self.db = 0.00000001

        #need to write training process here:

        #first loop:
        epoch = 0
        self.backpropagation(self.training_inputs, self.correct_outputs)
        new_loss = self.loss(self.w, self.b)
        print("New_loss: ", new_loss)
        file = open("history.csv", "w")
        file.write(str(epoch)+","+str(new_loss)+"\n")
        file.close()
        print("New_loss: ", new_loss)

        while (new_loss < benchmark) or (new_loss > tolerance):
        #for cycles in range(10):
            benchmark = new_loss
            epoch = epoch + 1
            self.backpropagation(self.training_inputs, self.correct_outputs)
            new_loss = self.loss(self.w, self.b)
            file = open("history.csv", "a")
            file.write(str(epoch)+","+str(new_loss)+"\n")
            file.close()
            print("New loss: ", new_loss)

            if self.stop == True:
                break
            #print("Sample w[0][0] = ", w[1][1])


        #need backpropagation function

        #after training, we want to return the ideal parameters
        return(self.w, self.b)

    def pack_parameters(self, l, k):
        # we can't make changes here so meh, we pack up parameters and parse them into the callback function'
        #print("thread begins")

        #now we must construct completely decoupled arrays. copy() is not enough
        upper_b = []
        lower_b = []
        for p in range(0, self.layer_number):
            this_row = []
            for q in range(0, self.structure[p+1]):
                this_component = self.b[p][q]
                this_row.append(this_component)
            upper_b.append(this_row)

        for p in range(0, self.layer_number):
            this_row = []
            for q in range(0, self.structure[p+1]):
                this_component = self.b[p][q]
                this_row.append(this_component)
            lower_b.append(this_row)

        #differentiate wrt b
        upper_b[l][k] += self.db
        lower_b[l][k] -= self.db
        upper = self.loss(self.w, upper_b)
        lower = self.loss(self.w, lower_b)
        b_gradient = (upper - lower) / (2 * self.db)
        # if b_gradient == 0:
        #     print("0 gradient error!!")


        w_gradients = []

        #construct completely decoupled arrays
        for j in range(0, self.structure[l]): #iterate though before column
            #now individual components of weights
            #let's do for example a simple gradient descent'
            upper_w = []
            lower_w = []
            #construction of completely decoupled arrays
            for p in range(0, self.layer_number):
                this_dyad = []
                for q in range(0, self.structure[p]): #iterate though before column
                    this_vector = []
                    for r in range(0, self.structure[p+1]):
                        this_component = self.w[p][q][r]
                        this_vector.append(this_component)
                    this_dyad.append(this_vector)
                upper_w.append(this_dyad)

            for p in range(0, self.layer_number):
                this_dyad = []
                for q in range(0, self.structure[p]): #iterate though before column
                    this_vector = []
                    for r in range(0, self.structure[p+1]):
                        this_component = self.w[p][q][r]
                        this_vector.append(this_component)
                    this_dyad.append(this_vector)
                lower_w.append(this_dyad)

            #now we are ready for differentiate wrt w element wise
            upper_w[l][j][k] = upper_w[l][j][k] + self.dw
            lower_w[l][j][k] = lower_w[l][j][k] - self.dw
            upper = self.loss(upper_w, self.b)
            lower = self.loss(lower_w, self.b)
            w_gradient = (upper - lower) / (2 * self.dw)
            #print(w_gradient)
            w_gradients.append(w_gradient)

        parameters = [l, k, b_gradient, w_gradients]

        return(parameters)

    def real_calculations(self, implicit_parameters):
        #implicit_parameters is the packed parameters from above. now we unpack them
        l = implicit_parameters[0]
        k = implicit_parameters[1]
        b_gradient = implicit_parameters[2]
        w_gradients = implicit_parameters[3]

        self.b_modifiers[l][k] = -self.learning_rate  * b_gradient
        self.b[l][k] += self.b_modifiers[l][k]

        for j in range(0, self.structure[l]):
            self.w_modifiers[l][j][k] = -self.learning_rate * w_gradients[j]
            self.w[l][j][k] += self.w_modifiers[l][j][k]

        'do calculation as normal'


    def backpropagation(self, training_inputs, correct_outputs):
        #intakes inputs, outputs, weights and biases, calculates improved weights and biases then return the improved parameters.

        #evaluate some constants to save work:
        #l range: 0<=l<=len(structure)-1
        #j <= self.structure[l]
        #k <= self.structure[l+1]

        #now calculate b modifiers
        pool = mp.Pool()

        for l in range(0, self.layer_number):

            for k in range(0, self.structure[l+1]):

                pool.apply_async(self.pack_parameters, args=(l, k), callback=self.real_calculations)
        pool.close()
        pool.join()
                #now we must construct completely decoupled arrays. copy() is not enough
                # upper_b = []
                # lower_b = []
                # for p in range(0, len(b)):
                #     this_row = []
                #     for q in range(0, len(b[p])):
                #         this_component = b[p][q]
                #         this_row.append(this_component)
                #     upper_b.append(this_row)
                #
                # for p in range(0, len(b)):
                #     this_row = []
                #     for q in range(0, len(b[p])):
                #         this_component = b[p][q]
                #         this_row.append(this_component)
                #     lower_b.append(this_row)
                #
                # upper_b[l][k] += db
                # lower_b[l][k] -= db




        #now calculate the modfiers for every parameter, separately
        # for l in range(0, layer_number):
        #     for k in range(0, self.structure[l+1]):


        #we have got the modifiers, now we can apply the changes altogether
        # for l in range(0, len(self.w_modifiers)): #iterate through all components. l is layers
        #     for j in range(0, len(self.w_modifiers[l])): #iterate though before column
        #         for k in range(0, len(self.w_modifiers[l][j])): #now individual components of weights
        #
        #
        # for l in range(0, len(self.b_modifiers)):
        #     for k in range(0, len(self.b_modifiers[l])):



    def loss(self, weights, biases):
        ai_outputs = []
        #we iterate through all given inputs and ask the AI to give an output for each
        for i in range(0, len(self.training_inputs)):
            # we expect training inputs to be a rectangular matrix containing all input feature vectors.
            this_output = self.run(self.training_inputs[i], weights, biases)
            ai_outputs.append(this_output)

        #first sanity check:
        if len(ai_outputs) != len(self.correct_outputs):
            print("input sizes are different, check sample, completion and transpose")

        total_loss = self.costfunction(ai_outputs, self.correct_outputs, self.cost_type)
        return(total_loss)

    def costfunction(self, prediction, correct, type=2):
        #let's go for the mean squared error'
        total_error = 0

        for i in range(0, len(prediction)):
            #iterating through all produced results
            this_error = 0
            for k in range(0, self.structure[-1]):
                #now we are down to individual components of each feature vector output
                this_error += (((prediction[i][k] - correct[i][k])**2)**0.5) / len(prediction)
                #we are comparing and adding errors component-wise. we can try ^4 or ^ any even power.
            total_error += this_error

        return(total_error)

    def activate(self, x, type):
        #choose tanh function for now
        #f = math.tanh(x)

        #leaky relu
        if type == "leakyRELU":
            f = 0
            if x < 0:
                f = -0.01 * x
            else:
                f = x
            return(f)

        if type == "tanh":
            return(math.tanh(x))

        if type == "sigmoid":
            f = 1 / (1 + math.e ** -x)
            return(f)

        if type == "swish":
            beta = 1
            f = x / (1 + math.e ** (-x * beta))
            return(f)

    def build_parameters(self):
        # first build the weights, which is a triad
        w = []

        b = [] #biases, but we allow different bias values within the same layer so it is dyad now

        for l in range(0, len(self.structure) - 1):
            #generate dyad of connections. dimensions = this x next
            rows = self.structure[l]
            cols = self.structure[l+1]
            this_dyad = []

            for j in range(0, rows):

                this_vector = []

                for k in range(0, cols):
                    variance = 4 / (self.structure[0] + self.structure[-1])
                    this_vector.append(random.gauss(0, 2))
                this_dyad.append(this_vector)

            w.append(this_dyad)

        #now build the b, a, z layers. They take the same shape.


        for l in range(1, len(self.structure)):
            #b starts after the inputs, between input and first hidden
            this_col = []
            for k in range(0, self.structure[l]):
                this_col.append(0)
            b.append(this_col)

        return(w,b) #now we return these values back to whoever called us.

    def build_nodes(self):
        z = [] #the temporary values stored in each node
        a = [] #the activated values in each node

        #must be made separately to prevent sticky coupling
        for l in range(0, len(self.structure)):
            #generate col vector
            this_col = []
            for k in range(0, self.structure[l]):
                this_col.append(0)
            z.append(this_col)

        for l in range(0, len(self.structure)):
            #generate col vector
            this_col = []
            for k in range(0, self.structure[l]):
                this_col.append(0)
            a.append(this_col)

        return(z, a)

    def build_weights(self, fill="random"):
        # first build the weights, which is a triad
        w = []

        for l in range(0, len(self.structure) - 1):
            #generate dyad of connections. dimensions = this x next
            rows = self.structure[l]
            cols = self.structure[l+1]
            this_dyad = []

            for j in range(0, rows):

                this_vector = []

                for k in range(0, cols):
                    #what initial values should we fill the grid with?
                    if fill == "random":
                        variance = 4 / (self.structure[0] + self.structure[-1])
                        this_vector.append(random.gauss(0, variance))
                    elif fill == 0:
                        this_vector.append(0)
                this_dyad.append(this_vector)

            w.append(this_dyad)

        #now build the b, a, z layers. They take the same shape.
        return(w)

    def build_biases(self):
        b = [] #biases, but we allow different bias values within the same layer so it is dyad now

        for l in range(1, len(self.structure)):
            #b starts after the inputs, between input and first hidden
            this_col = []
            for k in range(0, self.structure[l]):
                this_col.append(0)
            b.append(this_col)

        return(b) #now we return these values back to whoever called us.

