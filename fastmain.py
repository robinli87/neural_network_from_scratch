import turboNN1 as NN

AI = NN.NN([1,3, 5, 7, 9,9, 7,5,3,1])

training_inputs = []
training_outputs = []

#let's generate some datapoints
for i in range(0, 200):
    x = (i-100)/100
    input_vector = [x]
    training_inputs.append(input_vector)

    y = 2*x
    output_vector = [y]
    training_outputs.append(output_vector)

trained_weights, trained_biases = AI.train(training_inputs, training_outputs, "leakyRELU")
print("}==============================================={")
while True:
    num = float(input("Enter new number"))
    print(AI.run([num], trained_weights, trained_biases))
    print("}-----------------------------------------------------{")
