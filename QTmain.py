from PyQt6.QtWidgets import *
from PyQt6.QtCore import Qt, QSize
from PyQt6 import uic, QtWidgets
import sys
import threading
import time
import os
import signal
import turboNN2 as NN
import numpy as np

class GUI(QMainWindow):

    def __init__(self):
        super().__init__()
        # self.master = master

        self.setWindowTitle("Neural Netowrk Control Window")
        self.resize(700, 700)
        self.wm = []
        for i in range(0, 10):
            row = [None, None, None, None]
            self.wm.append(row)


        self.layout = QGridLayout()
        self.frame = QWidget()
        self.frame.setLayout(self.layout)
        self.setCentralWidget(self.frame)

        #labels in the definitive group
        self.wm[0][0] = QLabel("Input Feature Vectors")
        self.layout.addWidget(self.wm[0][0], 0, 0)
        self.wm[1][0] = QLabel("Output Vectors")
        self.layout.addWidget(self.wm[1][0], 1, 0)
        self.wm[2][0] = QLabel("Network Structure")
        self.layout.addWidget(self.wm[2][0], 2, 0)
        self.wm[3][0] = QLabel("Activation Function")
        self.layout.addWidget(self.wm[3][0], 3, 0)
        self.wm[4][0] = QLabel("Loss Function Type")
        self.layout.addWidget(self.wm[4][0], 4, 0)

        #definitive entries
        self.wm[0][1] = QLineEdit()
        self.layout.addWidget(self.wm[0][1], 0, 1)
        self.wm[1][1] = QLineEdit()
        self.layout.addWidget(self.wm[1][1], 1, 1)
        self.wm[2][1] = QLineEdit()
        self.layout.addWidget(self.wm[2][1], 2, 1)
        self.wm[3][1] = QLineEdit()
        self.layout.addWidget(self.wm[3][1], 3, 1)
        self.wm[4][1] = QLineEdit()
        self.layout.addWidget(self.wm[4][1], 4, 1)



        #for i in range(0, 5):
        # now add buttons

        self.wm[5][0] = QPushButton("Start Training")
        self.wm[5][0].clicked.connect(self.start)
        self.layout.addWidget(self.wm[5][0], 5, 0)
        self.wm[6][0] = QPushButton("Pause")
        self.wm[6][0].clicked.connect(self.pause)
        self.layout.addWidget(self.wm[6][0], 6, 0)
        self.wm[7][0] = QPushButton("Run with current parameters")
        self.wm[7][0].clicked.connect(self.evaluate)
        self.layout.addWidget(self.wm[7][0], 7, 0)
        self.wm[5][1] = QPushButton("Plot Loss Graph")
        self.wm[5][1].clicked.connect(self.graph)
        self.layout.addWidget(self.wm[5][1], 5, 1)
        self.wm[6][1] = QPushButton("Resume")
        self.wm[6][1].clicked.connect(self.resume)
        self.layout.addWidget(self.wm[6][1], 6, 1)
        self.wm[7][1] = QPushButton("Killall")
        self.wm[7][1].clicked.connect(self.kill)
        self.layout.addWidget(self.wm[7][1], 7, 1)

        self.wm[5][0].setCheckable(True)
        self.wm[6][0].setCheckable(True)
        self.wm[7][0].setCheckable(True)
        self.wm[5][1].setCheckable(True)
        self.wm[6][1].setCheckable(True)
        self.wm[7][1].setCheckable(True)

        #labels in the supplementary group
        self.wm[5][2] = QLabel("Initial learning rate")
        self.layout.addWidget(self.wm[5][2], 5, 2)
        self.wm[6][2] = QLabel("Tolerance")
        self.layout.addWidget(self.wm[6][2], 6, 2)
        self.wm[7][2] = QLabel("Optimisation Algorithm")
        self.layout.addWidget(self.wm[7][2], 7, 2)
        self.wm[8][2] = QLabel("Gamma for RMSprop")
        self.layout.addWidget(self.wm[8][2], 8, 2)
        self.wm[9][2] = QLabel("alpha for momentum")
        self.layout.addWidget(self.wm[9][2], 9, 2)

        #supplementary entries
        self.wm[5][3] = QLineEdit()
        self.layout.addWidget(self.wm[5][3], 5, 3)
        self.wm[6][3] = QLineEdit()
        self.layout.addWidget(self.wm[6][3], 6, 3)
        self.wm[7][3] = QLineEdit()
        self.layout.addWidget(self.wm[7][3], 7, 3)
        self.wm[8][3] = QLineEdit()
        self.layout.addWidget(self.wm[8][3], 8, 3)
        self.wm[9][3] = QLineEdit()
        self.layout.addWidget(self.wm[9][3], 9, 3)

        print(self.wm)


        # dropdown = QComboBox()
        # dropdown.addItem("Inverse Square Mass")
        # dropdown.addItem("Lennard Jones")
        # dropdown.addItem("Inverse Square Charge")
        # self.entries[0].append(dropdown)
        # self.layout.addWidget(dropdown, 4, 1)

    def extract_inputs(self):
        input_filemame = self.wm[0][1].text()
        output_filename = self.wm[1][1].text()
        #print(output_filename)

        structure = str(self.wm[2][1].text())
        #print(structure)
        layers = structure.split("/")
        self.structure = []
        for i in range(0, len(layers)):
            self.structure.append(int(layers[i]))


    def start(self):
        self.extract_inputs()

        with open("command_injection.dat", "w") as f:
            f.write("1")

        threading.Thread(target=self.train).start()

    def train(self):

        #initialise the class
        self.model_inputs = []
        self.model_outputs = []

        #let's do 200 points between -1 <= x <= 1
        #let the model function be y=2x, we are taking 200 sample points on the straight line as sample data
        #
        for i in range(-10, 10):
            x = (i)/10
            input_vector = [x]
            self.model_inputs.append(input_vector)

            y = x**2
            output_vector = [y]
            self.model_outputs.append(output_vector)

        AI = NN.NN(self.structure)
        self.trained_weights, self.trained_biases = AI.train(self.model_inputs, self.model_outputs)

    def pause(self):
        with open("command_injection.dat", "w") as f:
            f.write("0")

    def graph(self):
        def draw():
            os.system("python3 plot_training.py")
        threading.Thread(target=draw).start()

    def kill(self):
        os.system("sudo killall python3")

    def resume(self):
        with open("command_injection.dat", "w") as f:
            f.write("1")

        def go():#have to make this to enable threading - keep the main window alive
            AI = NN.NN(self.structure)
            self.trained_weights, self.trained_biases = AI.train(self.model_inputs, self.model_outputs, default_weights=self.trained_weights, default_biases=self.trained_biases)

        threading.Thread(target=go).start()

    def evaluate(self):
        self.test_inputs = self.model_inputs
        AI = NN.NN(self.structure)
        try:
            self.trained_weights = AI.w
            self.trained_biases = AI.b
            outputs = AI.run(self.test_inputs, self.trained_weights, self.trained_biases)
            print(outputs)
        except Exception as e:
            print(e)


app = QApplication(sys.argv)

window = GUI()
window.show()
app.exec()
