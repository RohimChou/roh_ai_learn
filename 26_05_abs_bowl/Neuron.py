class Neuron:
    def __init__(self, input_cnt):
        self.weights = []
        self.bias = 0

        for i in range(input_cnt):
            self.weights.append(0)
        self.bias = 0
