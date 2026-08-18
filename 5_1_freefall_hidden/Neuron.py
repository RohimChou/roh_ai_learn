import random

class Neuron:
    def __init__(self, input_cnt):
        # 隨機初始化, 不能全 0:
        # 1. 全 0 會讓每個 hidden neuron 學到一模一樣 (對稱性), 等於只有一顆
        # 2. ReLU 在 0 沒有梯度, 全 0 起步會整層卡死
        self.weights = []
        self.bias = random.uniform(-1, 1)

        for i in range(input_cnt):
            self.weights.append(random.uniform(-1, 1))
