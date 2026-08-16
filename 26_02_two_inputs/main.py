import numpy as np
import matplotlib.pyplot as plt
from utils import json_utils
from InputOutput import InputOutput
from Neuron import Neuron

def ax_plus_by_plus_c(a: float, x: float, b: float, y: float, c: float):
    return a * x + b * y + c

# try to fit y = 2a + 3b + 4
if __name__ == "__main__":
    # 所有資料
    # y = 2a + 3b + 4
    io1 = InputOutput(1, 1,  9)   # 2*1 + 3*1 + 4
    io2 = InputOutput(2, 1, 11)   # 4 + 3 + 4
    io3 = InputOutput(1, 2, 12)   # 2 + 6 + 4
    io4 = InputOutput(3, 2, 16)   # 6 + 6 + 4
    io5 = InputOutput(2, 3, 17)   # 4 + 9 + 4
    io6 = InputOutput(4, 1, 15)   # 8 + 3 + 4
    io7 = InputOutput(1, 4, 18)   # 2 + 12 + 4
    io8 = InputOutput(3, 3, 19)   # 6 + 9 + 4
    answers = np.array([io1, io2, io3, io4, io5, io6, io7, io8]);

    neu1 = Neuron(2)
    step_size = 0.1

    # 畫圖用
    fig, ax = plt.subplots()
    prediction_errors = []
    a_values = []
    b_values = []
    c_values = []

    for i in range(200):
        for ans in answers:
            predict = ax_plus_by_plus_c(
                neu1.weights[0],
                ans.input1,
                neu1.weights[1],
                ans.input2,
                neu1.bias
            )

            error = ans.output1 - predict

            # 修正
            neu1.weights[0] = neu1.weights[0] + error * ans.input1 * step_size
            neu1.weights[1] = neu1.weights[1] + error * ans.input2 * step_size
            neu1.bias = neu1.bias + error * 1 * step_size

            # 每次修正後，記錄 a、b、c 的實際值。
            prediction_errors.append(error)
            a_values.append(neu1.weights[0])
            b_values.append(neu1.weights[1])
            c_values.append(neu1.bias)

    display_count = 300
    updates = range(1, display_count + 1)
    ax.plot(updates, prediction_errors[:display_count], label="prediction error", alpha=0.7)
    ax.plot(updates, a_values[:display_count], label="a value")
    ax.plot(updates, b_values[:display_count], label="b value")
    ax.plot(updates, c_values[:display_count], label="c value")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Update")
    ax.set_ylabel("Value / Error")
    ax.legend()
    plt.grid()
    plt.show()

    predict = ax_plus_by_plus_c(
        neu1.weights[0],
        5,
        neu1.weights[1],
        6,
        neu1.bias
    )

    print(json_utils.dumps(neu1))
    print(predict) # 32
