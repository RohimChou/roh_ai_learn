import matplotlib.pyplot as plt
import numpy as np

from KeyVal import KeyVal
from Neuron import Neuron
from utils import json_utils

def ax_plus_b(a: float, x: float, b: float):
    """
    ax² + b : feature (x²) 直接寫在函式裡, 呼叫端照餵 x
    """
    return a * x ** 2 + b

if __name__ == "__main__":
    # y = x²
    kv1 = KeyVal(-3.0, (-3.0) ** 2)   # 9.0
    kv2 = KeyVal(-2.0, (-2.0) ** 2)   # 4.0
    kv3 = KeyVal(-1.5, (-1.5) ** 2)   # 2.25
    kv4 = KeyVal(-0.5, (-0.5) ** 2)   # 0.25
    kv5 = KeyVal( 0.0,   0.0  ** 2)   # 0.0
    kv6 = KeyVal( 1.0,   1.0  ** 2)   # 1.0
    kv7 = KeyVal( 1.5,   1.5  ** 2)   # 2.25
    kv8 = KeyVal( 2.5,   2.5  ** 2)   # 6.25
    kv9 = KeyVal( 3.0,   3.0  ** 2)   # 9.0
    answers = [kv1, kv2, kv3, kv4, kv5, kv6, kv7, kv8, kv9]

    neu1 = Neuron(1)
    learn_rate = 0.0125

    for i in range(200):
        for ans in answers:
            guess = ax_plus_b(neu1.weights[0], ans.key, neu1.bias)
            error = guess - ans.val

            # tune params
            neu1.weights[0] = neu1.weights[0] - error * learn_rate * ans.key ** 2
            neu1.bias = neu1.bias - error * learn_rate * 1

    print(ax_plus_b(neu1.weights[0], 4, neu1.bias))

    # 畫圖
    x_axis_points = np.linspace(-5, 5, 30)
    fig, ax = plt.subplots()
    ax.plot(x_axis_points, x_axis_points ** 2)
    ax.plot(x_axis_points, ax_plus_b(
        neu1.weights[0], x_axis_points, neu1.bias), alpha=0.2, linewidth=5)
    ax.grid()
    fig.show()

    # 訓練完後逐點看 error: 正 = 猜太高(線在拋物線上面), 負 = 猜太低
    for ans in answers:
        guess = ax_plus_b(neu1.weights[0], ans.key, neu1.bias)
        error = guess - ans.val
        print(f"x={ans.key:>5.1f}  正解 {ans.val:>5.2f}  預測 {guess:>5.2f}  error {error:>+6.2f}")

    allYs = [ans.val for ans in answers]
    print(f"\navg = {(sum(allYs)/len(allYs))}")
