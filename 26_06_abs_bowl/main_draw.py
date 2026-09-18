import numpy as np
from utils.draw.figure import Figure


def bowl(x):
    return abs(x + 2) + abs(x) + abs(x - 2)


def relu(x):
    return np.maximum(0, x)


if __name__ == "__main__":
    fig = Figure()

    # 不指定 color，依加入順序自動使用七色循環。
    # fig.add_line(lambda xs: xs, label="y = x")
    # fig.add_line(lambda xs: -xs, label="y = -x")
    fig.add_line(lambda xs: relu(-xs), label="ReLU(-x)")
    # fig.add_line(lambda xs: np.sin(xs) * 5, label="5 sin(x)")
    fig.add_line(lambda xs: np.cos(xs) * 5, label="5 cos(x)")
    line1 = fig.add_line(lambda xs: 0.1 * xs ** 2 - 5, label="0.1x² - 5")

    fig.pause(0.5)

    fig.update_line(line1, lambda xs: 0.2 * xs ** 2 - 5, label="0.2x² - 5")

    fig.pause(0.5)

    fig.update_line(line1, lambda xs: 0.3 * xs ** 2 - 5, label="0.3x² - 5")

    fig.enable_legend_toggle()

    fig.show()
