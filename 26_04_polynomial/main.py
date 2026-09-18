from sympy.printing.pretty.pretty_symbology import line_width

from KeyVal import KeyVal
import matplotlib.pyplot as plt
import numpy as np

def ax2_plus_bx_plus_c(a: float, b: float, c: float, x: float):
    return a * (x ** 2) + b * x + c

if __name__ == "__main__":
    # y = 3x² + 2x + 1
    kv1 = KeyVal(-3.0, 22.00)  # 3*9.00 + 2*(-3.0) + 1
    kv2 = KeyVal(-2.0, 9.00)  # 3*4.00 + 2*(-2.0) + 1
    kv3 = KeyVal(-1.5, 4.75)  # 3*2.25 + 2*(-1.5) + 1
    kv4 = KeyVal(-0.5, 0.75)  # 3*0.25 + 2*(-0.5) + 1
    kv5 = KeyVal(0.0, 1.00)  # 3*0.00 + 2*( 0.0) + 1
    kv6 = KeyVal(1.0, 6.00)  # 3*1.00 + 2*( 1.0) + 1
    kv7 = KeyVal(1.5, 10.75)  # 3*2.25 + 2*( 1.5) + 1
    kv8 = KeyVal(2.5, 24.75)  # 3*6.25 + 2*( 2.5) + 1
    kv9 = KeyVal(3.0, 34.00)  # 3*9.00 + 2*( 3.0) + 1
    answers = [kv1, kv2, kv3, kv4, kv5, kv6, kv7, kv8, kv9]

    a = 0
    b = 0
    c = 0
    learning_rate = 0.05515
    points = np.linspace(-10, 10, 21)
    fig, ax = plt.subplots()

    for i in range(120):
        for ans in answers:
            predict_val = ax2_plus_bx_plus_c(a, b, c, ans.key)
            error = predict_val - ans.val

            a -= error * learning_rate * ans.key * ans.key
            b -= error * learning_rate * ans.key
            c -= error * learning_rate

        if i % 24 == 0:
            ax.plot(points,
                    ax2_plus_bx_plus_c(a, b, c, points),
                    color='red',
                    alpha= 0.8 - i / 150,
                    linewidth=2)

    print(f"a: {a:.4}, {b:.4}, {c:.4}")

    ax.plot(points, 3 * points ** 2 + 2 * points + 1, linewidth=3)
    fig.show()