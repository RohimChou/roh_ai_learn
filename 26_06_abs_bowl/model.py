import numpy as np


def relu(x):
    return np.maximum(0, x)

def ax_plus_b(a: float, b: float, x: float):
    return a * x + b

def forward(a: float, b: float, c: float, d: float, e: float, f: float, g: float, h: float, bias: float, x: float):
    return relu(ax_plus_b(a, b, x)) + relu(ax_plus_b(c, d, x)) + relu(ax_plus_b(e, f, x)) + relu(ax_plus_b(g, h, x)) + bias   # bias: 把整條線往上搬, 碗底不在 0 才貼得上

def actual(x):
    return abs(x+2) + abs(x) + abs(x-2)
