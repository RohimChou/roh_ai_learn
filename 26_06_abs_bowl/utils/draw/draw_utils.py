import numpy as np
import matplotlib.pyplot as plt


def new_plot(title: str = None,
             xlabel: str = None,
             ylabel: str = None,
             xlim: tuple = None,
             ylim: tuple = None,
             grid: bool = True):
    """開一張新圖, 把每次都要寫的那幾行雜訊收起來。

    回傳 (fig, ax), 後面照常用 ax.plot(...) 畫線。
    只收「每張圖都一樣」的設定; 要畫什麼線留在各自的 main.py, 那才是練習的重點。

    例: fig, ax = new_plot(xlim=(-6, 6), ylim=(-5, 16))
    """
    fig, ax = plt.subplots()

    # 有給才設, 沒給就用 matplotlib 自動調整
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlim:
        ax.set_xlim(*xlim)   # xlim=(-6, 6) 展開成 set_xlim(-6, 6)
    if ylim:
        ax.set_ylim(*ylim)
    if grid:
        ax.grid()

    return fig, ax

def draw_function(f, xlim: tuple = (-10, 10), points: int = 200, label: str = None, show: bool = True):
    """丟一個 f(x) 進來, 直接把圖畫出來。

    f: 吃 numpy array 回 numpy array 的 function, 例: lambda x: x ** 2, 或 np.sin
    xlim: x 的範圍, 預設 -10 ~ 10
    points: 這個範圍內取幾個點連線, 越多越平滑
    label: 圖例名稱, 沒給就用 function 名字
    show: True 畫完直接跳視窗; False 只回傳 (fig, ax), 想再加東西再自己 plt.show()

    例: draw_function(lambda x: abs(x + 2) + abs(x) + abs(x - 2), xlim=(-5, 5))
    """
    xs = np.linspace(xlim[0], xlim[1], points)
    ys = f(xs)

    fig, ax = new_plot(xlim=xlim)
    ax.axhline(0, color="gray", linewidth=0.8)   # 在 y=0 畫水平參考線
    ax.axvline(0, color="gray", linewidth=0.8)   # 在 x=0 畫垂直參考線
    ax.plot(xs, ys, label=label or getattr(f, "__name__", "f(x)"))
    ax.legend()

    if show:
        plt.show()
    return fig, ax
