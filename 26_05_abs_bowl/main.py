from KeyVal import KeyVal
from model import ax_plus_b, forward, actual
from plot import TrainingPlot
import numpy as np


if __name__ == "__main__":
    # y = |x+2| + |x| + |x-2|
    answers = [KeyVal(x, actual(x)) for x in np.arange(-5, 5.1, 0.5)]

    # a,b 起始值為0會完全不動 (門永遠關著 → 鎖死)
    a = 0.5
    b = 0.5
    # c,d 一開使用0.5反而會卡住哩 (跟 a,b 一模一樣 → 修正量也一模一樣 → 永遠長一樣)
    c = -0.5
    d = -0.5
    e = 0.3
    f = 0.3
    g = 0.4
    h = -0.4
    bias = 0.0   # output 層的常數, relu 相加最低只到 0, 靠它抬高
    step_size = 0.05
    epochs = 500
    draw_every = 100   # 每幾個 epoch 逐筆重畫一次

    plot = TrainingPlot(answers, a, b, c, d, e, f, g, h, bias)

    # ---------- 訓練 ----------
    for i in range(epochs):
        for ans in answers:
            predict = forward(a, b, c, d, e, f, g, h, bias, ans.key)
            gradient = predict - ans.val

            # 門：relu 的輸入 ≤ 0 時輸出被砍成 0，這顆對 predict 沒影響 → 不更新
            if ax_plus_b(a, b, ans.key) > 0:
                a -= gradient * step_size * ans.key
                b -= gradient * step_size
            if ax_plus_b(c, d, ans.key) > 0:
                c -= gradient * step_size * ans.key
                d -= gradient * step_size
            if ax_plus_b(e, f, ans.key) > 0:
                e -= gradient * step_size * ans.key
                f -= gradient * step_size
            if ax_plus_b(g, h, ans.key) > 0:
                g -= gradient * step_size * ans.key
                h -= gradient * step_size
            bias -= gradient * step_size   # 輸入永遠是 1, 沒有門

            if i % draw_every == 0:
                plot.update(a, b, c, d, e, f, g, h, bias, ans)

        print(f"epoch {i}: a={a:.4f}, b={b:.4f}, c={c:.4f}, d={d:.4f}, e={e:.4f}, f={f:.4f}, g={g:.4f}, h={h:.4f}, bias={bias:.4f}")

    plot.finish()
