from model import relu, ax_plus_b, forward, actual
from utils.colors import Color
from utils.draw.figure import Figure


class TrainingPlot:
    """訓練過程的即時圖。main.py 只管算, 畫圖的事全部交給這裡。

    用法:
        plot = TrainingPlot(answers, a, b, c, d, e, f, g, h, bias)   # 開視窗、畫第一版
        plot.update(a, b, c, d, e, f, g, h, bias, ans)               # 參數被改過之後重畫
        plot.finish()                                                # 訓練完, 停在最後一張圖
    """

    def __init__(self, answers, a, b, c, d, e, f, g, h, bias):
        self.fig = Figure(xlim=(-6, 6), ylim=(-5, 16), legend_loc=(1.02, 1), figsize=(8.5, 5))   # 圖例放圖外右上, 不擋線; 視窗加寬補回被圖例佔掉的空間

        # 不會動的: 訓練資料點、目標曲線
        self.fig.add_points([ans.key for ans in answers], [ans.val for ans in answers], label="training data")
        self.fig.add_line(actual, linewidth=2, label="target: |x+2| + |x| + |x-2|")

        # 會動的: 每顆 neuron 兩條線, 折前 (虛線, 淺色) / 折後 (實線, 深色), 同一顆用同色系
        self.raw_line1 = self.fig.add_line(lambda xs: ax_plus_b(a, b, xs), color=Color.PLUM, linestyle="--", label="n1 raw")
        self.line1 = self.fig.add_line(lambda xs: relu(ax_plus_b(a, b, xs)), color=Color.PURPLE, label="n1 relu")
        self.raw_line2 = self.fig.add_line(lambda xs: ax_plus_b(c, d, xs), color=Color.BURLY_WOOD, linestyle="--", label="n2 raw")
        self.line2 = self.fig.add_line(lambda xs: relu(ax_plus_b(c, d, xs)), color=Color.SADDLE_BROWN, label="n2 relu")
        self.raw_line3 = self.fig.add_line(lambda xs: ax_plus_b(e, f, xs), color=Color.LIGHT_GREEN, linestyle="--", label="n3 raw")
        self.line3 = self.fig.add_line(lambda xs: relu(ax_plus_b(e, f, xs)), color=Color.SEA_GREEN, label="n3 relu")
        self.raw_line4 = self.fig.add_line(lambda xs: ax_plus_b(g, h, xs), color=Color.LIGHT_STEEL_BLUE, linestyle="--", label="n4 raw")
        self.line4 = self.fig.add_line(lambda xs: relu(ax_plus_b(g, h, xs)), color=Color.STEEL_BLUE, label="n4 relu")
        # 四顆加總 = 模型真正的輸出
        self.line_sum = self.fig.add_line(lambda xs: forward(a, b, c, d, e, f, g, h, bias, xs), color=Color.LIGHT_PINK, linewidth=2, label="predict (sum)")
        # 目前正在學的那一筆, 先放空的, update() 時再移到位
        self.active_point = self.fig.add_points([], [], color=Color.ORANGE, s=100, zorder=3, label="current sample")

        self.fig.enable_legend_toggle()   # 點圖例可以隱藏 / 顯示該條線
        self.fig.show(block=False)
        self.fig.pause(0.1)

    def update(self, a, b, c, d, e, f, g, h, bias, ans):
        """參數改過之後, 用新參數重算每條線並重畫"""
        self.fig.update_line(self.raw_line1, lambda xs: ax_plus_b(a, b, xs))
        self.fig.update_line(self.line1, lambda xs: relu(ax_plus_b(a, b, xs)))
        self.fig.update_line(self.raw_line2, lambda xs: ax_plus_b(c, d, xs))
        self.fig.update_line(self.line2, lambda xs: relu(ax_plus_b(c, d, xs)))
        self.fig.update_line(self.raw_line3, lambda xs: ax_plus_b(e, f, xs))
        self.fig.update_line(self.line3, lambda xs: relu(ax_plus_b(e, f, xs)))
        self.fig.update_line(self.raw_line4, lambda xs: ax_plus_b(g, h, xs))
        self.fig.update_line(self.line4, lambda xs: relu(ax_plus_b(g, h, xs)))
        self.fig.update_line(self.line_sum, lambda xs: forward(a, b, c, d, e, f, g, h, bias, xs))
        self.fig.update_points(self.active_point, [ans.key], [ans.val])

        # pause 會順便把最新資料畫到視窗上
        self.fig.pause(0.15)

    def finish(self):
        """訓練結束, 視窗留著直到手動關掉"""
        self.fig.show()
