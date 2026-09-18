import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D


class Figure:
    """管理一張圖的線條、散點與圖例；更新資料後由呼叫端決定何時重畫。

    title: 顯示在整張圖上方的標題。
    xlabel: 顯示在 X 軸下方的文字標籤。
    ylabel: 顯示在 Y 軸旁邊的文字標籤。
    xlim: X 軸顯示範圍，例如 (-5, 5)。
    ylim: Y 軸顯示範圍，例如 (0, 10)。
    grid: 是否顯示方便對照座標的格線。
    legend_loc: 圖例位置，例如 "upper left"、"lower right"、"center"；
                "best" 讓 Matplotlib 自己挑最不擋線的角落。
                想放到圖外面可給座標，例如 (1.02, 1) = 圖的右上角外側。
    figsize: 視窗大小 (寬, 高)，單位吋，預設 (6.4, 4.8)。圖例放外面時建議加寬，例如 (10, 5)。
    """

    # 保留柔和色調，但加大六種顏色的色相與明暗差，方便在白底上辨識。
    # 冷暖色交錯排列，讓相鄰線條不容易混在一起；第七條線回到第一色。
    LINE_COLORS = (
        "#3F7FC4",  # 藍
        "#E28A36",  # 杏桃橘
        "#4FA66A",  # 綠
        "#8A65B8",  # 紫
        "#279E9A",  # 青
        "#D95F8D",  # 莓果紅
    )

    # 預設外觀集中管理；呼叫端的 style 可覆蓋個別選項。
    LINE_STYLE = {
        "marker": ".",
        "markersize": 3,
        "linestyle": "-",
        "linewidth": 1,
        "alpha": 0.85,
    }

    def __init__(self,
                 title: str = None,
                 xlabel: str = None,
                 ylabel: str = None,
                 xlim: tuple = None,
                 ylim: tuple = None,
                 grid: bool = True,
                 legend_loc="best",
                 figsize: tuple = None):
        # fig 是整張圖；ax 是圖中的座標區域，線條與座標軸都畫在 ax 上。
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_prop_cycle(color=self.LINE_COLORS)

        # xlim 同時決定 X 軸顯示範圍與 function 的取樣範圍。
        self._xlim = (-10, 10) if xlim is None else xlim

        # 沒有提供的設定交給 Matplotlib 使用預設值或自動計算。
        if title is not None:
            self.ax.set_title(title)
        if xlabel is not None:
            self.ax.set_xlabel(xlabel)
        if ylabel is not None:
            self.ax.set_ylabel(ylabel)
        if xlim is not None:
            self.ax.set_xlim(*xlim)
        if ylim is not None:
            self.ax.set_ylim(*ylim)
        if grid:
            self.ax.grid(visible=True, alpha=0.4)
        else:
            self.ax.grid(visible=False)

        self._legend_loc = legend_loc

        # 「點圖例切換顯示」功能的狀態；enable_legend_toggle() 開啟後才會用到。
        self._legend_toggle_enabled = False
        self._legend_map = {}   # 圖例上的小圖示 → 圖中真正的線／點

    def add_line(self, function, points: int = 41, **style) -> Line2D:
        """將 function 的計算結果畫成線條，回傳供 update_line() 使用的物件。

        function: 接收一組 X 座標並回傳對應 Y 座標的函式，例如 np.sin。
        points: 在 X 軸範圍內取幾個點計算；數量越多，曲線通常越平滑。
        style: Matplotlib 的外觀選項，例如 color="red"、linestyle="--"。
               預設用小點標記資料並以實線連接；傳入的設定會取代預設值。
        """
        if not callable(function):
            raise TypeError("function must be callable")
        xs = self._sample_xs(points)
        ys = function(xs)

        style = {**self.LINE_STYLE, **style}

        # ax.plot() 回傳線條物件的 list；一次只畫一條，所以取第 0 個。
        line = self.ax.plot(xs, ys, **style)[0]

        # 有設定 label 就自動建立或刷新圖例，不必由呼叫端操作 ax.legend()。
        if line.get_label() and not line.get_label().startswith("_"):
            self._refresh_legend()
        return line

    def update_line(self,
                    line: Line2D,
                    function,
                    points: int = None,
                    label: str = None):
        """用 function 重新計算既有線條，但不立即重畫畫面。

        line: 這張圖的 add_line() 回傳的線條物件。
        function: 接收一組 X 座標並回傳對應 Y 座標的函式。
        points: 新的取樣點數；省略時沿用這條線原本的 X 座標。
        label: 新的圖例文字；省略時保留原本的文字。
        """
        if not isinstance(line, Line2D):
            raise TypeError("line must be a Line2D object")
        # 避免誤更新另一張圖的線條。
        if line.axes is not self.ax:
            raise ValueError("line does not belong to this figure")
        if not callable(function):
            raise TypeError("function must be callable")
        xs = line.get_xdata() if points is None else self._sample_xs(points)
        # 先完成計算，避免 function 失敗時只改到 X 座標。
        ys = function(xs)
        line.set_data(xs, ys)

        if label is not None:
            line.set_label(label)
            self._refresh_legend()

    def add_points(self, xs, ys, **style) -> PathCollection:
        """把一組座標畫成散點（不連線），回傳供 update_points() 使用的物件。

        xs, ys: 要畫的 X、Y 座標，長度要一樣，例：xs=[-1, 0, 1], ys=[3, 1, 3]。
        style: Matplotlib 的外觀選項，例如 color="red"、s=100（點的大小）。
        適合畫「固定的資料點」，而不是像 add_line() 那樣沿 X 軸整段取樣。
        """
        points = self.ax.scatter(xs, ys, **style)

        # 有設定 label 就自動刷新圖例，和 add_line() 行為一致。
        if points.get_label() and not points.get_label().startswith("_"):
            self._refresh_legend()
        return points

    def update_points(self, points: PathCollection, xs, ys):
        """把既有散點移到新座標，但不立即重畫畫面。

        points: 這張圖的 add_points() 回傳的物件。
        xs, ys: 新的座標，例：只想標一個點就傳 xs=[2], ys=[5]。
        """
        if not isinstance(points, PathCollection):
            raise TypeError("points must be a PathCollection object")
        # 避免誤更新另一張圖的點。
        if points.axes is not self.ax:
            raise ValueError("points do not belong to this figure")
        # scatter 內部用 (N, 2) 的座標表存位置，把 xs、ys 兩欄併起來即可。
        points.set_offsets(np.column_stack([xs, ys]))

    def enable_legend_toggle(self):
        """開啟「點圖例切換顯示」：點圖例上的某一項，對應的線／點就隱藏或顯示。

        只有 add_line() / add_points() 時有給 label 的才會出現在圖例，才點得到。
        關掉的項目在圖例上會變淡，再點一次就恢復。
        """
        if self._legend_toggle_enabled:
            return
        self._legend_toggle_enabled = True
        # pick_event：滑鼠點到「有設 picker 的物件」時觸發。
        self.fig.canvas.mpl_connect("pick_event", self._on_legend_pick)
        self._refresh_legend()

    def draw(self):
        """將最新資料真正畫到視窗，並處理滑鼠、鍵盤等視窗事件。"""
        # canvas 是實際呈現圖像的畫布；更新多條線後只重畫一次比較有效率。
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def show(self, block: bool = True):
        """顯示圖形視窗。

        block=True: 程式停在這裡，直到使用者關閉視窗。
        block=False: 視窗開啟後，程式繼續執行，適合即時更新訓練圖。
        """
        plt.show(block=block)

    def pause(self, seconds: float):
        """暫停指定秒數，同時讓圖形視窗繼續回應操作。"""
        plt.pause(seconds)

    def _sample_xs(self, points: int):
        """沿固定取樣範圍產生 X 座標，讓新增與更新使用相同規則。"""
        if points < 1:
            raise ValueError("points must be greater than 0")
        return np.linspace(self._xlim[0], self._xlim[1], points)

    def _refresh_legend(self):
        """重建圖例；若已開啟 toggle，順便讓每一項可以被點。

        Matplotlib 的 legend() 每次呼叫都是整個重做，之前設的 picker 會消失，
        所以新增線／點或改 label 後都要從這裡重建。
        """
        # get_legend_handles_labels() 只會列出有 label 的線／點，順序和圖例一致，可一對一配對。
        self._legend_map.clear()
        artists, labels = self.ax.get_legend_handles_labels()
        if not artists:
            # 最後一個 label 被移除時，也要清掉舊圖例。
            legend = self.ax.get_legend()
            if legend is not None:
                legend.remove()
            return
        if isinstance(self._legend_loc, tuple):
            # 給座標時，(0, 0) 是圖的左下角、(1, 1) 是右上角，超過 1 就在圖外面。
            legend = self.ax.legend(artists, labels, loc="upper left", bbox_to_anchor=self._legend_loc)
            self.fig.tight_layout()   # 重新排版，讓圖外的圖例不會被視窗邊緣切掉
        else:
            legend = self.ax.legend(artists, labels, loc=self._legend_loc)
        if not self._legend_toggle_enabled:
            return

        for legend_handle, artist in zip(legend.legend_handles, artists):
            legend_handle.set_picker(5)   # 設定圖例圖示的點選容許範圍
            legend_handle.set_alpha(1.0 if artist.get_visible() else 0.2)
            self._legend_map[legend_handle] = artist

    def _on_legend_pick(self, event):
        """點到圖例某一項 → 切換該線／點的顯示，並把圖例圖示變淡或恢復。"""
        artist = self._legend_map.get(event.artist)
        if artist is None:
            return
        visible = not artist.get_visible()
        artist.set_visible(visible)
        event.artist.set_alpha(1.0 if visible else 0.2)
        self.fig.canvas.draw_idle()
