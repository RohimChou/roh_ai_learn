from enum import StrEnum


class Color(StrEnum):
    """畫圖用的顏色名稱, 避免到處寫 magic string 打錯字。

    用 StrEnum 的好處: 本身就是字串, 可以直接丟給 matplotlib, 不用再寫 .value
    例: ax.plot(x, y, color=Color.SEA_GREEN)

    命名慣例: 每個色系都備「淺 / 中 / 深」三階,
    淺色畫折前的虛線 (raw), 深色畫折後的實線 (relu 後), 中階留給第三條線。
    """

    # ---------- 中性色: 座標軸、參考線、背景 ----------
    WHITE = "white"
    LIGHT_GRAY = "LightGray"
    GRAY = "gray"
    DIM_GRAY = "DimGray"
    BLACK = "black"

    # ---------- 紅系 ----------
    LIGHT_PINK = "LightPink"
    INDIAN_RED = "IndianRed"
    RED = "red"

    # ---------- 綠系 ----------
    LIGHT_GREEN = "LightGreen"
    DARK_SEA_GREEN = "DarkSeaGreen"
    SEA_GREEN = "SeaGreen"

    # ---------- 藍系 ----------
    LIGHT_STEEL_BLUE = "LightSteelBlue"
    CORNFLOWER_BLUE = "CornflowerBlue"
    STEEL_BLUE = "SteelBlue"

    # ---------- 紫系 ----------
    PLUM = "Plum"
    MEDIUM_PURPLE = "MediumPurple"
    PURPLE = "purple"

    # ---------- 橘黃系: 也常拿來當強調色 (標目前這筆 sample) ----------
    KHAKI = "Khaki"
    ORANGE = "orange"
    DARK_ORANGE = "DarkOrange"

    # ---------- 青綠系 (teal) ----------
    PALE_TURQUOISE = "PaleTurquoise"
    TURQUOISE = "turquoise"
    TEAL = "teal"

    # ---------- 棕系 ----------
    BURLY_WOOD = "BurlyWood"
    PERU = "Peru"
    SADDLE_BROWN = "SaddleBrown"

    # ---------- 桃紅系 ----------
    PINK = "pink"
    HOT_PINK = "HotPink"
    MEDIUM_VIOLET_RED = "MediumVioletRed"
