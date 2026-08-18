# 實驗結論: 線性 neuron fit 拋物線 → underfit
# MSE 降到 ~23 就卡住 (loss 平原), 學到 d ≈ 19.0·t − 11.8
#
# d
# 60┤                        ⡠⠊ ← 正解拋物線 d=4.9t²
#   │                    ⡠⠊╱
# 40┤                 ⡠⠊ ╱  ← 學到的直線 19t−11.8
#   │              ⡔  ╱        中段穿過、兩端翹掉
# 20┤          ⡠⠊ ╱
#   │      ⣀⠔  ╱
#  0┼──⣀⡠⠤⠊╱─────────────→ t
#   │    ╱ ← t<0.62 時預測是負的 (掉「負距離」, 物理上荒謬)
#
# 誤差不是隨機、是系統性彎曲 → underfit 招牌特徵
# t=3.0 剛好在直線與曲線交點附近所以誤差小, t=1.5 在中段誤差大
import utils.json_utils as json_utils
import numpy as np
from KeyVal import KeyVal
from Neuron import Neuron

def ax_plus_b(input_num, neu: Neuron):
    return input_num * neu.weights[0] + neu.bias

if __name__ == "__main__":
    # 自由落體: d = 4.9 * t^2 (g=9.8, d=½gt²)
    # key = 時間(秒), val = 落下距離(公尺)
    kv1 = KeyVal(0.5, 4.9 * 0.5 ** 2)   #  1.225
    kv2 = KeyVal(1.0, 4.9 * 1.0 ** 2)   #  4.9
    kv3 = KeyVal(1.5, 4.9 * 1.5 ** 2)   # 11.025
    kv4 = KeyVal(2.0, 4.9 * 2.0 ** 2)   # 19.6
    kv5 = KeyVal(2.5, 4.9 * 2.5 ** 2)   # 30.625
    kv6 = KeyVal(3.0, 4.9 * 3.0 ** 2)   # 44.1
    kv7 = KeyVal(3.5, 4.9 * 3.5 ** 2)   # 60.025

    # 訓練用資料 (kv3, kv6 留著當 test)
    training_data = np.array([kv1, kv2, kv4, kv5, kv7])
    test_data = np.array([kv3, kv6])

    # 開始
    neu1 = Neuron(input_cnt=1)
    learning_rate = 0.02

    for epoch in range(100):
        for key_val in training_data:
            ax_pb = ax_plus_b(key_val.key, neu1)
            error = key_val.val - ax_pb

            # 更新神經元
            neu1.weights[0] += learning_rate * error * key_val.key
            neu1.bias += learning_rate * error

        # 每個 epoch 算一次 MSE, 觀察 loss 卡在哪
        mse = np.mean([(kv.val - ax_plus_b(kv.key, neu1)) ** 2 for kv in training_data])
        if (epoch + 1) % 10 == 0:
            print(f"epoch {epoch + 1:>3}  MSE {mse:>8.4f}  w {neu1.weights[0]:.4f}  b {neu1.bias:.4f}")

    print(json_utils.dumps(neu1))

    # 用 test data 驗證: 線性 neuron 只能畫直線, 拋物線 fit 不起來
    print("\n--- test ---")
    for kv in test_data:
        pred = ax_plus_b(kv.key, neu1)
        print(f"t={kv.key}  正解 {kv.val:>7.3f}  預測 {pred:>7.3f}  error {kv.val - pred:>7.3f}")
