# 5_0_freefall 的續集: 同樣的資料 d=4.9t², 但加一層 hidden layer (ReLU)
# 目標: 看 MSE 能不能突破線性版卡住的 ~23
#
# 網路結構 (1 → 3 → 1):
#
#         hidden (ReLU)
#          ┌─ h1 ─┐
#   t ──── ┼─ h2 ─┼──── out ──→ 預測距離
#          └─ h3 ─┘
#
# 每顆 hidden neuron 學一段斜率, ReLU 負責「還沒輪到我就輸出 0」
# → 三段直線接起來逼近拋物線 (piecewise linear)
import random
import utils.json_utils as json_utils
import numpy as np
from KeyVal import KeyVal
from Neuron import Neuron

def relu(z):
    return max(0.0, z)

def forward(t, hidden_neurons, out_neuron):
    # 前向傳播: t → hidden(ReLU) → out(線性)
    hidden_outs = []
    for neu in hidden_neurons:
        z = t * neu.weights[0] + neu.bias
        hidden_outs.append(relu(z))

    pred = out_neuron.bias
    for i, h_out in enumerate(hidden_outs):
        pred += h_out * out_neuron.weights[i]
    return pred, hidden_outs

if __name__ == "__main__":
    # 固定亂數, 每次跑結果一樣, 方便跟 5_0_freefall 比較
    # 注意: seed 影響超大! 掃過 0~19: 有的 MSE≈0, 有的整層 ReLU 死掉 MSE=448
    # (權重初始化全負 → 對 t>0 的輸入 ReLU 永遠輸出 0 → 梯度 0 → 永遠不更新 = dead ReLU)
    # 可自己改成 42(兩顆死), 3(全死) 感受一下
    random.seed(8)

    # 自由落體: d = 4.9 * t^2, 跟 5_0_freefall 完全同一份資料
    kv1 = KeyVal(0.5, 4.9 * 0.5 ** 2)   #  1.225
    kv2 = KeyVal(1.0, 4.9 * 1.0 ** 2)   #  4.9
    kv3 = KeyVal(1.5, 4.9 * 1.5 ** 2)   # 11.025
    kv4 = KeyVal(2.0, 4.9 * 2.0 ** 2)   # 19.6
    kv5 = KeyVal(2.5, 4.9 * 2.5 ** 2)   # 30.625
    kv6 = KeyVal(3.0, 4.9 * 3.0 ** 2)   # 44.1
    kv7 = KeyVal(3.5, 4.9 * 3.5 ** 2)   # 60.025

    training_data = np.array([kv1, kv2, kv4, kv5, kv7])
    test_data = np.array([kv3, kv6])

    # 開始: 3 顆 hidden neuron + 1 顆 output neuron
    hidden_neurons = [Neuron(input_cnt=1) for _ in range(3)]
    out_neuron = Neuron(input_cnt=3)
    learning_rate = 0.001

    for epoch in range(5000):
        for key_val in training_data:
            pred, hidden_outs = forward(key_val.key, hidden_neurons, out_neuron)
            error = key_val.val - pred

            # --- 反向傳播 ---
            # output 層: 跟以前一樣, error * 該權重的輸入 (= hidden 的輸出)
            for i, neu in enumerate(hidden_neurons):
                out_neuron.weights[i] += learning_rate * error * hidden_outs[i]
            out_neuron.bias += learning_rate * error

            # hidden 層: error 要乘上「它對輸出的影響力」(out_neuron 的權重) 往回傳
            # ReLU 的梯度: 輸出 > 0 時是 1, 否則 0 (被關掉的 neuron 這次不更新)
            for i, neu in enumerate(hidden_neurons):
                if hidden_outs[i] > 0:
                    neu.weights[0] += learning_rate * error * out_neuron.weights[i] * key_val.key
                    neu.bias += learning_rate * error * out_neuron.weights[i]

        mse = np.mean([(kv.val - forward(kv.key, hidden_neurons, out_neuron)[0]) ** 2
                       for kv in training_data])
        if (epoch + 1) % 500 == 0:
            print(f"epoch {epoch + 1:>5}  MSE {mse:>8.4f}")

    print("\nhidden:", json_utils.dumps(hidden_neurons))
    print("out:", json_utils.dumps(out_neuron))

    # 用 test data 驗證 (跟 5_0_freefall 同兩個點, 直接比 error)
    print("\n--- test ---")
    for kv in test_data:
        pred, _ = forward(kv.key, hidden_neurons, out_neuron)
        print(f"t={kv.key}  正解 {kv.val:>7.3f}  預測 {pred:>7.3f}  error {kv.val - pred:>7.3f}")
