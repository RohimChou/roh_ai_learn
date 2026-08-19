import numpy as np
from KeyVal import KeyVal
from Neuron import Neuron

def ax_plus_b(a: float, x: float, b: float):
    """
    ax + b = input * weight + bias
    """
    return a * x + b

# 用神經元算 2 的倍數
if __name__ == "__main__":
    # 所有資料
    kv1 = KeyVal(1, 2)
    kv2 = KeyVal(2, 4)
    kv3 = KeyVal(3, 6)
    kv4 = KeyVal(4, 8)
    kv5 = KeyVal(5, 10)
    kv6 = KeyVal(6, 12)
    kv7 = KeyVal(7, 14)

    # 訓練用資料
    training_data = np.array([kv1, kv2, kv4, kv5, kv7])

    # 開始
    neu1 = Neuron(input_cnt=1)
    learning_rate = 0.0562 # 發散點在 0.086251 左右, 超過就發散

    for epoch in range(150):
        print(f"\n\n{epoch + 1}")
        for answer in training_data:
            ax_pb = ax_plus_b(neu1.weights[0], answer.key, neu1.bias)
            error = answer.val - ax_pb

            print(f'{answer.key} {answer.val:>2} ax_pb {ax_pb:>5.2f}  error {error:>5.2f} {neu1.weights[0]:.4f} += ',
                  f"{learning_rate * error * answer.key:.4f}")

            # 更新神經元
            neu1.weights[0] += learning_rate * error * answer.key
            neu1.bias += learning_rate * error

    print(ax_plus_b(neu1.weights[0], 50, neu1.bias))