import utils.json_utils as json_utils
import numpy as np
from KeyVal import KeyVal
from Neuron import Neuron

def ax_plus_b(input_num, neu: Neuron):
    return input_num * neu.weights[0] + neu.bias

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
    learning_rate = 0.0863 # 發散點在 0.086251 左右, 超過就發散

    for epoch in range(50):
        print(f"\n\n{epoch + 1}")
        for key_val in training_data:
            ax_pb = ax_plus_b(key_val.key, neu1)
            error = key_val.val - ax_pb

            print(f'{key_val.key} {key_val.val:>2} ax_pb {ax_pb:>5.2f}  error {error:>5.2f} {neu1.weights[0]:.4f} += ',
                  f"{learning_rate * error * key_val.key:.4f}")

            # 更新神經元
            neu1.weights[0] += learning_rate * error * key_val.key
            neu1.bias += learning_rate * error

    print(json_utils.dumps(neu1))