# 神經網路練習階梯 (從 0_clean 重來)

一次爬一階。每一階都是**上一階的網路壞掉了**, 而修好它的方法就是那一階的重點。

核心結構:

- **第 1~4 階**: 完全不用改 neuron 的 code, 只換資料。網路一直是「一顆 neuron, ax+b」
- **第 5 階**: 一顆線性 neuron 徹底做不到 → 解法是換 **feature**, 不是加 layer
- **第 7 階**: feature 也想不出來了 → 這時 hidden layer 才真的有存在意義

| # | 目標公式 | 新觀念 | code 要改什麼 |
| :- | :- | :- | :- |
| 1 | `y = 2x` | ✅ 已完成 (`0_clean`) | — |
| 2 | `y = 2x + 3` | bias 真的有在工作 | 只換資料。可以試著把 bias 鎖在 0, 看它怎麼學不起來 |
| 3 | `y = 2a + 3b + 4` | 多輸入 | `Neuron(input_cnt=2)`, weight 更新改成跑迴圈 |
| 4 | `y = 2a + 3b - 4c + 5` | 3 個變數也一樣 | 只換資料 — 從 2 變 N 完全不用改 code, 這就是重點 |
| 5 | `y = x²` | **一顆線性 neuron 做不到** | 把 `x²` 當輸入餵進去。還是一顆 neuron, 學出 weight ≈ 1 |
| 6 | `y = 3x² + 2x + 1` | 多項式迴歸 = 線性 neuron 換個 feature | 輸入 `[x², x]`, 一顆 neuron 學出全部 3 個係數 |
| 7 | `y = sin(x)` | feature 想不完了 | 這時才加 hidden layer |

---

## 第 3 階練習資料 (`y = 2a + 3b + 4`)

用現成的 `InputOutput` class (剛好就是 `input1` / `input2` / `output1` 這個形狀):

```python
    # y = 2a + 3b + 4
    io1 = InputOutput(1, 1,  9)   # 2*1 + 3*1 + 4
    io2 = InputOutput(2, 1, 11)   # 4 + 3 + 4
    io3 = InputOutput(1, 2, 12)   # 2 + 6 + 4
    io4 = InputOutput(3, 2, 16)   # 6 + 6 + 4
    io5 = InputOutput(2, 3, 17)   # 4 + 9 + 4
    io6 = InputOutput(4, 1, 15)   # 8 + 3 + 4
    io7 = InputOutput(1, 4, 18)   # 2 + 12 + 4
    io8 = InputOutput(3, 3, 19)   # 6 + 9 + 4

    # 訓練用資料
    training_data = np.array([io1, io2, io3, io4, io6, io8])
    # 留兩筆沒訓練過的驗證
    test_data = np.array([io5, io7])
```

更新的地方: 每個 weight 要乘上**自己對應的那個輸入**, 所以 a 跟 b 每次移動的量不一樣。

```python
    pred = neu1.weights[0] * io.input1 + neu1.weights[1] * io.input2 + neu1.bias
    error = io.output1 - pred

    neu1.weights[0] += learning_rate * error * io.input1
    neu1.weights[1] += learning_rate * error * io.input2
    neu1.bias       += learning_rate * error   # bias 的「輸入」永遠是 1
```

- `learning_rate` 要調小: `0_clean` 的 `0.0562` 是單輸入時調的, 兩個輸入的梯度會疊加, 大概 `0.1` 附近就發散。從 **`0.02`** 開始, 跑 300 epoch 左右
- 收斂後直接看 `neu1.weights ≈ [2, 3]`、`bias ≈ 4` — 對得起來才代表它真的學到公式, 而不是硬背那 6 筆資料

---

## 第 5、6 階的重點

第 5 階最值得慢慢玩。`y = x²` 看起來像是「網路不夠深」, 但實際上改一行就好 — **換餵進去的東西**。

第 6 階接著證明: 多項式迴歸就只是線性 neuron 換個 feature 而已。餵 `[x², x]` 進去, 出來的會是 `weights[0]=3, weights[1]=2, bias=1`, 跟公式一模一樣。能直接從 `neu1.weights` 讀出真正的係數, 是很好的驗證方式。

第 6 階也會自己冒出兩個真問題, 兩個都值得踩:

- **scaling**: `x=7` 時 `x²=49`, 這個 weight 的梯度是另一個的 ~50 倍, learning rate 一定會炸。你會被逼著要嘛把 LR 調到很小, 要嘛把 feature 正規化 — 然後你就會親身感受到為什麼要做這件事
- 你在 `0_clean` 註解裡記的那個「發散點」, 到這裡會變成**每個 feature 各自有一個**, 不再是整個網路一個

---

## 兩個現有 code 的提醒

- `Neuron.__init__` 把所有 weight 設成 `0`。第 1~6 階完全沒問題 (線性迴歸只有一個最低點, 從哪開始都會走到同一個地方)。但第 7 階會被它害死: 每顆 hidden neuron 拿到的梯度一模一樣 → 永遠長得一樣 (symmetry)。**random init 到第 7 階才有必要**, 在它壞掉之前不用先改
- `Neuron.py` 的 `bias = 0` 寫了兩次 (第 3 行跟第 8 行), 沒影響但可以刪掉一行

## 建議的資料夾命名

`01_2x` → `02_2x_plus_3` → `03_two_inputs` → `04_n_inputs` → `05_x_squared` → `06_polynomial` → `07_sin_hidden`

照學習順序排, 比現在混在一起的編號好找。
