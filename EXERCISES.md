# 神經網路練習計畫

更新：2026-09-15。依現有程式與筆記重排。每次增加一個主要概念，先用能手算的小問題驗證，再擴大規模。

## 已走過的主線

| 階段 | 內容 | 現況 |
| :-- | :-- | :-- |
| 01 | `2x`：neuron 與學習迴圈 | `26_01_solve2x` 與筆記 |
| 02 | `2a+3b+4`：多輸入與 bias | `26_02_two_inputs` 與筆記 |
| 03 | `x²`：直線限制與 feature | `26_03_x_squared` 與筆記 |
| 04 | 多項式迴歸 | `26_04_polynomial` 與筆記；部分延伸題未勾完 |
| 05 | abs → 四顆 ReLU 的碗 + output bias | 已保存 `26_05_abs_bowl`；`0_clean` 保留，仍有殘差 |

`2x+3` 的 bias 已在後續題使用，不另重做。N 輸入的收納需求延到 N 顆 hidden。舊 sin 暫停紀錄是背景，目前沒有對應程式與筆記可直接接續。

## 接下來的順序

| 順序 | 練習 | 新觀念 | 完成證據 |
| :-- | :-- | :-- | :-- |
| 06A | 固定 `relu(x),relu(-x)`，學 `-abs(x)` | output weight 可以為負 | weight 約 -1，內插最大誤差 < 0.05 |
| 06B | 同一倒 V，放開 hidden | chain rule 多乘 output weight | 手算與七參數有限差分吻合，內插達標 |
| 07 | 同一題改為 N 顆 | list / array 管理參數 | 兩顆時 forward 與單步更新一致；換數量不用複製更新式 |
| 08 | sin：固定折點 → 學折點 | 從手設 feature 到學 feature | 比較直線與折線的 validation MSE，分開看外推 |
| 09 | 少量帶雜訊的 sin | 過擬合、validation、early stopping | train/validation 曲線與未調參的 test 結果 |
| 10 | 手寫版對照 PyTorch | tensor 與 autograd | 同資料、同參數的 forward / gradient 吻合 |
| 11 | 二元分類 → XOR | sigmoid、cross entropy、決策邊界 | 先解線性可分題，再用 hidden 解 XOR，畫平面輸出 |

未來階段均為待做。Python／AI 練習固定在 `0_clean`；使用者表示 OK 後，直接建立 `26_xx_topic` 快照並驗證，保留練習區。每篇練習筆記末尾附完整 source 或實際入口與支援檔路徑，並隨歸檔更新。此流程不套用到其他學習主題。

## 練倒 V：拆開新增概念

### 固定 hidden，只學 output

- 訓練 `x=-3..3`、間隔 0.5，答案 `-abs(x)`。
- hidden 固定 `relu(x),relu(-x)`，兩個 output weight 與 bias 從 0 開始。
- step size 先試 0.01、1000 epoch；每 epoch 結束以同一組參數重算 MSE。
- 驗證 `-2.75,-1.25,0.25,2.25`，最大絕對誤差 < 0.05；外推 `-5,5` 另記。
- 這相當於線性迴歸換 feature，先看負權重如何把 V 翻轉。

### 放開 hidden，驗算責任

- hidden weight `1,-1`、hidden bias `0,0`、output weight `-0.5,-0.5`、output bias `0`。先用接近解的起點排除初始化干擾，再試其他起點。
- forward → 算全部梯度 → 更新全部參數；hidden gradient 必須乘更新前的 output weight。
- 單筆 loss 用 `0.5*(predict-answer)**2`。七參數中心差分 epsilon `1e-5`，避開 ReLU 折點，導數絕對差先要求 < `1e-5`。
- 詳細推導在 wiki `06 output weight — 讓折線能加也能減`。

## 管理 N 顆：只改收納方式

保持兩顆、同初值、同資料順序，把 hidden weights、hidden biases、output weights 改成 list / array。比對 forward 與單筆更新誤差 < `1e-10`，再試四顆。

此時不一起換目標、optimizer 或初始化，方便定位差異。先會用迴圈，再考慮矩陣化，不急著寫通用框架。

## 擬合 sin：先固定折點，再學折點

1. 用 `np.linspace(-np.pi,np.pi,41)` 訓練，40 個相鄰中點作 validation。
2. 先 fit 直線，記錄 validation MSE 作基準。
3. 用 8 顆向右開的 ReLU，折點 `np.linspace(-np.pi,np.pi,8,endpoint=False)`；hidden weight=1、hidden bias=-折點，只學 output weight 與 bias。左端折點提供區間的基礎斜率。
4. 放開 hidden，接續上述參數，維持同資料與評估；先確認能繼續降低誤差，再另做初始化實驗。
5. 比較 4、8、16 顆；隨機初始化另外跑 seeds `0..4`，記錄每組與中位數，不只展示最佳一次。
6. 教學目標先設 validation MSE < 0.01，且低於直線基準；未達標先查梯度，再一次改一個設定。
7. 獨立觀察 `[-2pi,2pi]` 外推：有限顆 ReLU 離開最外側折點後是直線，沒有內建週期性。

這裡先不增加 hidden layer 深度。帶正負 output weight 的折線已能表示上彎與下彎。

## 觀察過擬合

固定 seed，建立 21 個 `[-pi,pi]` 訓練點，答案加標準差 0.1 的常態雜訊。validation 用另一批區間內點；這個合成題可用無雜訊 sin 當驗證答案，觀察是否學到真曲線。

比較 4 與 32 顆、短與長訓練，畫 train/validation loss。保存 validation 最佳參數，另留獨立 test 點，只在選定設定後評估。若沒有出現過擬合，就如實記錄，不強迫符合故事。

## 後續方向

PyTorch 階段再確認官方安裝方式與 API，不在這裡綁定版本。分類階段另外推導 sigmoid、cross entropy 與數值穩定性。

基礎完成後依興趣分支：

- 強化學習：小型環境 → Q-table → 探索 → 函數近似。
- 語言模型：token → embedding → next-token prediction → attention。

強化學習不是 LLM 的必要先修。現在不必展開遠期課程。

## 每階的紀錄與判斷

記錄目標、模型、資料範圍、初值 / seed、資料順序、更新方式、step size、epoch、train / validation 指標與失敗解釋。上述門檻是教學設定，不是跨任務標準。

loss 卡住不一定是 local minimum；加顆數與 random init 不保證收斂。線性最小平方的參數唯一性需要資料矩陣滿秩；步長穩定性受整組 feature 尺度與相關性影響，並非各 feature 有獨立發散點。

最佳化與泛化的區別參考作者教材：[Deep Learning, Chapter 8](https://www.deeplearningbook.org/contents/optimization.html)。本計畫順序與門檻是依本專案進度設計。
