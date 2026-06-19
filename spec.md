# Neural Network Cars

## 目標

Neural Network Cars 是一個教學用賽車訓練遊戲。使用者可以設計與調整不同的 fitness function 策略，透過 Genetic Algorithm（GA）訓練 Neural Network Cars，並比較哪些策略能在未見過的賽道上取得最好的泛化效果。

核心目標不是只讓車在訓練地圖上取得高 fitness score，而是讓使用者理解不同 reward / penalty 設計如何影響車輛行為，最後訓練出能在 unseen map 上快速完賽的模型。

## 使用者流程

1. 使用者選擇訓練地圖。
2. 使用者選擇或調整 fitness function 策略。
3. 系統使用 GA + NN Cars 進行訓練。
4. 訓練過程中，系統每一 round 顯示訓練 fitness 與比賽 metric。
5. 使用者可以儲存不同策略訓練出的權重。
6. 使用者可以在 validation 區域載入權重，讓車在 unseen map 上測試。
7. 使用者確認結果後，可以將權重上傳到 server 進行正式測試與排名。

## Fitness 與比賽 Metric

Fitness score 是訓練訊號，可以包含多種 shaping rewards，例如速度、前進進度、貼近賽道中心、碰撞懲罰、stall 懲罰與 spinning 懲罰。

比賽 metric 是最終排名規則，應保持簡單、可驗證。兩者目標一致，但不應使用同一套複雜加權公式。

訓練與 validation UI 應同時顯示：

- 目前策略的 training fitness score。
- 該 round 使用比賽 metric 得到的結果。
- 是否在 30 秒內完賽。
- 若完賽，顯示 finish time。
- 若未完賽，顯示 max track progress 作為參考。

## 地圖設計

系統應支援多張可選訓練地圖，讓使用者能在不同地圖上訓練與比較策略。

Unseen map 分成兩類：

- Validation unseen maps：本地 validation 使用，讓使用者檢查模型是否泛化。
- Server hidden unseen map：正式排名使用，使用者訓練與 validation 時不可直接存取。

Validation metric 應與 server final metric 一致，避免使用者在本地看到的成績與正式排名邏輯不同。

每次新的實驗 run 應抽取一組新的 validation / test track seeds，避免長期只對固定 unseen maps 調參。為了公平比較，同一次實驗 run 中的所有 fitness strategies 必須使用同一組 validation / test seeds。實際使用的 seeds 必須寫入實驗結果，讓該次結果可以重現。

## 權重儲存

使用者可以儲存不同策略訓練完成的權重。每個儲存結果應包含：

- Neural network weights。
- Model architecture version。
- Fitness strategy config。
- Training map / seed。
- Generation count。
- Training fitness score。
- Validation result。
- Timestamp 或 run id。

Model architecture version 必須隨權重一起儲存，讓 validation runner 與 server 能正確載入模型。

## Validation

Validation 區域用來測試已儲存的權重。使用者可以選擇一組權重，將車放到 validation unseen map 上執行，並看到與正式比賽一致的 metric。

Validation 結果至少應包含：

- `finished_within_30s`
- `finish_time`
- `max_track_progress`
- `collision_count`
- `stall_time`
- `spin_time`
- `reset_count`

其中 `finish_time` 只在 30 秒內完賽時作為排名成績；未完賽時，`max_track_progress` 只作為診斷與參考。

## Server Submission

使用者最終可以將權重上傳到 server 進行正式測試。Submission 至少應包含：

- 使用者名稱或 team id。
- Model architecture version。
- Network weights。

Server 負責載入權重，在 hidden unseen map 上執行固定規則測試，並產生 leaderboard。

## 正式測試與排名規則

正式測試使用一張 hidden unseen map，時間上限為 30（可以再改） 秒。

排名規則：

1. 30 秒內完賽者排在未完賽者前面。
2. 完賽者依 `finish_time` 排名，時間越短排名越高。
3. 未完賽者不計正式完賽成績，可在 leaderboard 或 detail view 中顯示 `max_track_progress` 作為參考。
4. 若平手則直接並列名次，不需要tie-break

正式排名的核心分數不是訓練 fitness score，而是：

```text
finished_within_30s
finish_time
```

## 公平性與可重現性

Server 測試必須固定以下條件：

- 同一張 hidden unseen map。
- 同一個起點與初始方向。
- 同一個時間限制。
- 同一套 physics 與 simulation timestep。
- 同一套 collision、stall、spin 與 finish 判定。
- 可重現的 random seed 或 deterministic runtime。

這些條件應與 validation runner 盡量一致，差別只在 server 使用 hidden unseen map。
