# Headless Training Pipeline Proposal

## 目標

把目前綁在 `nnCarGame.py` 的互動式 Pygame 訓練流程，拆成可重現、可批次、可 headless 執行的實驗 pipeline，支援：

- 同一次 experiment run 平行比較多種 fitness function。
- 用 `spec.md` 定義的 validation metric 做正式 evaluation。
- 完整記錄 seeds、config、訓練結果、validation 結果與 best weights。
- 將儲存的 weights 載入指定地圖做 replay，可選 GUI 或離線輸出。

## 現況與主要問題

目前訓練邏輯、Pygame UI、地圖生成、GA 交配、car physics、score 更新都寫在 `nnCarGame.py` 中，造成幾個問題：

- 無法 headless 批次跑實驗。
- 目前 baseline fitness 只有隱含的 `self.score += self.velocity`。
- 沒有固定 run manifest，validation seeds 也不會被記錄。
- 權重沒有標準化存檔格式，無法穩定 replay 或 server submission。

## 建議架構

### 1. Simulation Core

抽出不依賴畫面的核心模組，例如：

- `core/simulator.py`
- `core/car.py`
- `core/track.py`
- `core/metrics.py`

這層負責固定 timestep、collision、stall、spin、progress、finish 判定。Pygame 只保留為 optional renderer，不再是訓練主流程依賴。

### 2. Fitness Strategy Layer

新增 `fitness/` 目錄，每個策略是一個 config-driven module，例如：

- `speed_only_baseline`
- `speed_progress_v1`
- `centerline_safe_v1`
- `anti_stall_v1`

其中 `speed_only_baseline` 必須完整保留現有邏輯，作為所有實驗的比較基線：

```text
score += velocity
```

每個策略都輸出：

- training fitness
- per-episode diagnostics

但 validation / ranking 一律使用 spec 的正式 metric：

- `finished_within_30s`
- `finish_time`
- `max_track_progress`
- `collision_count`
- `stall_time`
- `spin_time`
- `reset_count`

### 3. Experiment Runner

新增 CLI runner，例如：

```bash
python -m pipeline.run_experiment --config configs/exp_baseline.yaml
```

每個 experiment run 會先產生一份 manifest：

- `run_id`
- train map ids / seeds
- validation map seeds
- test seeds
- model architecture version
- GA hyperparameters
- fitness strategy list

同一次 run 的所有 fitness strategies 共用同一組 validation / test seeds，必要時也共用同一組 initial population seed，確保比較公平。

### 4. Parallel Training

平行化粒度建議放在 strategy-level，而不是單一 simulation frame。

- 一個 strategy = 一個獨立 job
- job 內部自行跑 generations
- 最後統一收斂到同一份 run artifact

初版可先用 Python `multiprocessing`；之後若要擴大可換成 job queue。

## GA Parent Selection Baseline

為了先忠實重現現有互動式訓練邏輯，初版 breeding 規則固定為：

1. 每一代先依 training fitness 排序。
2. 取 fitness score 最高的兩台車作為 parents。
3. 用這兩台車產生下一代其餘車輛。
4. 這兩台 parents 以 elite 形式直接保留到下一代。

這條規則應套用到所有 fitness strategies，避免不同策略同時改動 selection 機制造成比較失真。後續若要加入 tournament selection 或 top-k sampling，應作為獨立 ablation，不與 baseline 混在一起。

### 5. Artifact Store

新增標準輸出結構：

```text
artifacts/
  runs/<run_id>/
    manifest.json
    summary.csv
    strategies/<strategy_name>/
      train_log.jsonl
      validation.json
      best_model.npz
      replay_candidates.json
```

`best_model.npz` 至少包含：

- network weights
- biases
- architecture version
- fitness config
- train map / seed
- generation
- best training fitness
- validation summary
- timestamp

## Evaluation 與 Model Selection

每個 generation 都記錄兩套結果：

1. training fitness
2. validation metric

best model 的選擇不應只看 training fitness，建議規則是：

1. `finished_within_30s` 優先
2. 若完賽，比 `finish_time`
3. 若未完賽，比 `max_track_progress`

這樣 selection 邏輯會和 spec 的正式 ranking 對齊，不會把 overfit 的高 fitness 模型誤判成最佳模型。

## Replay 設計

新增 replay runner：

```bash
python -m pipeline.replay --model artifacts/runs/<run_id>/strategies/<name>/best_model.npz --map train_seed_12
```

支援兩種模式：

- `--headless`: 只跑 simulation，輸出 metric 與 trajectory
- `--render`: 用 Pygame 將車放回地圖上播放，必要時輸出 frame sequence 或 mp4

replay 時必須使用與 validation 相同的 physics、timestep、collision 與 finish 規則。

## 分階段實作計畫

### Phase 1

先抽離 simulator、car state、track loading、metric 計算，讓單車 episode 可 headless 執行。

### Phase 2

把現有 GA 流程改成可程式化 runner，加入 strategy config、seed control、artifact logging。

### Phase 3

加入 multi-strategy parallel experiment runner、validation evaluator、best model export。

### Phase 4

加入 replay CLI 與簡單 leaderboard / summary report。

## 我接下來會怎麼做

如果你同意這個方向，下一步我會先把實作拆解成具體檔案結構與介面設計，優先定義：

- simulator API
- fitness strategy interface
- experiment config schema
- weight serialization format

這一步完成後，就可以開始實作最小可跑版本。
