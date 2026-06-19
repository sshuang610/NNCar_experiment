# Neural Network Cars 自動化實驗 Pipeline

這個 pipeline 用 headless simulator 比較不同 fitness strategy。每個 strategy 都會獨立執行 genetic algorithm（GA），每一代用 training track 選父母，再用固定的 validation tracks 選出該 strategy 最值得保存的模型。

## 目前結論

依據 `artifacts/runs/20260619T090419Z_race_metric_focused_30/summary.csv`：

- `race_metric_proxy` 是唯一能在 validation 完賽的策略：3 個 seed 中完成 2 個，完賽平均時間 29.12 秒。
- seed `202` 只跑到 17.15%，表示目前模型仍有明顯的跨賽道穩定性問題。
- 其餘 9 個策略都未完賽，平均最大進度只有 3.95% 到 8.15%。
- `configs/experiment_focused.json` 已移除這 9 個失敗策略，改成保留 `race_metric_proxy` benchmark、同時探索 7 種不同 reward 結構。

失敗策略的 Python class 仍保留，僅從正式實驗 config 移除。舊模型 replay 時會根據 metadata 裡的 strategy name 建立 fitness strategy；刪除 class 會讓既有 artifact 無法重播。

## Pipeline 整體流程

```text
experiment JSON
    |
    v
載入 seeds、population、generations、strategies
    |
    v
每個 strategy 用 master_seed 與名稱產生可重現的初始 population
    |
    v
每一代：
  1. 所有 network 跑 train_seeds
  2. 依平均 training fitness 排序
  3. 取前兩名 crossover + mutation，產生下一代
  4. 當代 training 第一名跑 validation_seeds
  5. 若 validation ranking 更好，覆寫 best_model
    |
    v
輸出 train_log、validation、best_model、summary
    |
    v
用 replay 在指定 seed 重播最佳模型
```

訓練 fitness 只負責引導 GA；模型保存則依 validation 指標判斷：

1. `finish_count` 越多越好。
2. 完賽數相同時，平均完賽時間越短越好。
3. 前兩項相同時，平均最大賽道進度越高越好。

## 安裝環境

從 repository 根目錄執行：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pygame pillow shapely
```

## 跑實驗

快速確認 pipeline 可執行：

```bash
python3 -m pipeline.run_experiment \
  --config tmp_configs/smoke_experiment.json
```

確認 8 個 strategy worker 可以平行啟動：

```bash
python3 -m pipeline.run_experiment \
  --config tmp_configs/parallel_strategy_smoke.json
```

執行目前建議的 30-generation 多元策略實驗：

```bash
python3 -m pipeline.run_experiment \
  --config configs/experiment_focused.json
```

這是速度最快的執行方式。`parallel_workers` 目前設為 `8`，8 個 fitness strategy 會由 `ProcessPoolExecutor` 分配到最多 8 個獨立 process；每個 strategy 內部的 generations 仍依序演化。

需要即時 HTML dashboard 時加上 `--render`：

```bash
python3 -m pipeline.run_experiment \
  --config configs/experiment_focused.json \
  --render
```

`--render` 也會讓策略平行執行，但每一代會額外跑一次 trajectory rollout 並重寫 dashboard，因此純粹追求訓練速度時不要加這個選項。

完成後，終端機會印出新的 run 目錄，例如：

```text
artifacts/runs/20260619T120000Z_diverse_strategy_30
```

## 目前的多元策略實驗

`race_metric_proxy` 每一個 simulation step 的分數為：

```text
6.0 * progress_delta
+ 0.4 * velocity
+ 0.5 * progress_ratio
- 0.03 * time_elapsed
- stall/spin penalty
+ finish/collision event score
```

`configs/experiment_focused.json` 現在比較以下策略：

| Strategy | Reward 核心 | 想驗證的假設 |
| --- | --- | --- |
| `race_metric_proxy` | 進度、速度、目前位置、時間與事件的 dense reward | 保留目前唯一能完賽的 benchmark |
| `frontier_explorer` | 只重獎超越本次 episode 歷史最遠位置 | 避免停在後段持續累積 `progress_ratio` 分數 |
| `risk_adjusted_pace` | 前方空間越大，速度 reward 越高 | 學會直線加速、入彎主動降速 |
| `centerline_pace` | 速度乘上方向對齊與中心線信心 | 找到穩定且能泛化的 racing line |
| `two_phase_racer` | 前半段重安全與進度，後半段提高速度權重 | 先學會存活，再學會衝刺 |
| `smooth_control_pace` | 懲罰過度轉向與原地旋轉 | 減少左右震盪造成的速度損失 |
| `sparse_outcome` | 主要依新進度與終點事件計分 | 減少人工 shaping，直接最佳化完成比賽 |
| `sensor_balance_pace` | 前方空間 bonus 與左右感測器不平衡 penalty | 只靠車上可觀測資訊學會留在安全走廊 |

這些策略不是 `race_metric_proxy` 的係數微調，而是分別測試 frontier、risk、geometry、phase curriculum、control smoothness、sparse reward 與 sensor shaping。每個 strategy 的 random seed 由 `master_seed + strategy name offset` 決定，因此相同設定可以重現，也能保留既有 `race_metric_proxy` benchmark 的初始條件。

## Config 欄位

主要實驗設定在 `configs/experiment_focused.json`：

```json
{
  "run_name": "diverse_strategy_30",
  "output_dir": "artifacts/runs",
  "architecture": [6, 6, 4],
  "population_size": 20,
  "generations": 30,
  "mutation_rate": 90,
  "train_seeds": [101],
  "validation_seeds": [202, 203, 204],
  "time_limit_seconds": 30.0,
  "fps": 30,
  "parallel_workers": 8,
  "master_seed": 1234,
  "retry_generation": 15,
  "min_completion_rate": 0.2,
  "max_seed_retries": 1,
  "track_cell_size": 120,
  "track_half_width": 34.0,
  "strategies": [
    { "name": "race_metric_proxy" }
  ]
}
```

- `population_size`：每個 strategy 每代的 network 數量。
- `generations`：每個 strategy 的演化代數。
- `mutation_rate`：breeding 時修改的 weight / bias 數量。
- `train_seeds`：計算 training fitness 的賽道。
- `validation_seeds`：不參與父母選擇，只用於模型選擇與比較。
- `parallel_workers`：headless 模式同時訓練的 strategy process 數量；通常設為 `min(CPU logical cores, strategy 數量)`。
- `master_seed`：初始 network 與 mutation 的基礎 seed；pipeline 會加上 strategy name 的固定 offset，讓每個策略可重現且彼此獨立。
- `retry_generation`：在這一代檢查該 attempt 至今最佳 validation 完賽率。
- `min_completion_rate`：低於這個完賽率時安排新的 evolution seed。
- `max_seed_retries`：最多額外執行幾次；設為 `1` 表示最多兩個完整 attempts。
- `time_limit_seconds` / `fps`：episode 時限與 simulation timestep。
- `track_cell_size` / `track_half_width`：程序化賽道尺寸。

目前只有一個 training seed，適合快速篩選 fitness。找出下一個勝者後，應建立第二階段 config，增加 training seeds 和全新的 validation seeds 來檢查泛化能力。

## 自動更換 Evolution Seed

目前正式 config 會在 generation 15 檢查：

```text
best completion rate through generation 15
  = best validation finish_count / validation seed count
```

若結果低於 `0.2`，pipeline 會先讓原 attempt 跑完 30 generations，再以 `evolution_seed + 1` 完整重跑一次，最後從兩次 attempts 選 validation ranking 最好的模型。保留第一個 attempt 是必要的，因為已知 `race_metric_proxy` 要到 generation 25 才首次完賽。

目前有 3 個 validation seeds，所以 `0.2` 實際上代表 generation 15 前完全沒有任何 validation 完賽。觸發 retry 時，每個 strategy 最多執行 60 generations，因此總時間可能接近兩倍。

## 輸出與看結果

每次執行都建立新目錄，不會覆蓋舊實驗：

```text
artifacts/runs/<run_id>/
  manifest.json
  summary.csv
  summary.json
  dashboard.html
  strategies/
    <strategy_name>/
      train_log.jsonl
      validation.json
      best_model.npz
```

- `manifest.json`：本次實驗設定快照。
- `summary.csv`：跨 strategy 的主要排名表，包含最佳 attempt、evolution seed、retry 狀態、完賽數、時間、進度、碰撞、stall 與 spin。
- `train_log.jsonl`：每一代的 training 與 validation 結果。
- `validation.json`：最佳模型在每個 validation seed 的詳細結果。
- `best_model.npz`：最佳 network 權重與 replay metadata。
- `dashboard.html`：使用 `--render` 時產生的即時頁面。

快速比較：

```bash
cat artifacts/runs/<run_id>/summary.csv
```

查看最佳策略在每個 seed 的表現：

```bash
cat artifacts/runs/<run_id>/strategies/race_metric_proxy/validation.json
```

查看每代趨勢：

```bash
cat artifacts/runs/<run_id>/strategies/race_metric_proxy/train_log.jsonl
```

判讀順序應為：`finish_count`、`avg_finish_time`、`avg_max_track_progress`，最後才看 `avg_training_fitness`。不同 fitness strategy 的 raw training fitness 尺度不同，不能直接跨 strategy 比大小。

## Replay

從 `summary.csv` 取得 `best_model_path`，再指定一個 track seed：

```bash
python3 -m pipeline.replay \
  --model artifacts/runs/<run_id>/strategies/<strategy_name>/best_model.npz \
  --seed 202
```

指定輸出位置：

```bash
python3 -m pipeline.replay \
  --model artifacts/runs/<run_id>/strategies/<strategy_name>/best_model.npz \
  --seed 202 \
  --output-dir artifacts/replays/<run_id>
```

Replay 會產生：

```text
artifacts/replays/<run_id>/<strategy>_best_model_seed_<seed>.svg
artifacts/replays/<run_id>/<strategy>_best_model_seed_<seed>.json
```

`.svg` 用來檢查軌跡與撞牆位置，`.json` 包含 finish time、max progress、collision、stall 與 spin 等指標。使用 manifest 內的 validation seed 可以重現結果；使用新的 seed 可以做額外泛化測試。

## 下一輪篩選標準

完成 `diverse_strategy_30` 後：

1. 淘汰 `finish_count == 0` 且平均進度低於 control 的策略。
2. 優先保留 validation 完賽數較多的策略，不要只看單次最快時間。
3. 完賽數相同時，再比較平均完賽時間與碰撞數。
4. Replay 所有 validation seeds，確認高分不是錯誤路徑或偶發 trajectory。
5. 最佳 1 到 2 個策略使用更多 train seeds 與未見過的 validation seeds 再跑一次。

新增策略時，在 `pipeline/fitness.py` 建立 `FitnessStrategy` subclass、實作 `score_step()`，並註冊到 `STRATEGIES`。正式比較前先把它加入 smoke config，確認可以完整產生 model、summary 與 replay。
# NNCars-Fitness-Experiments
