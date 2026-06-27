# Neural Network Cars 自動化實驗 Pipeline

這個 pipeline 用 headless simulator 比較不同 fitness strategy。現在的主流程是 BeginnerMix：先用 presets 跑出可重現的基準實驗，再用 auto-tune 在 winners 周圍做座標搜尋，最後把表現好的配方整理成 `templates/`。

## BeginnerMix 工作流

1. 先跑 Stage 1 presets，產出基準 run。
2. 從 run 的 `manifest.json` 與 `summary.json` 擷取 winner 的 `rewards` / `penalties`。
3. 用 `pipeline.tune` 在 winner 周圍做 coordinate search。
4. 把最後的勝者用 `pipeline.export.promote_template()` 轉成 `templates/<name>/`。
5. 用 `pipeline.replay` 驗證模型在指定 seed 上可重播。

推薦的兩段命令如下：

```bash
# Stage 1: presets
uv run python -m pipeline.run_experiment --config configs/presets/starter_presets.json

# Stage 2: auto-tune around a winner（--out 會把調好的 winner 配方寫出，供 promote 使用）
uv run python -m pipeline.tune \
  --base-config configs/tune/auto_base.json \
  --out configs/tune/auto_winner.json \
  --rounds 2 --step 15

# Stage 3: 用調好的 winner 跑一次乾淨 run，再 promote 成 template
uv run python -m pipeline.run_experiment --config configs/tune/auto_winner.json
uv run python -c "from pipeline.export import promote_template; promote_template('<那個 run dir>', 'base', 'winner_v1')"
```

> 注意：`pipeline.tune` 只有加上 `--out` 才會把調校後的 winner 配方存檔。沒有 `--out` 時它只印出 winner，調校結果不會被保存，promote 就會誤用未調校的 base 配方。

更多 block 規格與調參方式請看 [docs/beginner_mix.md](docs/beginner_mix.md)，Colab 入口請看 [notebooks/run_experiments.ipynb](notebooks/run_experiments.ipynb)。

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
uv sync
```

這會建立或更新 `uv` 管理的虛擬環境，並安裝 `pyproject.toml` 裡宣告的套件。

## Config 欄位

現在的主要設定檔改成 starter presets 與 tune base config：

- `configs/presets/starter_presets.json`：Stage 1 的正式起手式。
- `configs/presets/smoke.json`：快速 smoke 測試。
- `configs/tune/smoke_base.json`：Stage 2 auto-tune 的 smoke 起點。

`ExperimentConfig` 仍然保留相同欄位結構，`strategies` 只是從舊的多元實驗改成 BeginnerMix 配方與兩個 baseline。

```json
{
  "run_name": "beginner_mix_presets",
  "output_dir": "artifacts/runs",
  "architecture": [6, 6, 4],
  "population_size": 20,
  "generations": 30,
  "mutation_rate": 90,
  "train_seeds": [101],
  "validation_seeds": [202, 203, 204],
  "time_limit_seconds": 30.0,
  "fps": 30,
  "parallel_workers": 6,
  "master_seed": 1234,
  "retry_generation": 15,
  "retry_min_avg_max_track_progress": 0.2,
  "max_seed_retries": 1,
  "track_cell_size": 120,
  "track_half_width": 34.0,
  "strategies": [
    { "name": "progress_first", "strategy": "beginner_mix" }
  ]
}
```

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

- `manifest.json`：本次實驗設定快照，包含 `git_commit`。
- `summary.csv`：跨 strategy 的主要排名表。
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
cat artifacts/runs/<run_id>/strategies/<strategy_name>/validation.json
```

查看每代趨勢：

```bash
cat artifacts/runs/<run_id>/strategies/<strategy_name>/train_log.jsonl
```

判讀順序應為：`finish_count`、`avg_finish_time`、`avg_max_track_progress`，最後才看 `avg_training_fitness`。不同 fitness strategy 的 raw training fitness 尺度不同，不能直接跨 strategy 比大小。

## 產出模板

當你有一個不錯的 run，可以用 `pipeline.export.promote_template()` 轉成 `templates/<name>/`。每個模板會有：

- `recipe.json`
- `reproduce.json`
- `result.json`
- `best_model.npz`
- `model.json`

`model.json` 是 final_goal 格式，`templates/index.json` 會記錄所有模板與它們的摘要。Colab notebook 會示範完整流程：presets → auto-tune → promote template。

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

## 舊素材

舊版遊戲腳本與歷史實驗素材已搬到 `archive/`。如果你要看舊的手動遊戲入口或歷史 run，請直接從 archive 讀取，不要把它們當成目前的正式 workflow。

# NNCars-Fitness-Experiments
