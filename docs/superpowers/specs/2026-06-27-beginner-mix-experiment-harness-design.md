# BeginnerMix 10-param 自動化實驗 Harness — 設計規格

狀態：**草稿，待 user review**
日期：2026-06-27
範圍：`NNCars-Fitness-Experiments` repo（GA Fitness Function Research 組的實驗 harness）

---

## 0. 目標與脈絡

把 `NNCars-Fitness-Experiments` 從「以數十個 hardcoded fitness strategy class 為實驗單位」改成
「以 [beginner_mix_10param_design.md](../../../../../Neural-Network-Cars/docs/beginner_mix_10param_design.md) §9 定案的 block 化 10 參數為實驗單位」，
跑自動化實驗找出**可重現的優秀參數組合（template）**，最終服務於
[final_goal.md](../../../../../Neural-Network-Cars/docs/final_goal.md) 中 Group B（GA Fitness）的交付：
推薦 fitness preset 與權重範圍建議。

### 已確認的決策（來自 brainstorming）

| 決策 | 選擇 |
| --- | --- |
| 參數探索方式 | **先預設配方（presets）再自動微調（auto-tune）** |
| 計分模型版本 | **§9 定案 block 模型**（鼓勵自動歸一 + 懲罰獨立 + 每秒 `×dt` + crash 一次性 + 固定 `FINISH_BONUS`） |
| 既有 ~40 個 strategy class | **用 `BeginnerMix` 取代**，保留少數 baseline 當控制組 |
| 舊輸出與雜亂檔 | **封存舊 run + 刪除雜亂檔，重整結構** |
| Colab 程式碼來源 | user 自己開新 repo 上傳；notebook 以**可設定的 repo URL** clone |
| template 內容 | block 配方 + 重現 metadata + 成績 + **一併存訓練好的 best model** |
| 保留的 baseline | `speed_only_baseline` + `progress_only` |
| 封存 vs 刪除 | 舊 `artifacts/runs`、`tmp_configs`、舊 `configs`、legacy 遊戲檔 → **移到 `archive/`** |

---

## 1. 現況關鍵事實（實作前必讀）

1. **設計文件描述的系統尚未存在**：`beginner_mix_10param_design.md` 描述的 block 化 `BeginnerMix`、`configure()`、
   `configs/experiment_beginner_mix.json`、`tests/test_beginner_mix.py`、metadata 內的 `strategy_params` ——
   在本 repo 都還沒有。文件的 `StepContext` / per-step 計分架構對應的是**本 repo 的 pipeline**
   （非 `Neural-Network-Cars/GA/fitness.py`，那個是 whole-car 計分）。換言之，該文件實質上是**本 pipeline 的規格**。
2. **`params` 宣告了但沒接線**：[`config.py`](../../pipeline/config.py) 的 `StrategyConfig.params` 存在，但
   [`training.py`](../../pipeline/training.py) 呼叫 `build_strategy(name)` 時沒帶 params。所以文件宣稱
   「params 串接管線已完成」**在本 repo 不成立**，需要實作。
3. **目前實驗單位是 hardcoded class**：[`fitness.py`](../../pipeline/fitness.py) 內 ~40 個 `FitnessStrategy` subclass，
   靠 `STRATEGIES` dict 以名稱選取。本次改為「選一個 block 配方（rewards/penalties 權重）」。
4. **`StepContext` 已具備 block 模型所需全部欄位**：見 [`simulator.py:193`](../../pipeline/simulator.py)
   —— `velocity`、`progress_delta`、`progress_ratio`、`normalized_center_offset`、`heading_alignment`、
   `min_clearance`、`is_stalled`、`is_spinning`、`collided`、`finished`、`frame`、`time_elapsed` 全有。
   `dt = time_elapsed / frame`，可在 strategy 內推導，達成 fps 無關。
5. **`build_strategy(name)` 被呼叫於 3 處**：training（`_evaluate_network`、`_render_payload`）與 `replay.py`。
   全部要改成帶 params。
6. **model 儲存格式**：[`storage.py`](../../pipeline/storage.py) `save_model()` 以 npz 存 `weight_*` / `bias_*` /
   `sizes` / `metadata_json`。需把 `strategy` 與 `strategy_params` 寫入 metadata，讓 replay 完全重現。
7. **架構 `[6,6,4]`** 對齊 final_goal 的 model JSON：`weights[0]` (6×6)=36、`weights[1]` (4×6)=24、
   `biases[0]` (6×1)=6、`biases[1]` (4×1)=4。匯出 template model 時依此 flatten。

---

## 2. Fitness 模型：`BeginnerMix`（§9 block 模型）

### 2.1 計分公式

令 `dt = time_elapsed / frame`（fps 無關）。常數：`B = 10`、`CRASH_SECONDS = 15`、
`B_CRASH = B × CRASH_SECONDS = 150`、`FINISH_SECONDS = 300`、`FINISH_BONUS = B × FINISH_SECONDS = 3000`。

```text
# 鼓勵側：自動歸一 → 固定每秒預算
reward = ( Σ_鼓勵 wᵢ·factorᵢ / Σ_鼓勵 wᵢ ) × B × dt          # 無鼓勵 block 時 reward = 0

# 避免側（每幀型）：各自獨立，滑桿 = 該行為最大強度百分比
penalty = Σ_避免每幀 (wⱼ / 100) × B × factorⱼ × dt

step = reward − penalty
if collided: step -= (w_crash / 100) × B_CRASH               # 一次性
if finished: step += FINISH_BONUS                            # 固定，不在 block 內
```

### 2.2 factor 對照（StepContext 來源）

**鼓勵 block（rewards，0~100，互相競爭、自動歸一）**

| block | factor | StepContext 來源 |
| --- | --- | --- |
| `speed` | `velocity / 10` | `velocity` |
| `progress` | `min(progress_delta / 10, 1)` | `progress_delta` |
| `centered` | `max(0, 1 − normalized_center_offset)` | `normalized_center_offset` |
| `alignment` | `max(0, heading_alignment)` | `heading_alignment` |
| `safety` | `min(min_clearance, 90) / 90` | `min_clearance` |

**避免 block（penalties，0~100，彼此獨立）**

| block | 類型 | factor | StepContext 來源 |
| --- | --- | --- | --- |
| `stall` | 每幀 | `1.0 if is_stalled else 0` | `is_stalled` |
| `spin` | 每幀 | `1.0 if is_spinning else 0` | `is_spinning` |
| `wrong_way` | 每幀 | `1.0 if heading_alignment < 0 else 0` | `heading_alignment` |
| `time` | 每幀 | `1.0`（每幀都成立） | —（恆真） |
| `crash` | 一次性 | 套 `B_CRASH`（見公式） | `collided` |

### 2.3 `configure()` 介面（向後相容）

接受兩種輸入（§9.3）：

```python
# 新格式
{ "rewards": {"progress": 40, "speed": 30, "safety": 30},
  "penalties": {"stall": 60, "crash": 80} }

# 舊 flat（rewards-only）→ 視為 rewards
{ "speed": 30, "progress": 40, "centered": 10, "alignment": 10, "safety": 10 }
```

- 只計入有出現的 block；沒加的不計分。
- penalty 滑桿夾 `>= 0`（避免「負懲罰變獎勵」footgun）。
- 未知 block key 忽略。

### 2.4 與既有 baseline 並存

`fitness.py` 精簡為：`BeginnerMix` + `SpeedOnlyBaseline` + `ProgressOnly`（控制組）。
其餘 ~38 個 class 移除（連同舊 config / 舊 run 一起封存；replay 相容性刻意捨棄，已封存故可接受）。

---

## 3. Config schema 變更

`StrategyConfig` 新增 `strategy` 欄位（fitness 型別），`name` 變成唯一 label / 輸出資料夾名：

```json
{
  "name": "progress_first",
  "strategy": "beginner_mix",
  "params": {
    "rewards":   {"progress": 40, "speed": 30, "safety": 30},
    "penalties": {"stall": 60, "crash": 80}
  }
}
```

- `strategy` 預設 `"beginner_mix"`；baseline 用 `"speed_only_baseline"` / `"progress_only"`（其 params 忽略）。
- `build_strategy(strategy_type: str, params: dict) -> FitnessStrategy`：
  - `beginner_mix` → `BeginnerMix()` 後 `configure(params)`。
  - baseline → 直接回傳，忽略 params。
- 向後相容：舊 config（只有 `name`、無 `strategy`）→ 若 `name` 命中既有 baseline class 則沿用，否則預設 `beginner_mix`。

---

## 4. 兩階段實驗流程（presets → auto-tune）

### 4.1 Stage 1 — presets

`configs/presets/starter_presets.json` 列出數組手工設計配方，至少涵蓋 final_goal §3.2 要求的三類起手式：

| preset | 重點 | 大致配方 |
| --- | --- | --- |
| `progress_first` | 最大進度 | rewards 重 `progress`，penalties 開 `stall`/`crash` |
| `speed_first` | 完成後圈速 | rewards 重 `speed`+`alignment`，penalties 開 `crash`+`time` |
| `stable_generalist` | 兼顧進度/穩定/懲罰 | rewards 均衡，penalties 開 `stall`/`spin`/`crash` |
| （再加 2~3 組探索 `centered`/`safety`/`wrong_way` 的變體） | | |

既有 GA harness（[`training.py`](../../pipeline/training.py)）逐一訓練，依 validation ranking 排序：
`finish_count` → `avg_finish_time`（短佳）→ `avg_max_track_progress`（高佳）。

### 4.2 Stage 2 — auto-tune（`pipeline/tune.py`，新增）

對 Stage 1 winner 做**座標搜尋（coordinate search）**：

1. 以 winner 配方為 base。
2. 對每個 active slider，產生 `±step`（例如 ±15，夾在 0~100；crash 用較大 step）的鄰居配方。
3. 每個鄰居當成一個 strategy 丟進既有 `run_experiment`（沿用平行訓練）。
4. 取 validation ranking 最佳者為新 base；可重複 N 輪（round）。
5. 全程用固定 `master_seed`，每個鄰居名稱穩定 → 完全可重現。

選座標搜尋而非 random search：可解釋（知道是哪個 slider 改善）、可重現、收斂行為穩定。
`tune.py` 也提供 CLI：`--base-config`、`--rounds`、`--step`、`--promote-top K`。

---

## 5. Template 交付物

每個被提拔的 winner → 一個 commit 進 repo 的 `templates/<name>/`：

| 檔案 | 內容 |
| --- | --- |
| `recipe.json` | `{rewards, penalties}` 配方 |
| `reproduce.json` | 完整 experiment config + seeds + **git commit hash** + pipeline 版本（重跑所需一切） |
| `result.json` | validation 成績摘要（finish_count / avg_finish_time / avg_max_track_progress / collision / stall / spin） |
| `best_model.npz` | pipeline 原生權重（含 metadata 內的 strategy_params，可直接 replay） |
| `model.json` | best model 以 **final_goal 格式**匯出（`group_id`/`username`/`weights`/`biases`，flatten 36/24/6/4） |

另：`templates/index.json` 彙整所有 template + headline metrics，方便挑選。

新增 `pipeline/export.py`：
- `export_final_goal_model(npz_path, group_id, username) -> dict`：載入 npz，flatten 成 final_goal model JSON。
- `promote_template(run_dir, strategy_name, template_name, ...)`：從一次 run 把指定 strategy 打包成 template 資料夾。

重現性保證：`reproduce.json` 記錄 git commit；`manifest.json` 也加上 git commit 欄位。
固定 `master_seed` + 穩定的 strategy-name offset（[`training.py:29`](../../pipeline/training.py)）→ 同 commit + 同 config = 同結果。

---

## 6. Colab Notebook

`notebooks/run_experiments.ipynb`：

1. **設定**：可填 `REPO_URL`（user 的新 repo）、`REPO_BRANCH`。
2. **取得程式碼**：`git clone $REPO_URL`（private 時提示用 token）。
3. **安裝**：core pipeline 只需 `numpy`（pygame 僅 legacy 遊戲用、已封存）→ Colab 上 `pip install numpy` 即可。
   實作前驗證 core pipeline 不 import pygame。
4. **Stage 1**：跑 `configs/presets/starter_presets.json`，用 pandas 顯示 `summary.csv`。
5. **Stage 2**：對 winner 跑 `pipeline.tune`，顯示每輪改善。
6. **提拔 template**：呼叫 `promote_template`，顯示 `templates/index.json` 與每個 `result.json`。
7. **（選配）**：inline 顯示 best model 的 trajectory（沿用 [`render.py`](../../pipeline/render.py) SVG）。
8. **下載**：把 `templates/` 打包 zip 供下載；或提示 push 回 repo。

---

## 7. 檔案結構重整（in place）

```
NNCars-Fitness-Experiments/
  pipeline/                 # 核心（擴充）
    fitness.py              # 精簡為 BeginnerMix + speed_only_baseline + progress_only
    config.py               # StrategyConfig 加 strategy 欄位
    training.py             # build_strategy 帶 params；manifest 加 git commit
    simulator.py replay.py render.py nn.py track.py storage.py paths.py visualize.py
    tune.py                 # 新增：座標搜尋 auto-tune driver
    export.py               # 新增：final_goal model 匯出 + template 打包
    run_experiment.py
  configs/
    presets/starter_presets.json
    tune/                   # auto-tune 設定（base config 等）
  templates/                # 交付：commit 進 repo 的可重現 winners
    index.json
    <name>/ recipe.json reproduce.json result.json best_model.npz model.json
  artifacts/                # gitignore 的暫時 run 輸出
    runs/ replays/
  notebooks/run_experiments.ipynb
  tests/
    test_beginner_mix.py    # 新增（TDD）
    test_tune.py            # 新增
  docs/
    beginner_mix.md         # block 模型 how-to（調參指南 + 重現流程）
    superpowers/specs/      # 本規格
  archive/                  # 封存
    runs/                   # 舊 artifacts/runs（9 個舊 run）
    configs/                # 舊 configs/* + tmp_configs/*
    legacy_game/            # nnCarGame.py mapGen.py Images/ bg4.png bg7.png randomGeneratedTrack*.png
  README.md AGENTS.md pyproject.toml uv.lock .gitignore
```

- 舊 `artifacts/runs`（9 個）、`tmp_configs/`、舊 `configs/`、legacy 遊戲檔 → 移到 `archive/`。
- 散落 PNG（`bg4.png`、`bg7.png`、`randomGeneratedTrack*.png`）→ 移到 `archive/legacy_game/`。
- `.gitignore` 把 `artifacts/` 設為忽略（run 輸出視為 ephemeral）；`templates/` 入版控。
- git remote / 新 repo 由 user 處理；本次只重整檔案。

---

## 8. 測試（TDD）

延伸現有 `tests/`（目前 repo 無 tests 目錄，將新建）：

- `test_beginner_mix.py`
  - 向後相容：flat rewards-only dict == 純鼓勵行為。
  - 校準：任一 penalty 拉到 100，對應壞行為發生該幀可抵銷全部 reward（reward 與單一 maxed penalty 同尺度）。
  - 迴歸：多加鼓勵 block 不稀釋既有懲罰效果（自動歸一）。
  - crash 一次性、finish 固定大獎。
  - fps 無關：同情境換 fps（30 vs 60）每秒行為一致（`×dt`）。
  - penalty 夾 `>= 0`。
- `test_tune.py`
  - 座標搜尋產生正確鄰居集合、夾 0~100、可重現（同 seed 同結果）。
- `test_export.py`
  - final_goal flatten 形狀正確（36/24/6/4）、round-trip 不失真。

---

## 9. 實作順序（給後續 writing-plans）

1. `BeginnerMix` + `configure()`（TDD）→ 精簡 `fitness.py`、保留 2 baseline。
2. `build_strategy(strategy_type, params)` 接線（config / training / replay）+ metadata 寫入 `strategy_params`。
3. `configs/presets/starter_presets.json`。
4. `pipeline/export.py`（final_goal 匯出 + `promote_template`）+ manifest git commit。
5. `pipeline/tune.py`（座標搜尋）。
6. 檔案結構重整（建 `archive/`、搬移、改 `.gitignore`）。
7. `notebooks/run_experiments.ipynb`。
8. `docs/beginner_mix.md` how-to + 更新 `README.md`。
9. 跑一次完整 Stage1→Stage2→promote，產出首批 templates 當驗收。

---

## 10. 驗收標準

- `BeginnerMix` 通過 §8 全部測試（含校準、迴歸、fps 無關）。
- `configs/presets/starter_presets.json` 能跑出 summary 與可 replay 的 best_model。
- Stage 2 auto-tune 能在固定 seed 下重現，且能對 winner 產生可比較的鄰居結果。
- 每個 template 含 5 個檔案，且 `reproduce.json` 足以重跑出相同結果（同 commit + 同 config + 同 seed）。
- `model.json` 可被當成 final_goal 的 submission/parent 直接匯入（形狀正確）。
- Colab notebook 從乾淨環境（只裝 numpy）跑完整流程並下載 templates。
- repo 根目錄整潔：核心碼在 `pipeline/`、交付在 `templates/`、雜物在 `archive/`、run 輸出被 gitignore。
```
