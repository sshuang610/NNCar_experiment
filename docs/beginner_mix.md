# BeginnerMix 使用說明

這份文件對應 `pipeline/fitness.py` 的 `BeginnerMix` block 模型。核心原則很簡單：

- 獎勵 block 用比例分配，會自動歸一。
- 懲罰 block 各自獨立，按強度扣分。
- `configure()` 同時支援舊的 flat rewards dict，以及新的 `{rewards, penalties}` 配方。

## 10 個 block

### 獎勵 block

- `speed`：速度越高越好，來源是 `velocity / 10`。
- `progress`：每幀前進量越大越好，來源是 `progress_delta / 10`，上限為 1。
- `centered`：越靠近中心線越好，來源是 `1 - normalized_center_offset`。
- `alignment`：車頭越對準前方越好，來源是 `heading_alignment`。
- `safety`：離牆越遠越好，來源是 `min_clearance`。

### 懲罰 block

- `stall`：停滯時扣分。
- `spin`：原地旋轉時扣分。
- `wrong_way`：車頭朝反方向時扣分，實作上由 `heading_alignment < 0` 推導。
- `time`：每幀都扣分，用來鼓勵更快完賽。
- `crash`：撞牆時一次性大扣分。

## 計分公式

```text
dt = time_elapsed / frame

reward = (Σ w_reward · factor) / Σ w_reward × B × dt
penalty = Σ (w_penalty / 100) × B × factor × dt

step = reward - penalty
if collided: step -= (crash / 100) × B_CRASH
if finished: step += FINISH_BONUS
```

其中 `B = 10`、`B_CRASH = 150`、`FINISH_BONUS = 3000`。

這個設計的重點是：多加幾個獎勵 block 不會稀釋懲罰，因為獎勵會自動歸一；懲罰則保持獨立，使用者可以直接用 0~100 的滑桿控制強度。

## 配方格式

新版配方長這樣：

```json
{
  "rewards": {
    "progress": 40,
    "speed": 30,
    "safety": 30
  },
  "penalties": {
    "stall": 60,
    "crash": 80
  }
}
```

如果只給平的 dict，例如 `{"speed": 30, "progress": 40}`，它會被當成 rewards-only 的舊格式。

## 調參指南

| 症狀 | 建議調整 |
| --- | --- |
| 停住不動 | ↑ `speed`、↑ `stall` |
| 原地打轉 | ↑ `spin`、↑ `progress` |
| 逆向亂跑 | ↑ `wrong_way`、↑ `alignment` |
| 太拖 | ↑ `time`、↑ `progress` |
| 橫衝直撞 | ↑ `crash`、↑ `safety` |
| 太怕被罰 | ↓ 各個 `penalties` |

原則上一次只調一個 block，跑一次再觀察行為。

## 從模板重現

當你有一個 `templates/<name>/`，裡面通常會有：

- `recipe.json`：這個模板的 `rewards` / `penalties` 配方。
- `reproduce.json`：重跑所需的設定快照，包含 run 資訊與 git commit。
- `result.json`：該模板的驗證結果摘要。
- `best_model.npz`：對應的模型檔。
- `model.json`：final_goal 格式的匯出模型。

重現流程通常是：讀 `reproduce.json`，把同樣的配置與 seed 再跑一次，必要時再用 `pipeline.replay` 檢查軌跡是否一致。