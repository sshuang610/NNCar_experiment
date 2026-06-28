# BeginnerMix 實驗發現紀錄

日期：2026-06-27
環境：本地 4 核（`parallel_workers=4`，BLAS 鎖單執行緒），fast 預算（pop 12 × 15 generations × 2 validation seeds）。

這份文件記錄第一輪本地實驗的可重現發現，供 GA Fitness 組調 preset 與權重範圍時參考。

---

## 1. 高 `crash` 懲罰會讓車「凍住不動」（死車 mode）

**第一版 presets**（`crash` 設 80~90，且含 `centered`/`alignment`/`safety` 等定位型獎勵）跑出來的 BeginnerMix 車輛**整場 30 秒完全不動**：

| 策略 | finish | max_progress | stall | collision |
| --- | --- | --- | --- | --- |
| progress_first | 0 | **0.000** | **30.0s** | 0.0 |
| speed_first / stable_generalist / anti_wrong_way | 0 | 0.000 | 30.0s | 0.0 |
| safe_centerline | 0 | 0.035 | 15.0s | 0.5 |
| speed_only_baseline（無 crash 懲罰） | 0 | 0.058 | 0.1s | 1.0 |
| progress_only（無 crash 懲罰） | 0 | 0.020 | 0.1s | 1.0 |

**原因**：BeginnerMix 的鼓勵側自動歸一，每秒預算只有 `B = 10`，而且實際 factor 多是分數（很少接近 1），所以「實際每秒賺到的獎勵」遠小於 10。此時 `crash` 一次性扣 `(80/100)×150 = 120`，相當於十幾秒的滿格獎勵。對隨機初始網路而言，一動就撞牆（−120），不動反而：

- 不會撞牆 → 不吃 −120。
- 停在起點時 `centered`/`alignment`/`safety` 的 factor 都接近 1 → 定位型獎勵照領。

於是 GA 很快收斂到「乾脆不動」。這正是設計文件 §7.2 預警的死車 mode。

**對 preset 設計的啟示**：
- `crash` 不能相對於每秒獎勵預算太大；本實驗 80→15~25 才解凍。
- 定位型獎勵（`centered`/`alignment`/`safety`）會付錢給靜止車，初期應少給，主力放在 `progress`/`speed`（靜止時 factor = 0，逼車移動）。
- `stall` 懲罰本身擋不住死車，因為定位型獎勵把 stall 懲罰抵掉了。

## 2. 低 crash + progress/speed 主導 → 車會動，但還不會開

**第二版 presets**（`crash` 5~25，獎勵主力 `progress`/`speed`）成功讓車移動（stall 從 30s 降到 ~0.1s）：

| 策略 | rewards | penalties | max_progress |
| --- | --- | --- | --- |
| move_speed | speed 55, progress 45 | stall 35, crash 15 | **0.065** |
| progress_safe | progress 50, speed 30, safety 20 | stall 40, crash 25 | 0.065 |
| move_progress | progress 60, speed 40 | stall 40, crash 15 | 0.052 |
| aligned_mover | progress 45, speed 35, alignment 20 | stall 40, spin 30, crash 20 | 0.047 |
| low_crash_explore | progress 50, speed 50 | stall 50, crash 5 | 0.036 |

但**沒有任何策略完賽**，最佳只到賽道 ~6.5%。車會動但不會導航（早早撞牆或繞圈）。這與原 repo 歷史一致：數十個策略只有一個（`race_metric_proxy`）曾完賽，且需 30+ 代。**15 代 / pop 12 的預算太小，學不到真正的過彎導航。**

## 3. 會「存活」的車模擬成本暴增（效能特性）

同一輪 7 個策略，訓練完成時間差異巨大：

- `move_progress` / `move_speed`（車很快撞牆 → episode 短）：**~3 分鐘**完成 15 代。
- `progress_safe` / `aligned_mover`（車存活較久 → episode 接近滿 30s）：**~3.5 小時**完成 15 代。

整輪「fast」run 實際跑了 **3h41m**（不是預估的 33 分）。原因是 `simulator.py` 每一幀對 5 個感測器做最多 1000 步的 raycast，**車開得越遠、感測器伸得越長，每幀越貴**。諷刺的是：表現越好（越會存活）的策略，模擬越慢。

**啟示**：
- 在 4 核本機上，要靠 GA 跑出「會完賽」的模型，預算實際是數小時～整夜，而且 auto-tune（一次跑十幾個 neighbor，多半是貴的存活型）會更久。
- 想加速可考慮：降低 `time_limit_seconds`（犧牲完賽機會）、或優化 `simulator.py` 的 sensor raycast（程式層面，本輪未做）、或改用更多核心 / Colab。

## 4. 目前產出的 template

`templates/move_speed_v1/`：目前最佳 mover（`rewards {speed:55, progress:45}`, `penalties {stall:35, crash:15}`）。

**這是一個弱 baseline template**（0 完賽、~6.5% 進度），`result.json` 已如實記錄。它的價值是：
- 證明整條 pipeline（preset → 訓練 → promote → final_goal model JSON 匯出）可端到端運作且可重現。
- 提供一個「會動」的起點配方，之後在更大算力上再加代數 / auto-tune。

## 5. 第二輪（2026-06-28）：simulator 加速 + checkpoint + 賽道難度

### 5.1 simulator 加速 ~25×
把 `Track.is_on_track` 改成 early-exit + 空間索引、sensor raycast 改成 coarse-then-fine，端到端從 **5.79 ms/frame → 0.23 ms/frame（~25×）**，行為不變（is_on_track 在 40 萬點 0 mismatch）。原本 3.7 小時的 run 現在約 9 分鐘。這讓高代數實驗變可行。

### 5.2 更多代數沒有打破窄賽道的 ~8% plateau
在標準窄賽道（half_width 34）上，非-checkpoint 的 motion 配方跑到 **120–150 代仍只有 ~6–8.6%、0 完賽**。→ 窄賽道的瓶頸不是代數，是「一動就在彎道撞牆」。

### 5.3 關鍵突破：放寬賽道 + `safety` 獎勵 = 真正會導航
把 `track_half_width` 從 34 放寬到 55 後：
- **`progress_safe`（唯一含 `safety` 獎勵的配方）→ 賽道 66%、0 碰撞、兩個 seed 都跑完整 30 秒不撞牆。** 其餘配方仍 ~8%（照撞）。
- → **`safety` 獎勵是「學會不撞牆」的關鍵成分**；放寬賽道讓這個小 NN/GA 有空間學會導航。

### 5.4 `checkpoint` block 沒有幫助
4 個 checkpoint 配方在放寬賽道上都輸給 `progress_safe`，部分甚至 stall（一次性大獎勵反而誘發退化行為）。→ checkpoint 這條線可放棄。

### 5.5 加速度 vs 安全是硬取捨；瓶頸轉為「能力」而非 fitness
把 `progress_safe` 往 speed 調（降 safety、升 speed）→ **全部變差**（更會撞、跑更短）。用更長 time_limit 測 `progress_safe` 存下的模型：
- seed 202：給 60s **會完賽（49.7s、100%、0 碰撞）** → 是「會開但太慢」的完整 driver。
- seed 203：~90% 處撞牆 → 接近完成但泛化還不夠穩。

→ 要在 30 秒內完賽，需要「又快又不撞」，這超出目前 6→6→4 NN + 簡單 GA 在此代數內的能力。**瓶頸已從 fitness 配方轉為模型能力（NN 大小 / 感測器 / GA）。**

## 6. 目前產出的 template

- `templates/progress_safe_wide_v1/`（**旗艦**）：`rewards {progress:50, speed:30, safety:20}`, `penalties {stall:40, crash:25}`，放寬賽道（half_width 55）。validation 66% / 0 碰撞；給足時間會完賽（seed 202 約 49.7s）。目前最強模型。
- `templates/move_speed_v1/`：早期弱 baseline（窄賽道 ~6.5%）。

## 7. 建議的下一步（依優先序）

1. **要 30 秒內完賽的競賽模型**：這已非 fitness 問題，建議動架構——加大隱藏層（如 6→12→4 或 6→8→8→4）、或強化 GA（保留更多 parents、tournament selection、調 mutation sigma）。
2. **curriculum**：先在放寬賽道（55）訓練出會開的 driver，再逐步收窄（55→48→40→34），把「會導航」遷移到競賽寬度。
3. **fitness 推薦（已收斂）**：主力 `progress` + `speed`，**務必含 `safety`** 才不會撞牆；`crash` 5~25；避免高 `checkpoint`；speed 不要壓過 safety。`progress_safe` 配方為推薦起點。
4. 窄賽道若仍要硬解：結合 1+2（大 NN + curriculum）。
