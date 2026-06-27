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

## 5. 建議的下一步

1. 想要會完賽的 template：用更大算力（Colab/多核）跑 `generations ≥ 30`、`population ≥ 20`，並對 §2 的 motion-first 配方做 auto-tune。
2. 若要在本機續跑：先評估是否優化 sensor raycast，否則每輪數小時。
3. preset 推薦範圍（給 Game Engine UI 預設）：`crash` 建議 5~25（避免死車）、主力獎勵放 `progress`/`speed`、定位型獎勵初期小給。
