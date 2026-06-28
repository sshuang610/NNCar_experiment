# NNCars Fitness 實驗 — 階段性統整報告

日期：2026-06-28
範圍：`NNCars-Fitness-Experiments`（GA Fitness Research 組的實驗 harness）
目的：找出推薦的 fitness 配方與權重範圍，服務 [final_goal.md](final_goal.md) §3.2.4。

> 註：multi-seed 穩健性實驗（4 seed）已完成，結論見 §3.5。

---

## 1. 一句話結論

**`progress_safe` 是經 4-seed 驗證的最佳配方**（平均與最佳值都最高），關鍵成分是 **`safety` 獎勵**（讓車學會不撞牆）。
但這個 GA **高變異**：同一配方換 seed，結果在 9%~66% 間擺盪——`progress_safe` 約每 4 次抽到 1 次好解（66%），其餘 ~10%。
→ 實務做法：**用最佳配方（progress_safe）+ 多次重試 seed + 取最佳**。配方決定「機率與天花板」，seed 決定「這次有沒有抽到」。

---

## 2. 這次建立的東西（已交付、可重現、已 push）

| 項目 | 說明 |
| --- | --- |
| **Sim 加速 ~25×** | `is_on_track` early-exit + 空間索引、raycast coarse-then-fine。端到端 **5.79 → 0.23 ms/frame**，行為不變（40 萬點 0 mismatch）。原本 3.7 小時的 run 現在 ~9 分鐘。這是最有複用價值的成果。 |
| **`checkpoint` 獎勵 block** | 過里程碑給一次性獎勵（後證實無效，但功能完整、可關閉、向後相容）。 |
| **完整實驗 pipeline** | preset → GA 訓練 → auto-tune（座標搜尋）→ promote template → 匯出 final_goal model JSON。全部 TDD、有測試。 |
| **Templates** | `progress_safe_wide_v1`（旗艦）、`move_speed_v1`（早期弱 baseline）。 |
| **Configs / 證據** | 所有實驗 config 都進版控，每個結論可重跑。 |

---

## 3. 實驗時間線與發現

### 3.1 死車（高 crash 懲罰 → 車不動）
第一版 presets（`crash` 80）讓車**整場 30 秒不動**（停在起點還能領定位型獎勵）。
→ **`crash` 必須壓低（5~25）**，否則「不動」比「動了撞牆」分數高。

### 3.2 窄賽道（half_width 34）撞牆 plateau
低 crash 後車會動了，但在標準窄賽道上**跑到 120–150 代仍只有 6–8.6%、0 完賽**——一動就在彎道撞牆。更多代數沒用。

### 3.3 關鍵突破：放寬賽道 + `safety` 獎勵
把賽道放寬到 half_width 55 後：

| 配方 | maxProg | 碰撞 |
| --- | --- | --- |
| **progress_safe**（含 safety） | **0.662** | **0** |
| 其餘（無 safety / 偏 speed） | ~0.05–0.09 | 多為 1（照撞） |

`progress_safe` 兩個 seed 都全程 30 秒不撞牆（62% / 70%），**給足時間會完賽**（seed 202 約 49.7 秒）。
→ **`safety` 是「學會不撞牆」的關鍵成分**；放寬賽道給小 NN/GA 學習空間。

### 3.4 一連串「沒用」的方向（皆有實驗證據）
| 嘗試 | 結果 |
| --- | --- |
| `checkpoint` 獎勵（4 個配方） | 都輸 progress_safe，部分還 stall。**無效**。 |
| 往 speed 調（降 safety、升 speed） | 全部更差（更會撞）。**速度↔安全是硬取捨**。 |
| 加大網路 6→12→4 / 6→16→4 / 6→12→8→4 | 全部更差（0.05–0.10）。**GA 搜不動大空間**；且 6→6→4 是合約固定值。 |
| 加大 population(60) / 降 mutation(30) / 150 代 | progress_safe 掉到 0.086。**降 mutation 過早收斂**。 |
| 16 個 safety-family 變體 | 只有 progress_safe 0.662，其餘全 0.03–0.09。 |

### 3.5 決定性發現：配方有差，但 GA 高變異需多 seed
multi-seed 測試（6 配方 × 4 個 GA seed，依跨 seed 平均排名）：

| 配方 | mean | best-of-4 | worst | 每 seed maxProg |
| --- | --- | --- | --- | --- |
| **progress_safe** | **0.241** | **0.662** | 0.094 | 0.66 / 0.11 / 0.09 / 0.10 |
| ps_hi_safe | 0.118 | 0.251 | 0.047 | 0.07 / 0.25 / 0.05 / 0.10 |
| ps_lo_safe | 0.079 | 0.100 | 0.047 | 0.09 / 0.10 / 0.05 / 0.08 |
| ps_align | 0.075 | 0.168 | 0.037 | 0.05 / 0.04 / 0.17 / 0.05 |
| ps_full | 0.069 | 0.092 | 0.037 | 0.07 / 0.09 / 0.07 / 0.04 |
| ps_even | 0.062 | 0.084 | 0.047 | 0.07 / 0.05 / 0.05 / 0.08 |

兩個結論：
1. **配方確實有差**：`progress_safe` 的**平均(0.241)與最佳值(0.662)都是第一**——不是純運氣，這個含 safety 的配方真的比較好。
2. **但 GA 高變異**：`progress_safe` 4 個 seed 是 0.66 / 0.11 / 0.09 / 0.10——約 1/4 機率抽到好解(66%)，其餘 ~10%。
→ 要可靠拿到好模型：**最佳配方 + 跑多個 seed + 取 best-of-N**。單一 seed 不可靠。

---

## 4. 為什麼會這樣（瓶頸分析）

三個合約固定 / 難改的因素疊加：

1. **網路太小且固定**：6→6→4（70 參數）由 final_goal §1.1 鎖定，表達力有限。
2. **GA 弱且高變異**：隨機初始 + 取 top-2 突變，沒有 elitism / tournament，單次結果由 seed 主導。
3. **賽道難**：窄賽道的 90° 彎對這組 NN/GA 太難。

→ fitness 配方只能決定「有沒有機會」（safety 讓不撞牆變可能），**seed 決定「這次有沒有抽到好解」**。

---

## 5. 對 final_goal 的具體建議

### 5.1 推薦 fitness 配方（給 Game Engine UI 預設）
- **務必含 `safety`**（10~30）——這是不撞牆的關鍵，缺了就 ~8%。
- 主力 `progress`（40~60）+ `speed`（25~40）。
- **`crash` 壓低（10~25）**——太高會死車。
- `stall` 中等（30~50）防不動。
- `centered`/`alignment` 初期小給（0~15）；`checkpoint`/`wrong_way`/`time` 可不加。
- 起手式：`{progress:50, speed:30, safety:20 / stall:40, crash:25}`（= progress_safe）。

### 5.2 更重要的流程建議
- **單一 seed 訓練不可靠 → 要多次重試 seed 取最佳**（對應 §2.2 的「換 seed / reset」按鈕）。
- 評估配方好壞時，應跑多 seed 取 **best-of-N**，不要只看一次。
- 競賽設計可考慮：賽道別太窄（窄賽道對小 NN 幾乎不可能完賽）、或放寬 30 秒限制。

### 5.3 §5 待決參數的觀察
- **mutation 偏高（~90）較好**——降低反而過早收斂。
- 30 秒 / 900 frame 對窄賽道太緊（會導航也常完不了賽）。
- stall 門檻 `velocity < 0.5` 運作正常。

---

## 6. 目前最佳組合（如要挑一個）

`templates/progress_safe_wide_v1/`：
```json
{ "rewards": {"progress": 50, "speed": 30, "safety": 20},
  "penalties": {"stall": 40, "crash": 25} }
```
- 放寬賽道（55）、6→6→4、pop 24、mutation 90、~60 代。
- 這是 `progress_safe` 的 **best-of-4 seed 實例**（66% 進度、0 碰撞、給足時間會完賽）。
- 經 4-seed 驗證 `progress_safe` 是**最佳配方**（平均 0.241、最佳 0.662），但**單一 seed 平均只有 ~24%**——要拿到這個 66% 實例需跑多個 seed 取最佳。

---

## 7. 還能做什麼（依槓桿大小）

1. **強化 GA**（需改程式，但會偏離競賽合約——須先確認是否允許）：elitism / tournament / 多 seed 平均，直接壓低 §3.5 的高變異。
2. **multi-seed best-of-N 量化**（進行中）：每個配方跑 N seed，用 best-of-N 排名，給出有統計意義的推薦。
3. **curriculum**（需 warm-start 功能）：放寬賽道訓練 → 逐步收窄遷移到競賽寬度。
4. **收尾交付**：把 §5 寫成 `recommended_presets.md` + config 給 Game Engine 組。

---

## 附錄：產出檔案
- `pipeline/`：加速後的 sim、BeginnerMix（含 checkpoint）、tune、export。
- `configs/presets/`、`configs/arch/`、`configs/multiseed/`：所有實驗 config。
- `templates/`：可重現的 winner（含 final_goal 格式 model.json）。
- `docs/experiment_findings.md`：逐項實驗細節。
- `docs/beginner_mix.md`：block 模型規格與調參指南。
- `notebooks/run_experiments.ipynb`：Colab 入口。
