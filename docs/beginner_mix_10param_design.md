# BeginnerMix 擴充設計：5 → 10 參數（加入 5 種懲罰）

狀態：**定案**（block 化 UI + 統一滑桿 + 鼓勵自動歸一，見 §9）
範圍：`pipeline/fitness.py` 的 `BeginnerMix` 策略，以及其 config `params` 介面

> 註：§3~§8 為背景與推導；**§9 為最終定案的 block 化模型**，凌駕前面「固定 10 格參數」的描述。

---

## 1. 目標

把目前 `BeginnerMix` 的 5 個「獎勵比例」擴充成 **10 個可調參數**，新增 5 種懲罰，讓初學者能直接調整：

1. 停滯 stall
2. 原地旋轉 spin
3. 逆向 wrong-way
4. 時間 time
5. 撞牆 crash

目標不變：**初學者只改數字、不碰程式**，而且每個參數都對應一個看得出來的駕駛行為。

---

## 2. 現況回顧

目前 `BeginnerMix`（[pipeline/fitness.py](../pipeline/fitness.py)）：

```text
reward_per_step = ( Σ wᵢ · factorᵢ ) / Σ wᵢ  × 10        # 0~10
score = reward_per_step
if is_stalled:  score -= 5      # 寫死
if is_spinning: score -= 5      # 寫死
score += 10000 (完成) / -500 (撞牆)                      # 寫死
```

5 個 reward factor（皆正規化到 0~1）：`speed / progress / centered / alignment / safety`，預設比例 `30 / 40 / 10 / 10 / 10`。

**問題**：stall、spin、撞牆都是寫死的，逆向與時間根本沒有。初學者無法調「壞行為要罰多重」。

---

## 3. 設計評估：懲罰也要做成「比例」嗎？

直覺上會想「那就把 10 個都丟進同一個 100% 比例池」。**不建議**，原因：

- 「比例」的語意是**把 100% 的注意力分配給好行為**（搶資源、彼此取捨）。獎勵彼此競爭很合理。
- 懲罰是**絕對的扣分**（做壞事就扣多少），不該跟獎勵搶同一個 100%。把「撞牆」放進比例池會變成「我給撞牆 20% 注意力」——語意不通。

更關鍵的是 **兩類懲罰的時間尺度完全不同**：

| 類型 | 例子 | 行為 |
| --- | --- | --- |
| 每幀懲罰（會累積） | stall, spin, wrong-way, time | 每一個畫面只要條件成立就扣一次，整場累積上百次 |
| 一次性事件 | crash | 整場只發生一次（且撞牆後回合結束） |

所以**不能**把它們塞進同一個正規化池。

### 推薦模型：兩組分開

> **獎勵 = 比例分配（搶 100%）；懲罰 = 各自獨立設強度（絕對扣分）。**

- **獎勵組（5 個）**：維持現狀的比例加權平均 → 每幀 0~10 分。
- **懲罰組（5 個）**：每個有自己的強度，直接從分數扣掉。其中 4 個是「每幀扣分」（和獎勵同一個每幀尺度，好理解），crash 是「一次性大扣分」。

這樣 10 個數字對初學者來說：獎勵是「占多少比例」，懲罰是「做這件壞事每次扣幾分」，兩種心智模型都單純。

---

## 4. 推薦設計：10 參數規格

### 4.1 參數總表

**獎勵組（比例，互相競爭，總和自動正規化）**

| 參數 | 白話 | StepContext 來源 | 正規化後 factor | 範圍 | 預設 |
| --- | --- | --- | --- | --- | --- |
| `speed` | 開快 | `velocity` | `velocity/10` | 0~100 | 30 |
| `progress` | 往終點前進 | `progress_delta` | `min(progress_delta/10, 1)` | 0~100 | 40 |
| `centered` | 走在路中間 | `normalized_center_offset` | `max(0, 1-offset)` | 0~100 | 10 |
| `alignment` | 車頭對準前方 | `heading_alignment` | `max(0, alignment)` | 0~100 | 10 |
| `safety` | 離牆遠一點 | `min_clearance` | `min(min_clearance,90)/90` | 0~100 | 10 |

**懲罰組（絕對扣分，彼此獨立）**

| 參數 | 白話 | StepContext 來源 | 套用方式 | 範圍 | 預設 |
| --- | --- | --- | --- | --- | --- |
| `pen_stall` | 幾乎停住不動 | `is_stalled` | 每幀扣 `pen_stall` | 0~10 | 5 |
| `pen_spin` | 原地打轉 | `is_spinning` | 每幀扣 `pen_spin` | 0~10 | 5 |
| `pen_wrong_way` | 逆向（車頭朝後）| `heading_alignment < 0` | 每幀扣 `pen_wrong_way` | 0~10 | 0 |
| `pen_time` | 拖時間 | （每一幀都成立）| 每幀扣 `pen_time` | 0~5 | 0 |
| `pen_crash` | 撞牆 | `collided` | **一次性**扣 `pen_crash` | 0~2000 | 500 |

> `is_wrong_way` 在 `StepContext` 裡沒有現成欄位，由 `heading_alignment < 0`（與賽道方向夾角超過 90°）推導。

### 4.2 計分公式

```text
# 1) 獎勵：比例加權平均（同現狀）
reward = ( Σ w_reward · factor ) / Σ w_reward  × 10           # 0~10

# 2) 每幀懲罰：和獎勵同一個每幀尺度
penalty  = pen_stall      if is_stalled    else 0
penalty += pen_spin       if is_spinning   else 0
penalty += pen_wrong_way  if heading_alignment < 0 else 0
penalty += pen_time                                            # 每幀都扣

step = reward - penalty

# 3) 一次性事件
if collided: step -= pen_crash
if finished: step += FINISH_BONUS        # 固定 10000，不在 10 個可調參數內

return step
```

### 4.3 為什麼這樣切

- **每幀懲罰和獎勵同尺度**：每幀最多賺 ~10，stall 扣 5 就是「這幀好處幾乎被抵銷」，初學者立刻有感。
- **crash 單獨一次性大數**：撞牆會直接結束回合，本來就少賺很多後續獎勵，再加一次性大罰 → 強力嚇阻，且不會因為「每幀」而被放大到失控。
- **time 用每幀固定扣分**：每一幀都扣一點 → 整場累積 ∝ 花的總時間，自然形成「早點完成比較好」。

---

## 5. 預設值與向後相容

**關鍵性質：不給任何 `params` 時，新版行為與舊版逐字相同。**

預設 `pen_stall=5, pen_spin=5, pen_wrong_way=0, pen_time=0, pen_crash=500`，代入公式即等於現在寫死的 `-5 / -5 / -500 / +10000`。逆向與時間預設為 0 = 關閉，**初學者主動設大於 0 才會啟用**。好處：

- 既有的 5-key config（只設獎勵比例）行為完全不變。
- 新功能採「opt-in」，不會悄悄改掉別人正在跑的實驗。

另提供一組「推薦起手式」放進範例 config（溫和開啟新懲罰）：

```json
{ "speed": 30, "progress": 40, "centered": 10, "alignment": 10, "safety": 10,
  "pen_stall": 5, "pen_spin": 5, "pen_wrong_way": 3, "pen_time": 0.3, "pen_crash": 500 }
```

---

## 6. 調參指南（症狀 → 調哪個）

| 症狀 | 調整 |
| --- | --- |
| 🐢 龜速 / 停著不動 | ↑ `speed`、↑ `pen_stall` |
| 🌀 原地打轉 | ↑ `pen_spin`、↑ `progress` |
| ↩️ 逆向亂跑 | ↑ `pen_wrong_way`、↑ `alignment` |
| 🐌 會跑但太拖 | ↑ `pen_time`（小幅，0.1 起跳）、↑ `progress` |
| 💥 橫衝直撞 | ↑ `pen_crash`、↑ `safety` |
| 🥶 什麼都不敢做（過度被罰） | ↓ 各 `pen_*`，先把懲罰調小再慢慢加 |

**紀律**：一次只動一個參數、跑一次看行為再決定下一步。

---

## 7. 風險與注意事項

1. **每幀懲罰會累積**：`pen_stall=5` 在 30fps、卡住整場 30 秒 = 900 幀 ≈ −4500。這是刻意的強嚇阻，但**文件/UI 必須講清楚「每幀 × 上百幀」**，否則初學者會低估小數字的威力。
2. **懲罰壓過獎勵的死車**：把 `pen_*` 開太大，最佳策略會變成「乾脆不動以免被罰」（但 `pen_stall` 又會罰不動……）導致學不起來。預設保守、範圍上限有限制、症狀表引導，皆為緩解。
3. **總分可能為負**：GA 只看相對排名，負分無妨；但 summary 報表要避免讓初學者誤會。
4. **`alignment` 獎勵與 `pen_wrong_way` 部分重疊**：逆向時 `alignment` factor 已是 0；`pen_wrong_way` 是在「0 獎勵」之上再主動扣分，訊號更強。屬刻意設計，非 bug。
5. **`pen_time` 與 fps 相關**：每幀固定扣分，換 fps 會改變總量。本專案 fps 固定（30），可接受；若日後要 fps 無關，可改成乘 `dt`（見開放決策）。
6. **懲罰設負值的footgun**：負的懲罰會變成獎勵。建議 `configure()` 對 `pen_*` 做 `max(0, value)` 夾住。

---

## 8. 實作計畫

1. **`BeginnerMix` 重構**（[pipeline/fitness.py](../pipeline/fitness.py)）
   - 拆成 `REWARD_DEFAULTS`（5）與 `PENALTY_DEFAULTS`（5），合併成 `self.weights`（10 key）。
   - `configure()` 接受任何子集；`pen_*` 夾到 `>= 0`；未知 key 忽略（維持現狀）。
   - `score_step()` 改用 §4.2 公式；`is_wrong_way` 由 `heading_alignment < 0` 推導；`FINISH_BONUS = 10000` 留常數。
2. **Config**：更新 `configs/experiment_beginner_mix.json` 的 `params` 為 10 key（用 §5 推薦起手式）。`params` 串接管線已完成，無需改 `training.py`。
3. **Metadata / replay**：`strategy_params` 已存入 model metadata 並於 replay 套用，新增 key 自動涵蓋，無需改動。
4. **測試**（TDD，延伸 `tests/test_beginner_mix.py`）
   - 向後相容：不給 params == 舊行為（stall −5、spin −5、crash −500、finish +10000）。
   - 每個新懲罰：逆向偵測（alignment<0 才扣）、time 每幀都扣、crash 一次性且可調、pen 夾 `>=0`。
   - 累積性：連續 N 幀 stall = N × pen_stall。
   - 獎勵組行為不變（既有測試續綠）。

---

## 9. 定案：Block 化 UI 與統一滑桿

最終採用 **block 化、可自由組合** 的介面；本節凌駕前面把參數視為固定 10 格的描述。

### 9.1 介面模型
- 一個 **block 調色盤**：每個參數 = 一個 block。使用者把想要的 block 加進配方，沒加的就不計分。
- 每個 block 長一樣：圖示、白話名、一行說明、**方向（➕鼓勵 / ➖避免）**、一條 **0~100 正規化滑桿**。
- 使用者眼中只有「一種 block、一種滑桿」，不需要知道背後鼓勵/懲罰的數學差異。

### 9.2 背後計分（統一滑桿、鼓勵自動歸一）
令 `dt = 1/fps`（由 `time_elapsed / frame` 推導），鼓勵與懲罰共用同一個每秒預算 `B`。實作常數：`B = 10`、`B_CRASH = B × CRASH_SECONDS`（`CRASH_SECONDS = 15`，即撞牆滿格 = 15 秒完美獎勵）、`FINISH_BONUS = B × FINISH_SECONDS`（`FINISH_SECONDS = 300`，完成壓倒性大獎）。因為每一項都含 `B`，`B` 的絕對值不影響 GA 排名，只有各滑桿、`CRASH_SECONDS`、`FINISH_SECONDS` 的比值有意義。

```text
# 鼓勵側：自動歸一 → 固定預算，懲罰永不被稀釋
reward = ( Σ_鼓勵 wᵢ·factorᵢ / Σ_鼓勵 wᵢ ) × B × dt

# 避免側（每幀型）：各自獨立，滑桿 = 該行為最大強度的百分比
penalty = Σ_避免每幀 (wⱼ / 100) × B × factorⱼ × dt

step = reward − penalty
if 撞牆:  step -= (w_crash / 100) × B_CRASH     # 一次性，B_CRASH 為較大常數
if 完成:  step += FINISH_BONUS                  # 固定，不在 block 內
```

- **校準性質**：把一個避免滑桿拉到 100，那個壞行為發生時就完全抵銷該幀全部鼓勵（reward 與單一 maxed penalty 同尺度）。直覺、好解釋。
- **鼓勵側自動歸一 = 前面說的「比例」，但使用者看不到「比例」字眼**，只會發現「把所有鼓勵拉滿不會讓車什麼都在乎，因為它們在搶同一份注意力」——這正是駕駛的真相。
- **為什麼鼓勵要歸一**：見 §7 與對話結論——否則多加幾個鼓勵 block 會悄悄稀釋懲罰，使用者拉高懲罰卻沒效果。歸一把獎勵鎖在固定預算，懲罰永遠咬得住。
- 每幀型一律 `× dt`（每秒制）→ 換 fps 行為不變、所有滑桿可比。

### 9.3 資料格式
配方序列化為兩組，只放有加的 block：

```json
{
  "rewards":   { "progress": 40, "speed": 30, "safety": 30 },
  "penalties": { "stall": 60, "crash": 80 }
}
```

- `rewards` 自動歸一成比例；`penalties` 各自 = 強度%。
- 與既有 `params` 機制相容：`configure()` 同時接受舊的「平的 rewards-only dict」（視為 rewards）與新的 `{rewards, penalties}`。

### 9.4 起手式（避免空白畫布）
內建數組預設配方（就是上面的 JSON），初學者選一個會跑的 → 加一個 block / 拉一條滑桿 → 看差別。學習靠改範例。

### 9.5 其餘決策（定案）
1. **新懲罰**：opt-in——預設配方不含 `wrong_way`/`time`，使用者自行加入 block。
2. **命名**：block 以方向區分（➕/➖）；資料層歸入 `rewards` / `penalties` 兩群組，不混淆。
3. **`time` 尺度**：採每秒 `× dt`（統一滑桿、fps 無關的前提）。← 取代前面「每幀固定」的暫定。
4. **`FINISH_BONUS`**：不可調，維持固定大獎，不佔 block。

---

## 10. 驗收標準

- block 可自由增刪；只計入有加的 block。
- 所有滑桿 0~100；鼓勵側自動歸一、懲罰側獨立；換 fps 行為一致（每秒 `× dt`）。
- 把任一避免滑桿拉到 100，對應壞行為發生時可抵銷該幀全部鼓勵（校準測試）。
- 多加鼓勵 block 不會稀釋既有懲罰效果（迴歸測試）。
- `configure()` 同時吃舊 flat dict 與新 `{rewards, penalties}`。
- 撞牆為一次性扣分、完成為固定大獎。
- `tests/test_beginner_mix.py` 全綠；`configs/experiment_beginner_mix.json` 能跑出 summary 與可 replay 的 best_model。
