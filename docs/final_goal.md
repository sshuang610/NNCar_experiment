## **0. 共同目標**

本次 v2 的核心目標是讓參賽者訓練出的神經網路權重可以跨地圖持續演化，而不是每次換地圖都重新開始。參賽者會在本地 client 訓練、挑選模型並跑出 submission result；server 負責接收結果、播放 replay 與依上傳 metrics 更新排行榜。

整體訓練與競賽流程如下：

1. **本地訓練 parents**：使用者在 Game Engine client 輸入 `group_id`、`username`，選擇 `easy`、`hard` 或 `random` train map，從零開始訓練或 import 兩個 parents。訓練過程中，GA Fitness 組提供可組合的 fitness functions，client 依 fitness score 持續追蹤目前表現最好的兩台車作為 `parent_a` 與 `parent_b`。
2. **跨地圖持續演化**：使用者可以切換 train map 難度、random map 或車子 random seed。切換地圖時保留 parent weights、biases 與訓練 lineage，但重置車輛位置、碰撞狀態、圈數、計時、run metrics 與畫面暫存狀態。validation map 只用於 inference 與泛化檢查，不自動 breed、不 mutation、不更新 parents。
3. **本地產生 submission candidate**：準備提交時，client 使用目前 fitness score 最好的 `parent_a` 與 `parent_b` 建立 20 台 candidates。Candidate 0 保留 `parent_a`，Candidate 1 保留 `parent_b`，Candidate 2-19 由 crossover 與 mutation 產生。
4. **本地選出 local winner**：20 台 candidates 必須在同一張 competition map、同一 spawn、同一 simulation limits 與同一 scoring rules 下執行。client 依正式 run metrics / ranking tuple 選出 metric 表現最好的一台 local winner；有完賽者優先比完賽時間，未完賽者比 `max_progress`，再比到達 `max_progress` 的時間。
5. **第一階段 Easy / Hard submission**：client 只提交 local winner 的 weights、biases、`group_id`、`username` 與本地已跑好的 `client_result`，不提交 parents 或其他 19 台 candidates。Easy 與 Hard 是兩個獨立 competition，各自有獨立 submission、replay 與 leaderboard。
6. **Server replay + ranking**：Competition Server 驗證 payload 與 cooldown，不執行 breed、mutation 或 20 台 candidate selection，也不重新計算分數。server 在 replay batch 中使用提交的 winner model 播放對應 competition map，並依 client 上傳的 run metrics / ranking tuple 更新 leaderboard。
7. **五分鐘 replay / leaderboard 更新**：第一階段每 5 分鐘更新 replay batch 與 leaderboard。submission 尚未完成 replay 排程或播放時顯示 `queued` 或 `running`，未趕上本輪 replay 的 submission 延後到下一輪。
8. **第二階段 Final Hard Map**：每隊從自己的訓練結果或第一階段 winners 中選出一個最強 final model。final model 只能提交一次，直接在主辦方設計的 Final Hard Map 上 inference，不再 breed 或 mutation，並使用同一套 ranking rules 產生 final leaderboard。

## **1. Shared Contract與分工邊界**

### **1.1 Model Export 格式**

三組都必須支援同一份 model JSON 格式：

```json
{
  "group_id": "1",
  "username": "player1",
  "weights": [[36], [24]],
  "biases": [[6], [4]]
}
```

- `group_id`：隊伍識別碼，字串。
- `username`：顯示暱稱。
- `weights[0]`：input -> hidden，原始 shape `(6, 6)`，flatten 後長度 36。
- `weights[1]`：hidden -> output，原始 shape `(4, 6)`，flatten 後長度 24。
- `biases[0]`：hidden layer，原始 shape `(6, 1)`，flatten 後長度 6。
- `biases[1]`：output layer，原始 shape `(4, 1)`，flatten 後長度 4。

Export 與 submission 使用相同的單一 model 欄位格式，但用途不同：

- **Export**：保存 parents，也就是目前依 fitness score 表現最好的兩台車 `parent_a` 與 `parent_b`。兩個 parents 各自都是一份完整 model，可用於之後繼續訓練、crossover 與 mutation。
- **Submission**：保存 metric 表現最好的一台車，也就是從 20 台 candidates 中依正式 run metrics / ranking tuple 選出的唯一 local winner。submission 只送出這一台 winner model，不送出 parents 或其他 candidates。

### **1.2 Submission Result 格式**

client 本地跑完 local winner 後，submission 必須帶上這組 run metrics；server 依此 result 排名與顯示 replay：

```json
{
  "group_id": "1",
  "username": "player1",
  "weights": [[36], [24]],
  "biases": [[6], [4]],
  "client_result":
	  {
	  "completed": false,
	  "lap_ticks": null,
	  "max_progress": 1250.5,
	  "ticks_to_max_progress": 840
	}
}
```

排序規則使用字典序：

```
completed:     (1, -lap_ticks, 0, -candidate_index)
not completed: (0, max_progress, -ticks_to_max_progress, -candidate_index)
```

### **1.3 組間責任切分**

- Game Engine 組負責參賽者本地使用的完整操作流程：輸入隊伍資料、訓練、選地圖、local inference、選 winner、送出 submission。
- GA Fitness Function Research 組負責可組合的 score function 與 fitness 實驗。
- Competition Server 組負責接收 submission、驗證 payload、排程 replay、依 submission result 更新 leaderboard，以及第二階段 final competition。

## **2. Group A: Game Engine**

### 2.1 責任範圍

Game Engine 組負責官方 client 與本地訓練體驗，包含 Pygame UI、訓練 pipeline、地圖選擇、本地 inference、local winner selection，以及把 winner model 提交到 server。

### 2.2 TODO：使用者流程

#### 1. 初始設定

使用者進入 client 後，必須能設定：

- `group_id`：隊伍 ID，提交與匯出時使用字串格式。
- `username`：顯示暱稱。
- server base URL：submission API endpoint 的主機位置。

client 應保存上述設定，避免每次重新開啟都需要重填。

#### 2. 本地訓練流程

1. 使用者選擇 train map：`easy`、`hard` 或 `random`。
2. 使用者選擇從新開始訓練，或 import 兩個 parents：`parent_a` 與 `parent_b`。
3. 使用者選擇訓練策略並設定不同參數，然後開始訓練。
4. client 需顯示目前 fitness score 最高的兩台車，並同時顯示上一輪 parents。
5. 訓練過程需支援下列操作：
    - **換 seed / reset**：重新產生或重置車子 / 初始族群的 random seed；目前地圖不變。
    - **切換成 validation**：使用者可以隨時切到 validation map 執行 inference，支援 `easy` 與 `hard`。validation run 只評估泛化能力，不自動 breed、不 mutation、不更新 parents。
    - **更換地圖難度**：提供 `easy`、`hard`、`random` 三個按鈕；切換時保留 parent weights 與 biases，並重置車輛位置、碰撞狀態、圈數、計時、run metrics 與畫面中的暫存狀態。
    - **return to home page**：返回前詢問是否要 export 目前權重，之後回到初始設定流程。
    - **export**：匯出 `parent_a` 與 `parent_b` 的權重；這兩台 parents 應是目前依 fitness score 表現最好的兩台車。
    - **submit**：進入 submission 流程，送出的 payload 必須符合 server API 格式。

#### 3. Submission 流程

1. 使用者選擇要提交到 `easy` 或 `hard` competition。
2. 在開始本地 evaluation 前，client 必須先以目前的 `group_id` 與 `username` 呼叫對應的 eligibility API：
    - Easy：`POST /v2/competitions/easy/eligibility`
    - Hard：`POST /v2/competitions/hard/eligibility`
    - Final：`POST /v2/finals/eligibility`
3. client 檢查 eligibility response。只有 `eligible: true` 時，才能開始建立 candidates、執行本地 evaluation，並於完成後呼叫相對應的 submission API。response 同時包含 `next_submission_at`、目前 stage 與 `competition_config_version`，client 應保存並顯示這些資訊。
4. 若 `eligible: false`，client 不得開始本地 evaluation，並應在 UI 顯示 server 回傳的 rejection reason 與 `next_submission_at`（若有）。常見情況包含：
    - 尚在五分鐘 cooldown：顯示目前不可提交及下一次可提交時間；submission endpoint 會以 HTTP `429`、`error: submission_cooldown` 拒絕。
    - competition 尚未開放或已關閉：顯示目前 stage 與不可提交原因；submission endpoint 會回 HTTP `409`。
    - Final group 已鎖定或已完成提交：顯示 Final 僅能提交一次及 server rejection reason；submission endpoint 會回 HTTP `409`。eligibility 僅是預先提示；真正提交時，submission endpoint 仍會以原子操作再次檢查資格。因此即使先前取得 `eligible: true`，client 也必須正確處理提交當下的 `429` 或 `409`，顯示最新錯誤與 `next_submission_at`，不可假設 submission 一定成功。
5. client 使用目前表現最好的 `parent_a` 與 `parent_b` 建立 20 台 candidates。
    - 這裡的表現最好指依照 GA Fitness 組提供的 fitness function，選出 fitness score 最高的兩台車。
    - submission 前的 local winner run 使用 30 秒限制；有完賽者先比完賽時間，未完賽或需要進一步比較時比 `max_progress`，再比到達 `max_progress` 的時間。
6. 20 台 candidates 在同一張 competition map、同一 spawn、同一 simulation limits 與同一 scoring rules 下執行。
7. client 依 ranking tuple 選出唯一 local winner。
8. client 顯示 local winner 的 client result。
9. client 呼叫對應的 submission API，只提交 local winner 的 weights、biases、`group_id`、`username` 與 `client_result`：
    - Easy / Hard：`POST /v2/competitions/{easy|hard}/submissions`
    - Final：`POST /v2/finals/submissions`所有 genes 與 progress 數值都必須是 finite number。未完賽時 `lap_ticks` 必須為 `null`；完賽時 `lap_ticks` 必須是正整數；所有 tick 值不得超過 `900`。
10. client 顯示 server 回傳的 submission status 與下一次可提交時間。Easy 與 Hard 各自有獨立的五分鐘 cooldown；成功的 Phase 1 submission 會先顯示 `queued`，等待下一次 snapshot，Final submission 則會立即成為 `completed`。

### 2.3 需與其他組對接

- 與 GA Fitness 組對接：
    - 使用其提供的 fitness function registry 或 score API。
    - 提供 run metrics 給 fitness function，例如 speed、progress、spin、collision、stagnation。
    - 允許使用者選擇或組合不同 score terms。
    - UI 顯示 raw score、weight、weighted score、total score breakdown、上一輪 parents、目前 fitness top 2 標記時，應參考 GA Fitness 組提供的 panel data / parent selection API。
    - 使用 GA Fitness 組的 API 選出並更新目前最佳兩台 parents：`parent_a` 與 `parent_b`。
- 與 Competition Server 組對接：
    - 使用共同 model export 格式。
    - 使用共同 client result 格式。
    - 開始本地 evaluation 前，先以 `group_id` 與 `username` 呼叫對應的 POST eligibility API；只有回傳 `eligible: true` 才繼續 submission 流程。
    - eligibility 被拒絕或 submission 當下資格改變時，顯示 rejection reason、目前 stage、`next_submission_at` 與對應的 cooldown / competition closed / Final locked 錯誤。
    - 使用 server 回傳的 `submission_id`、`next_submission_at`、`competition_config_version` 與 status。

### 2.4 驗收標準

- 可以輸入 group ID 與暱稱，並成功出現在匯出檔和 submission payload。
- 可以在本地使用兩個 parents 建立 20 台 population。
- 可以顯示目前 fitness score 最高的兩台車與上一輪 parents。
- raw score、weight、weighted score、total score breakdown 與 top parent 標記需使用 GA Fitness 組提供的 panel data API。
- export 會匯出目前最佳 `parent_a` 與 `parent_b` 的權重。
- 可以跑完 competition map 並選出 local winner。
- 每次只提交一台 local winner model 與 client result。
- 開始本地 evaluation 前會呼叫對應的 POST eligibility API，且只在 `eligible: true` 時繼續。
- `eligible: false`、HTTP `429 submission_cooldown`、HTTP `409` competition closed / Final locked 都有清楚 UI 回饋，並在可用時顯示 rejection reason 與 `next_submission_at`。
- 換 random map 後能保留 parents 並繼續訓練。
- validation run 不會自動更新 parents。
- cooldown、queued、running、completed、failed 狀態有清楚 UI 回饋。

## **3. Group B: GA Fitness Function Research**

### **3.1 責任範圍**

GA Fitness Function Research 組負責研究、實作與比較多種 fitness function 組合，提供可自由組合的 score terms，讓使用者能用不同策略訓練模型。

### **3.2 必做功能**

1. Fitness term 定義
    - 至少提供 10 個可獨立啟用 / 停用的 fitness functions，讓使用者自行組合策略。
    - （以下只是我叫他隨便生的，可以自己研究）
    - `progress_score`：根據標準賽道中心線或 checkpoints 的 `max_progress` 計分。
    - `speed_score`：鼓勵更快抵達 checkpoint 或完成圈數。
    - `completion_bonus`：完成一圈時給予額外獎勵。
    - `collision_penalty`：碰撞時扣分。
    - `spin_penalty`：懲罰原地打轉或方向劇烈震盪。
    - `stagnation_penalty`：懲罰長時間沒有增加有效 progress。
    - `reverse_penalty`：懲罰逆向或倒退行為。
    - `smooth_control_score`：鼓勵穩定輸出，減少控制震盪。
    - 可再補充其他 function，例如 checkpoint reward、centerline alignment、wall distance safety、steering efficiency、time penalty 等。
2. 可組合 score API
    - 提供統一的 fitness function API，讓 Game Engine 可以取得所有 function 的名稱、說明、預設權重、可調範圍與目前啟用狀態。
    - 允許使用者在 Game Engine UI 上選擇每一個 fitness function 的權重，組出自己的訓練策略。
    - score 計算方式應是多個 fitness functions 的 weighted combination。
    - 保留每個 function 的原始分數、權重與加權後分數，方便 UI 顯示與 debug。
    - 預設提供至少三組 starter presets，確保系統不調參也可以跑過基本訓練流程；presets 只是參考起點，不限制使用者自行設計 score function：
        - progress-first：重視最大進度。
        - speed-first：重視完成後圈速。
        - stable-generalist：兼顧進度、穩定性與懲罰項。
3. 提供給 Game Engine UI 使用的 panel data API，包含目前每台車的 score breakdown、目前 fitness top 2、上一輪 parents，以及每台車是否應被標記為 top parent。
4. 實驗與比較
    - 使用固定 seeds 或固定 tracks 比較不同 fitness 組合。
    - 記錄每組 fitness 對 completion rate、max progress、lap ticks、collision rate 的影響。
    - 產出推薦的 default fitness preset 與各 fitness function 權重範圍建議。
5. 與正式 ranking 的一致性
    - local winner 最終選擇必須能使用 spec v2 的正式排序規則。
    - fitness score 可以用於訓練與 parent selection，但不應取代 leaderboard 的正式 ranking tuple。

### **3.3 建議介面**

```python
def list_fitness_functions() -> list[FitnessFunctionSpec]:
    ...

def score_run(metrics: RunMetrics, weights: dict[str, float]) -> FitnessBreakdown:
    ...

def score_population_with_breakdown(
    population: list[CarLike],
    weights: dict[str, float],
) -> list[CarFitnessBreakdown]:
    ...

def select_top_parents(
    population: list[CarLike],
    weights: dict[str, float],
    k: int = 2,
) -> ParentSelectionResult:
    ...

def build_fitness_panel_data(
    population: list[CarLike],
    weights: dict[str, float],
    previous_parent_ids: list[str],
) -> FitnessPanelData:
    ...
```

`FitnessFunctionSpec` 建議包含：

- `id`
- `display_name`
- `description`
- `default_weight`
- `min_weight`
- `max_weight`
- `enabled_by_default`

`FitnessBreakdown` 建議包含：

- `total_score`
- `raw_scores`
- `weights`
- `weighted_scores`
- 各 fitness function 的 breakdown，例如 `progress_score`、`speed_score`、`completion_bonus`、`collision_penalty`、`spin_penalty`、`stagnation_penalty`、`reverse_penalty`、`smooth_control_score`

`FitnessPanelData` 建議包含：

- `rows`：每台車的 `car_id`、rank、raw scores、weights、weighted scores、total score。
- `current_top_parent_ids`：目前 fitness score 最高的兩台車。
- `previous_parent_ids`：上一輪 parents。
- `highlight_flags`：提供給 Game Engine UI 使用的標記，例如 `is_current_top_parent`、`is_previous_parent`。

GA Fitness 組只提供資料與 selection result，不直接修改 Pygame rendering、車輛 sprite 或 Game Engine 畫面流程。

### **3.4 需與其他組對接**

- 與 Game Engine 組對接：
    - 定義 Game Engine 需要收集哪些 per-run metrics。
    - 提供 fitness function metadata，讓 UI 可以自動產生權重調整控制項。
    - 提供 starter preset 列表，但允許使用者覆寫每個 fitness function 的權重。
    - 提供 `FitnessPanelData`，支援 UI 顯示 raw score、weight、weighted score、total score breakdown、上一輪 parents 與目前 top 2 parents。
    - 提供 parent selection API，讓 Game Engine 不需要自行重寫 top 2 fitness 排名邏輯。
- 與 Competition Server 組對接：
    - 確認 `max_progress`、`lap_ticks`、`ticks_to_max_progress`、`completed` 的定義一致。
    - 不把實驗性 fitness score 寫入正式 leaderboard。

### **3.5 驗收標準**

- 至少實作 10 個可獨立組合的 fitness functions。
- 至少實作三組 starter presets，且 presets 可被使用者修改。
- 使用者可以在 Game Engine UI 自由調整每個 fitness function 的權重。
- 每次 run 可以產生 total score 與 breakdown。
- 可以產生給 Game Engine UI 使用的 `FitnessPanelData`，包含目前 top 2 parents、上一輪 parents 與每台車的 highlight flags。
- 有測試或實驗資料證明不同 fitness 組合會產生可比較的結果。
- local winner selection 仍可回到 spec v2 的正式排序 tuple。

## **4. Group C: Competition Server**

### **4.1 責任範圍**

Competition Server 組負責所有正式比賽流程：接收 submission、驗證 payload、rate limit、排程 replay、依 client 上傳的 submission result 更新 leaderboard、保存稽核資料，以及執行第二階段 final competition。

### **4.2 必做功能**

1. Submission API
    - `POST /v2/competitions/{competition_id}/submissions`
    - `GET /v2/competitions/{competition_id}/submissions/{submission_id}`
    - `POST /v2/finals/submissions`
    - 第一階段 `competition_id` 只能是 `easy` 或 `hard`。
2. Payload 驗證
    - 檢查 `group_id`、`username`、`weights`、`biases`、`client_result` 是否存在。
    - 檢查 weights 與 biases shape。
    - 拒絕 `NaN`、infinity 與非有限數值。
    - 檢查欄位長度與 payload 大小。
    - 第二階段 final submission 必須確認該隊尚未鎖定 final model。
3. Rate limit
    - 每位參賽者每 5 分鐘最多成功提交一次。
    - 以 server 收到上一筆成功 submission 的時間計算。
    - 無效 payload 不占用 cooldown。
    - cooldown 期間回傳 HTTP `429 Too Many Requests`。
4. Replay batch
    - Easy 與 Hard 各自維護獨立 queue、replay 與 leaderboard。
    - 每 5 分鐘建立 replay batch。
    - batch 第一次 playback 負責建立 replay 紀錄並更新 leaderboard。
    - 後續 replay loop 只重播紀錄，不重新排名。
    - 未趕上本輪 replay 的 submission 顯示 deferred notification。
5. Replay 與排名
    - Server 不執行 breed、mutation 或 20 台 candidate selection。
    - Server 不重新跑分，也不覆寫 client 上傳的 `client_result`。
    - Server replay client 上傳的單一 local winner model，用於視覺化與稽核。
    - leaderboard 使用 submission 內的 `client_result`，並依正式 ranking tuple 排序。
6. Leaderboard / Dashboard
    - Dashboard 每 5 分鐘建立 snapshot。
    - Easy 與 Hard 有獨立 leaderboard。
    - 每位參賽者在每個 competition 中以歷史最佳 completed submission 進榜。
    - 顯示排名、group id、username、submission id、狀態、圈速或最大進度、完成 replay 時間、下一次可提交時間。
7. Replay 與稽核資料保存
    - 保存 winner weights、biases、client result、地圖 id/version/seed、simulation version、終止原因、run metrics。
    - 保存 replay batch 的 window、included submission ids、deferred submission ids、leaderboard snapshot。
    - 保存逐 tick replay 或 deterministic input/state log。
8. 第二階段 competition
    - 每隊只能提交一個 final model。
    - Final Hard Map 不使用五分鐘 replay batch。
    - 所有 final models 使用相同 spawn、simulation limits 與評分條件。
    - 使用正式排序規則產生 Final leaderboard。

### **4.3 需與其他組對接**

- 與 Game Engine 組對接：
    - 提供 submission endpoint、status endpoint 與 final endpoint。
    - 回傳 `submission_id`、`status`、`submitted_at`、`next_submission_at`、`competition_config_version`。
    - 回傳 server 接受並用於排名的 client result、replay id 與 leaderboard rank。
- 與 GA Fitness 組對接：
    - 對齊正式 metrics 定義。
    - 確保 leaderboard 使用 submission result 與正式 ranking tuple。

### **4.4 驗收標準**

- 能拒絕錯誤 shape、缺欄位、無效數值與 cooldown 期間的 submission。
- Server 不執行 breed、mutation 或 population scoring。
- 能在 replay batch 中播放每筆 submission 的 replay。
- leaderboard 使用 client 上傳的 submission result。
- 通關者以圈速排序，未通關者以 `max_progress` 與 `ticks_to_max_progress` 排序。
- Easy 與 Hard queue、replay、leaderboard 完全獨立。
- 每隊能鎖定一個 final model 並產生 Final leaderboard。
- completed submission 可以透過 replay 對照其上傳 metrics。

## **5. 比賽前仍需共同決定**

- mutation probability 與 mutation sigma 正式數值。
- 單次 run 的 frame limit 與停滯門檻。
- Dashboard 五分鐘 snapshot 的整點對齊方式。
- 同分時是否使用較早提交時間作為最終 tie-breaker。
- train、validation、competition 與 final map 的版本與 seeds。

## 6. 各組 TODO

### Group A：Game Engine

- [ ]  初始設定頁：支援輸入/保存 `group_id`、`username`、server base URL
- [ ]  Train map 選擇：`easy` / `hard` / `random`
- [ ]  Import/從零開始：支援載入 `parent_a`、`parent_b` 或全新訓練
- [ ]  訓練時追蹤 parents：依 fitness score 持續維護 `parent_a` / `parent_b`（並顯示上一輪 parents）
- [ ]  操作支援：換 seed/reset（地圖不變）
- [ ]  操作支援：切到 validation（`easy`/`hard`）僅 inference、不 breed、不 mutation、不更新 parents
- [ ]  操作支援：切換地圖難度/隨機地圖（保留 weights/biases/lineage；重置位置/碰撞/圈數/計時/metrics/畫面暫存）
- [ ]  操作支援：return to home page（返回前詢問是否 export）
- [ ]  Export：輸出 `parent_a`、`parent_b`，格式符合共同 model JSON（shape 檢查）
- [ ]  Submission 流程 UI：選擇提交 `easy` 或 `hard`
- [ ]  Submission 前 GET 檢查：顯示是否可提交與 `next_submission_at`
- [ ]  Candidate 生成：用 `parent_a`/`parent_b` 產生 20 台（0/1 保留 parents；2-19 crossover+mutation）
- [ ]  Local winner run：固定同一張 competition map / spawn / limits / scoring rules，並計算正式 ranking tuple
- [ ]  Winner 選擇：依 spec 排序（completed 優先；否則比 `max_progress`/`ticks_to_max_progress`；再比 `candidate_index` tie-break）
- [ ]  Payload 組裝/送出：只提交 local winner 的 model + `group_id`/`username` + `client_result`
- [ ]  Submission 狀態顯示：`queued`/`running`/`completed`/`failed` + cooldown 提示

### Group B：GA Fitness Function Research

- [ ]  定義 RunMetrics：與 Game Engine/Server 對齊 `completed`、`lap_ticks`、`max_progress`、`ticks_to_max_progress` 等定義
- [ ]  Fitness term 實作：至少 10 個可獨立啟用/停用的 functions（含描述、預設權重、可調範圍）
- [ ]  Score API：提供 function registry（名稱/說明/預設權重/min/max/enabled_by_default）
- [ ]  Weighted combination：輸出 total score 與完整 breakdown（raw/weights/weighted）
- [ ]  Starter presets：至少 3 組（progress-first / speed-first / stable-generalist），可被使用者覆寫
- [ ]  Panel data API：產出 `FitnessPanelData`（每台車 breakdown、current top 2、previous parents、highlight flags）
- [ ]  Parent selection API：提供選 top-2 parents 的統一邏輯（k=2）
- [ ]  實驗與比較：固定 seeds/tracks 比較不同組合，記錄 completion rate、max progress、lap ticks、collision rate
- [ ]  建議值產出：推薦 default preset 與各 term 權重範圍
- [ ]  與正式 ranking 一致性驗證：確保 local winner 最終仍用 spec v2 ranking tuple（fitness 不取代 leaderboard）

### Group C：Competition Server

- [ ]  API 介面：實作
    - [ ]  `POST /v2/competitions/{competition_id}/submissions`（competition_id=easy|hard）
    - [ ]  `GET /v2/competitions/{competition_id}/submissions/{submission_id}`
    - [ ]  `POST /v2/finals/submissions`
- [ ]  Payload 驗證：欄位存在（`group_id`/`username`/`weights`/`biases`/`client_result`）+ shape 驗證 + 拒絕 NaN/Inf + payload size 限制
- [ ]  Cooldown / Rate limit：每人每 5 分鐘最多 1 次成功 submission；無效 payload 不占 cooldown；429 回應
- [ ]  Replay queue：easy/hard 各自獨立 queue + leaderboard
- [ ]  Replay batch：每 5 分鐘建立 batch；首次 playback 建紀錄+更新 leaderboard；後續 loop 只重播不重排
- [ ]  Deferred 處理：未趕上本輪的 submission 顯示 deferred/延後到下一輪
- [ ]  Ranking：server 不 breed/mutation/selection、不重算分數；直接用 client 上傳 `client_result` 依正式 ranking tuple 排序
- [ ]  Dashboard/Leaderboard：每 5 分鐘 snapshot；每人以歷史最佳 completed 進榜；顯示 rank、group id、username、submission id、狀態、圈速/最大進度、replay 完成時間、next submission time
- [ ]  Replay/稽核資料保存：保存 model、client result、map id/version/seed、simulation version、終止原因、run metrics、batch window、included/deferred ids、leaderboard snapshot、tick replay/log
- [ ]  Finals：每隊僅一次 final model submission；Final Hard Map inference only；同 spawn/limits/rules；產生 Final leaderboard