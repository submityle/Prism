# Prism World SDF Physics Runtime

本文记录 World sparse SDF 到 physics 查询消费者的运行时边界。它是派生数据接缝，不改变 `prism_scene` 作者真相，也不宣称已经把 SDF 接入刚体 solver、渲染或 GI。

## 已交付

- `prism_world_physics_bridge` 为 sparse brick 增加显式 `origin_mm`、`voxel_size_mm`、`distance_scale_mm` 与 `revision`，拒绝零尺度。
- `WorldSdfPhysicsBridge::stage` 在发布前检查 brick、sample 和逻辑字节预算；`publish` 对版本保持单调，失败保留旧快照。
- `prism_world_physics_cell_runtime::SdfPhysicsCellHost` 将解码、每 cell staging、聚合驻留预算和 `SceneHost` 生命周期接成一个提交点；reload、decode、revision、预算或 host 失败都不替换旧快照。
- 查询入口按 cell 返回 `Value`、`Miss` 或 `Unknown`。未驻留 cell 和缺失 brick 都是 `Unknown`，不会被解释为空场。
- 物理查询快照保留量化 grid 与世界尺度，逻辑 footprint 计为 48-byte header、16-byte brick metadata 和 2-byte sample；这不是 allocator RSS 或 GPU 显存峰值。

## SDF cell 派生缓存格式（设计 §17.1–17.3）

`SdfCellCodec` 实现已有 `SdfCellDecoder` 接口。PSDC v1 使用固定 128-byte header、按完整 `(x,y,z,mip)` 严格递增的 brick records 和末尾 CRC32 IEEE；每个 record 为 16-byte 完整键/保留位，加 `edge³` 个 x-fastest 小端有符号 i16 样本。保留所有量化值、物理尺度、i64 原点和 u64 revision，不把稀疏缺失区域填成空场。

| Header 偏移 | 内容 |
|---|---|
| 0–15 | PSDC、u16 version=1、u16 flags=0、u32 header size=128、u32 record header size=16 |
| 16–31 | u64 body bytes、u32 brick count、u8 edge、u8 mip count、2-byte reserved=0 |
| 32–63 | u8 sector level、7-byte reserved=0、三个小端 i64 sector 坐标 |
| 64–87 | 三个小端 i64 origin_mm |
| 88–103 | u32 voxel_size_mm、u32 distance_scale_mm、u64 revision |
| 104–127 | u32 samples_per_brick、20-byte reserved=0 |

Record 的 0/4/8 偏移分别为 i32 X/Y/Z，12 为 u8 mip，13–15 必须为零；样本紧随其后。总长度严格为 `128 + brick_count × (16 + 2 × edge³) + 4`。空 grid 仍保留尺度和 revision，总长 132 字节。edge 支持 1–32，mip count 支持 1–255；存储支持的高 mip 不意味着物理查询尺度一定可计算，查询仍执行既有溢出检查。

四项独立预算限制完整 payload 字节、brick 数、总样本数和解码逻辑字节（`48 + 16B + 2S`）。解码先检查尺寸、尺度、计数、精确长度、CRC、manifest sector，以及所有 record 的保留位、mip、排序与唯一性，再分配样本；失败不返回部分 grid。编码拒绝带 dirty invalidation 的 grid，避免缓存静默丢失待重烘标记。编码直接借用样本，不创建另一份完整 grid；现有坐标枚举仍分配一个 keys Vec。CRC 使用编译期查表，每字节一次查询；它用于检测损坏，不提供篡改认证。

缓存是可丢弃的派生数据。codec 只校验 manifest sector，不把 opaque derived_hash 解释为内容摘要。外层 World PRWC header 仍由 coordinator 校验并剥离；`manifest.io_bytes` 包含外层 header，codec 的 payload 限额只覆盖 PSDC frame。宿主负责原子发布与 revision 单调检查。文件集成测试使用真实 archive offset 和 7-byte 分块读取，覆盖损坏/旧版拒绝、下游拒绝、成功替换与驱逐，以及同一 coordinator 修复缓存后的重试。

样本和输出 Vec 使用 fallible reservation；现有 grid 的 BTreeMap 插入及 keys 枚举沿用分配器的 infallible 行为。逻辑预算不覆盖 BTreeMap 节点、分配器元数据、输入 frame、编码 scratch、旧快照与候选快照并存期间的实际峰值，不能据此宣称 RSS 或 VRAM 有同等上限。

### 编解码多维质量评估

| 维度 | 当前实现与取舍 |
|---|---|
| 性能 | 完整预检后构造，CRC 查表，编码不复制完整 grid；加载阶段仍有 Vec/树节点分配及 `O(B log B + S)` 索引/样本工作 |
| 效果 | i16 样本、完整键与物理尺度逐值保留，独立 wire oracle 与变异测试验证；不恢复原始未量化几何 |
| 易用性 | 正式 decoder 可直接交给既有 SDF host，无需项目自定义二进制格式 |
| 易维护性 | 独立 codec 模块，格式与业务宿主分离，测试收口 text |
| 易拓展性 | 版本、保留位与显式预算固定，未知版本拒绝；项目仍可实现其他 decoder |
| 易迭代性 | 增量 API，保留既有 terrain codec 与 SDF 查询接口 |
| 健壮性 | checked arithmetic、完整预检、CRC、sector identity、dirty 拒绝与旧快照保留；未做分配失败注入 |
| 安全 | 无 unsafe，固定长度校验后读取；CRC 不承担认证，实际内存上限仍由宿主管理 |

## 连续距离与梯度查询（设计 §17.2）

`SdfPhysicsSnapshot::query_world_distance_gradient_mm` 和 cell host 的同名入口返回 `SdfDistanceGradient`：`distance_mm` 为三线性插值距离，`gradient` 为该插值对世界 X/Y/Z 的解析导数，单位为 mm/mm。梯度朝距离增加方向，不假设长度为一；`unit_normal()` 返回单位方向，平坦场返回 `None`。原 `query_world_distance_mm` 保持所在体素的整数距离语义。

查询只读取当前已提交快照。世界位置与原点先做 `i128` 差、欧几里得商与余数，再把局部比例转成 `f64`，避免大世界原点吞掉毫米差。梯度同时使用距离尺度和 mip 后体素间距。八个角点可跨 brick；任一角点缺失都返回 `Unknown`，包括角点权重为零的格点位置，不伪造空场或法线。当前 cell 以外的其他快照不会被混合采样；提供者须为边界准备同一快照内一致的 halo。

距离在完整相邻采样区间内连续，格点梯度采用正向相邻区间的单侧导数，跨区间不保证梯度连续。插值不恢复量化/截断前的精确几何距离，也不保证梯度归一。角点超出 i32 体素 ABI 或 mip 间距算术溢出时返回错误。非二次幂 brick 在最低 i32 体素处使用欧几里得余数寻址，避免构造越界 brick 原点。

查询栈上保存八个样本，每次最多八次只读 brick 查询，复杂度为 `O(8 log B)`；无堆分配、锁、快照复制或网格修改。先对角点作差再插值，常量场的梯度严格为零。cell 入口使用同一版本快照计算距离和梯度；load/reload/evict 失败时完整保留旧查询结果及预算统计。

专项测试位于 `text/unit/sdf_sampling.rs` 与 `text/integration/sdf_sampling_lifecycle.rs`，以独立解析仿射场、含 xyz 混合项的场及有限差分验证数值；覆盖负坐标跨八个 brick、每个角点缺失、i64 世界原点、mip/距离尺度、i32 边界、饱和值、格点单侧梯度和宿主失败原子性。

### 多维质量评估

| 维度 | 当前实现与取舍 |
|---|---|
| 性能 | 固定八角点、无分配无锁；稀疏索引查询仍随 brick 数增长，微基准不代替产品帧时 |
| 效果 | 同一插值的解析导数，解析场与有限差分校验；不宣称精确几何距离或连续法线 |
| 易用性 | 快照/cell 同名入口，显式物理单位，平坦场无任意法线 |
| 易维护性 | 采样独立模块，共用 mip 间距校验，新增测试隔离在 text |
| 易拓展性 | 复用已有版本化 sparse snapshot，其他消费者可使用同一尺度和缺失语义 |
| 易迭代性 | 增量 API，保留原整数查询语义 |
| 健壮性 | 缺角点失败封闭、检查坐标和尺度溢出，宿主失败保留旧发布 |
| 安全 | 无 unsafe，借用不可变快照，角点索引由已验证 brick 尺寸界定 |

## 局部编辑原子重烘（设计 §17.3，SHIPPED）

`WorldSdfPhysicsBridge::rebake_region` 接收某个 mip 上的 inclusive 体素编辑区、严格更新的 source revision、独立工作预算和有符号 i16 场求值闭包。它按每级覆盖矩形一次性推导受影响 brick 与所有更粗父 mip，不重复枚举父键，不枚举或复制整张驻留 grid。调用者须把对象影响半径与采样 halo 纳入编辑区，并确保闭包读取同一版本的完整合成场；该入口不规定对象 CSG、几何作者规则或滤波方式。

重烘先检查所有 mip 的坐标、物理间距及 i128 世界位置端点，再检查工作 brick/sample/scratch 限额、更新后的完整驻留 brick/sample/逻辑字节预算，最后为全部替换 brick fallible 分配样本缓冲。所有这些步骤成功后才调用求值器。求值按 mip 递增、brick z/y/x、brick 内 z/y/x（x-fastest）执行，每个目标 brick 只采样一次。`SdfRebakePoint` 提供完整 brick key、i64 体素位置、精确 i128 世界毫米位置、该 mip 间距与统一距离量化尺度，避免巨大原点或负坐标的 float 舍入。

全部采样成功后才替换/插入目标 brick，并同时更新 revision 与逻辑字节；旧样本在求值期间仍完整保留。未受影响 brick 的样本 allocation 保持原址。闭包返回错误或 unwind 均保留旧快照，可用同一 revision 重试；不回滚求值器自身的外部副作用。现有 grid 已有 dirty invalidation 时拒绝操作，避免误清与本次编辑无关的待处理失效。commit 不调用求值器，现有 BTreeMap 节点分配仍使用 infallible allocator，不承诺从进程级 OOM 恢复。

设受影响总 brick 数为 K、总样本数为 S、驻留 brick 数为 B、mip 数为 M：预检为 `O(M + K log B)`，重烘/提交为 `O(S + K log(B + K))`；scratch 为 `K × size_of::<Replacement>() + 2S` 逻辑字节，尺寸随目标 ABI 而定。它不包含驻留数据、树节点、allocator 元数据或 source 自有内存；不是 RSS/VRAM 上限。每个受影响 brick 都完整重新求值，尚未做 brick 内稀疏样本编辑或已知空场压缩。

重烘成功后可从 `evict()` 的 `Some` 结果取得 snapshot，再调用 `into_source()` 转移所有权至 `SdfCellSnapshot`，无需复制样本，再交给 PSDC codec 和既有 cell host 的聚合预算/宿主提交。真实文件集成测试覆盖本地求值失败后重试、编码、archive offset/分块读取、下游拒绝后旧查询与 usage 保留，以及成功重载后的新梯度/距离。同步 `rebake_region` 不调度任务；下节可选 bridge `tasks` adapter 提供 worker 求值与显式 owner 提交。桥自身不更新 cell host 内部聚合统计；再下一节 cell runtime `tasks` 装配负责该计费与生命周期。上述入口不执行 GPU 上传；相机 clipmap 滚动、对象自动脏传播和多消费者作业调度仍需独立运行时接线。

### 局部重烘多维质量评估

| 维度 | 当前实现与取舍 |
|---|---|
| 性能 | 成本随受影响 brick/sample 增长，无全 grid clone/扫描；prepare 阶段有受预算限制的分配，查询仍保持零分配 |
| 效果 | 父 mip 重新采样完整场，精确整数位置与固定物理单位；不推断未提供的几何影响范围或滤波策略 |
| 易用性 | 单个闭包与报告，错误保留失败 brick/voxel 和 source error；rustdoc 示例可执行 |
| 易维护性 | 独立 rebake 模块，复用原稀疏 grid 和 physics 快照，不修改 residency/terrain/solver |
| 易拓展性 | 场求值由调用者提供，可复用不同对象合成与存储；不绑定项目物理或 gameplay 规则 |
| 易迭代性 | 增量公开 API，原 publish/query/codec 格式保持兼容，源快照可零样本复制转移 |
| 健壮性 | 独立 voxel/parent oracle、每个采样序号失败注入、panic、精确预算与溢出检查；未做 allocator OOM 注入 |
| 安全 | 无 unsafe，独占借用保证提交期间无并发读写，失败不破坏已发布数据；外部求值副作用由调用者负责 |

## 分离式重烘与共享任务池（设计 §17.3，SHIPPED）

同步入口现复用三个阶段：`prepare_rebake(&self, ...)` 完成全部预检和目标样本缓冲分配，返回拥有数据的 `SdfRebakeWork`；`work.evaluate(...)` 在不借用 resident grid 的情况下求值，返回 `SdfRebakeUpdate`；`commit_rebake(&mut self, update)` 检查原发布身份和精确 base revision，成功后才写入。准备和求值期间旧快照可继续查询，不复制或扫描整张 grid。未提交的 work/update 直接丢弃即可；求值错误、unwind 和被拒绝的发布不使其他已准备工作失效。

每次完整加载/重载分配私有 `Arc<()>` 发布身份；work 保留强引用，防止驱逐后同 revision 重载的 ABA。独立 bridge（包括 clone）不能接收源 bridge 的 update。成功局部重烘通过严格递增 revision 使该 base 上所有其他 update 过期，即使编辑区不相交也采用首个提交成功、其余拒绝的语义，不自动合并。身份和 revision 检查均在首个 resident 写入之前；失效返回 `StalePublication`，当前没有快照时返回 `NoSnapshot`。

可选 `tasks` feature 增加 `work.spawn(&TaskPool, &CancelToken, source)`：复用现有 `prism_tasks::TaskPool::spawn_catch`，不另建线程池。闭包须拥有同版本合成场且满足 `Send + 'static`。`SdfRebakeTask<'pool, E>` 实际借用 pool，编译器禁止 pool owner 先于句柄销毁，避免句柄析构触发最后一个 pool 引用的 shutdown/join。worker 数为零或启用 `prism_tasks/single` 时，求值在 spawn 返回前同步完成；该模式不能声称后台执行。

`try_commit` 对尚未完成的任务立即返回 `Ok(None)`，不 help-run 队列。完成后仅收集一次结果并在 owner 独占 bridge 时提交；重复收集返回 `AlreadyCollected`。完成结果使用既有 result-slot mutex，提交还有树查找、替换/节点分配及旧样本释放，不承诺无锁或零分配。求值错误保留完整 brick/voxel/source error；unwind 被捕获为原 panic payload，pool 可继续使用。

每个 task 使用 scope 的独立 child token，单独取消不影响 sibling；父 scope 取消会传播。取消在每次 source 回调之前、最终回调之后及 owner 提交之前检查，因此计算已完成但尚未提交的结果仍可取消。正在执行的回调不能被抢占；最后提交检查之后的并发取消不撤销已开始的提交。丢弃未收集句柄只请求取消、不等待、不发布；pool 的最终 shutdown 由其 owner 管理。

工作 scratch 限额仍为单个 work 的 descriptor 加样本缓冲，不限制同时准备/排队的任务总数或总字节；token/header、allocator/树节点、source 自有数据和 pool 队列均不在该逻辑预算内。直接使用 bridge adapter 的调用者必须额外限制 in-flight 总量，并在 frame 热路径之外安排 prepare/spawn；cell 层可使用下节装配。该基础接口本身没有实现相机/对象调度器、跨消费者预算或 GPU 上传。

### 分离式重烘多维质量评估

| 维度 | 当前实现与取舍 |
|---|---|
| 性能 | 样本求值可在共享 worker 执行，pending poll 不执行作业；准备分配和 owner 提交仍有有界但非零成本 |
| 效果 | 同步/分离/任务路径逐样本一致；后台结果仅在正确发布和 base revision 上可见 |
| 易用性 | 显式 prepare/evaluate/commit 与可选 spawn/try_commit，旧查询持续可用；无 worker 时行为明确 |
| 易维护性 | 独立 task adapter，无额外线程池、不改 cell/solver/renderer，测试隔离在 text |
| 易拓展性 | source 与外部 pool/cancel scope 注入；总任务预算、重试与合并由未来运行时 owner 统一管理 |
| 易迭代性 | 原同步 API 与 codec 不变，默认构建不引入 tasks，新增接口可独立采用 |
| 健壮性 | ABA/foreign/clone/intervening edit 拒绝，取消边界、panic、单次收集及 pool 生命周期有回归 |
| 安全 | 无 unsafe；类型限制 pool 生命周期，私有 update 防伪造，提交独占；OOM 与回调副作用边界保持明确 |

## cell host 后台重烘与聚合预算

对应 World 设计 §17.3 的局部增量更新。cell runtime 的默认关闭 `tasks` feature 增加 `SdfCellRebakeRuntime`，拥有现有 `SdfPhysicsCellHost` 并借用调用者的 `TaskPool`。cell host 内部保存 bridge，公开 `resident()` 仍返回同一查询快照；加载和后台发布均不复制 resident grid。`physics()` 只读，`decoder_mut()` 仅影响未来加载，避免绕过聚合统计的可变 bridge 入口。

`SdfCellRebakeBudget` 约束未收集任务数、总样本数和逻辑 scratch 字节。构造时 fallible 预分配全部 task index；`start` 在准备前先限制任务数，把 per-work 样本/scratch 上限收紧为剩余额度，再调用 bridge prepare。因此拒绝不会先临时分配超出 pending 额度的样本缓冲，也不会调用 source。就绪结果、被取消/被替换但仍执行中的回调继续占用额度，只有 worker terminal 且被 `try_commit(id)` 收集才释放。`pending_usage()` 可用于上层背压；取消并不是完成。

发布增长在 admission 和 commit 两处检查 cell host 的 aggregate bricks/samples/bytes。多个 cell 的工作可能同时通过 admission，先提交者消耗空间后，后提交者重新检查并原子拒绝；不承诺为每个 pending 工作预留最终 resident 容量。成功重烘仅变更派生 physics/revision/usage，不重新调用下游 scene load，不把派生修改冒充作者内容事务。

wrapper 实现 `SceneHost`：只有加载/驱逐成功才取消并标记同 cell 工作失效；decode/stage/宿主返回错误或 unwind 保留旧快照和原工作。驱逐再加载相同 revision 不恢复旧任务的发布资格。第一个成功重烘也使所有同 cell 同 base 工作失效，包含互不相交编辑；不同 cell 工作保留。任务 ID 具有私有 runtime identity 和不复用的 checked u64 序号，foreign/collected ID 被拒绝。

`try_commit` 对未完成工作返回 `None`，不执行 source、不 help-run、不等待 worker。终态成功/失败均只收集一次。新增 bridge `try_discard` 允许 owner 拒绝发布时仍收集 source 错误和原 panic payload；这些错误优先保留，已失效工作的 cancellation 则返回 `Invalidated`。`cancel`/`cancel_all` 不撤销已开始的提交。失败重试须新建工作；没有隐式 merge 或自动重试。

所有任务收集后可 `try_into_host` 取回原 host；仍有任务时返回装箱的原 runtime，这个失败路径分配内存。直接 drop 请求取消且不 join；活动 source 回调不可抢占，可能在 runtime 销毁后继续执行。预算只作用于当前 runtime 生命周期，不能通过连续销毁/重建 runtime 声称全进程受限；有序替换应先取消、逐个收集到 idle，再取回 host。

resident 与 pending 是两套独立逻辑预算，同时驻留时必须合并考虑。task index、Arc/token/队列、source 捕获、allocator/树节点以及重载候选快照不计入 pending scratch，未提供 RSS/VRAM 上限。prepare、spawn、终态结果 mutex、树节点插入和旧样本释放仍有成本；pending ID 查找和同 cell 失效扫描是 O(在途任务数)，index 不在 poll 时增长。相机/对象自动脏传播、跨消费者统一调度、实际 solver/GI/renderer 与 GPU 上传仍未完成。

| 维度 | 本片取舍与证据要求 |
|---|---|
| 性能 | 复用共享 pool；不 clone resident grid，index 预分配，unfinished poll 不求值/等待；线性任务查找须按配置容量实测 |
| 效果 | 明确提交前旧查询可用；新样本、revision 与聚合 usage 同次发布；拒绝不部分写入 |
| 易用性 | SceneHost wrapper、opaque ID、pending usage、cancel/drain 与 idle host 提取 |
| 易维护性 | 生命周期复用原 host；调度单列 rebake 模块，原查询 API 不变 |
| 易拓展性 | 泛型 decoder/scene host/source，使用既有 pool，不绑定几何或 gameplay 规则 |
| 易迭代性 | 默认关闭 tasks，原同步路径保留；失败后可重新提交新工作 |
| 健壮性 | terminal-only 释放；成功-only invalidation；发布前复验总预算；错误与 panic 不被覆盖 |
| 安全 | 无 unsafe；pool 借用约束，私有 ID/bridge 防绕过计费；逻辑预算、不可抢占与 OOM 边界明确 |

## 验证边界

应至少运行：

```text
cargo test -p prism_world_physics_bridge --lib --offline
cargo test -p prism_world_physics_cell_runtime --lib --offline
cargo test -p prism_world_physics_bridge --test sdf_sampling --offline
cargo test -p prism_world_physics_bridge --test sdf_rebake --offline
cargo test -p prism_world_physics_bridge --features tasks --test sdf_rebake_task --offline
cargo test -p prism_world_physics_bridge --features tasks,prism_tasks/single --lib --test sdf_rebake --test sdf_sampling --test sdf_rebake_task --offline
cargo test -p prism_world_physics_bridge --doc --offline
cargo test -p prism_world_physics_bridge --features tasks --doc --offline
cargo test -p prism_world_physics_cell_runtime --test sdf_sampling_lifecycle --offline
cargo test -p prism_world_physics_cell_runtime --test sdf_cell_codec --test sdf_cell_file_runtime --offline
cargo test -p prism_world_physics_cell_runtime --features tasks --test sdf_cell_rebake --offline
cargo bench -p prism_world_physics_bridge --bench sdf_sampling --offline
cargo bench -p prism_world_physics_bridge --bench sdf_rebake --offline
cargo bench -p prism_world_physics_bridge --features tasks --bench sdf_rebake_task --offline
cargo bench -p prism_world_physics_cell_runtime --bench sdf_cell_codec --offline
cargo bench -p prism_world_physics_cell_runtime --features tasks --bench sdf_cell_rebake --offline
```

这些测试证明 cell 生命周期到 physics 查询的原子接入，不证明 `prism_physics_core` 已使用该场生成 SDF 接触，也不证明 `prism_render_scene`、GI 或 GPU renderer graph 已消费同一份 World SDF。后续消费者必须复用 `WorldSdfSnapshot` 的尺度、revision 和 `Unknown` 语义，不能另建未版本化的平行场。

### 2026-10-21 验证记录

独立 target、offline、单编译任务、dev/test debug=0、incremental=0 下，bridge 库 11 项、连续采样 10 项、cell runtime 库 22 项和生命周期集成 1 项全部通过（共 44 项，无忽略）。连续采样包含最小正 subnormal 与最大有限梯度的归一化回归。另行回归 `prism_world_runtime --lib` 108 项与 `prism_world_gpu --features gpu --lib` 123 项全部通过，历史 `LayeredCellManifest::admitted_layers` 编译错误未复现。依赖 `prism_physics_core` 仍报告既有警告，本记录不宣称全 workspace 无警告。

改动 Rust 文件的定向 rustfmt 与 diff whitespace 检查通过。`cargo clippy -p prism_world_physics_bridge -p prism_world_physics_cell_runtime --all-targets --no-deps --offline -- -D warnings` 未通过：bridge 原有代码报告 1 处 `std_instead_of_alloc` 和 3 处 `collapsible_if`，在 cell 检查前终止；没有为本片改动无关代码或放宽 lint。

PSDC v1 后续最终回归使用同样的独立 target/offline 配置，运行 `cargo test -j 1 -p prism_world_physics_bridge -p prism_world_physics_cell_runtime --lib --tests --offline`：原 44 项加 codec 11 项、真实文件集成 2 项，共 57 项通过，0 失败、0 忽略。codec 专项包含独立 wire/CRC oracle、所有截断前缀、完整帧逐位变异、四项独立预算及 signed-i16 极值；文件专项包含同一 coordinator 的损坏缓存修复重试。`cargo rustdoc -j 1 -p prism_world_physics_cell_runtime --lib --offline -- -D warnings` 与改动 Rust 文件的定向 rustfmt 检查通过。

cell crate 的严格 all-target Clippy 仍被既有 `src/sdf.rs:328` 的 `collapsible_if` 阻断；诊断性排除此 lint 后，全目标检查还报告既有 `sdf_sampling_lifecycle` 测试的 `doc_markdown`、`std_instead_of_alloc`。针对本片 lib、`sdf_cell_codec`、`sdf_cell_file_runtime` 和 codec bench 的检查，在命令行使用 `-D warnings -A clippy::collapsible_if` 后通过，没有加入源码 allow 或修改上述既有文件。这是定位新增告警的诊断结果，不是严格 Clippy 门禁通过；依赖 `prism_physics_core` 的既有警告亦未清零。

Apple M2、默认优化 bench profile、bench debug=0、incremental=0，预先逐位置验证 4096 个确定性位置的解析距离/梯度与缺失语义，再运行 128 批、每批 4096 次查询。下表分位数是各批平均 ns/query 的分布，不是单次查询尾延迟；没有独立预热批。计时包括查询及成功结果检查，不包括建图、位置生成、正确性 oracle、输出或销毁。

| 负载 | P50 / P95 / P99（ns/query） | 快照逻辑字节 |
|---|---|---|
| 64 个 edge=8 brick、32768 个样本、混合内部/边界位置 | 172.729 / 183.268 / 196.371 | 66608 |
| 空 grid、首角点缺失即返回 Unknown | 31.820 / 34.587 / 56.773 | 48 |

这是 CPU 查询微基准，不是 GPU timestamp、RSS、实际显存、产品帧时或多消费者负载验收；空 grid 的快速失败也不代表最后一个角点缺失的成本。

### 2026-10-21 cell 编解码基准

Apple M2、默认优化 bench profile、debug=0、incremental=0、offline、单编译任务；确定性 edge=8 signed-i16 数据，8 次预热后交替测量完整 encode/decode 各 128 次。完整 roundtrip oracle 先于预热；每次结果也在停止计时后检查。计时包含 API 内分配、CRC 与样本构造，排除数据准备、相等性检查、结果销毁和文件 IO。分位数为 nearest-rank 的每次完整 API 主机墙钟延迟。

| Brick / 样本数 | Frame / 逻辑快照字节 | Encode P50 / P95 / P99（ns） | Decode P50 / P95 / P99（ns） |
|---|---:|---:|---:|
| 64 / 32768 | 66692 / 66608 | 178250 / 187333 / 193333 | 174291 / 192750 / 203584 |
| 1024 / 524288 | 1065092 / 1065008 | 2910917 / 3265916 / 3979833 | 2849292 / 3216333 / 4171625 |

这只衡量 CPU 派生缓存编解码成本，不包含 coordinator/宿主 staging 的其他开销，不是冷盘、RSS 峰值、GPU 或产品帧预算验收。重载必须为输入 frame、旧/候选快照及索引同时驻留预留实际内存。

### 2026-10-21 局部重烘验证与基准

独立 target `/private/tmp/prism-world-sdf-rebake-20261021.bHIZVI`，offline、单编译任务、dev/test debug=0、incremental=0；bridge/cell 联合 `--lib --tests` 共 71 项通过（bridge lib 11、连续采样 10、重烘 13、cell lib 22、codec 11、文件集成 3、生命周期 1），0 失败、0 忽略。独立 voxel/parent oracle 覆盖负坐标、非零起始 mip 与非二次幂 brick；专项包含全部采样序号失败注入、unwind 恢复、六项工作/驻留预算、edge=1/32、revision=u64::MAX、完整 brick 坐标溢出预检。复审发现并修复了中间乘积溢出但原点抵消后最终 i128 世界位置合法的误拒绝，新增 i128::MAX/MIN 端点回归。

`cargo test -j 1 -p prism_world_physics_bridge --doc --offline` 的 1 项可执行示例通过；`cargo rustdoc -j 1 -p prism_world_physics_bridge --lib --offline -- -D warnings` 通过。严格 Clippy 复验仍被前节记录的 4 处 bridge 既有告警阻断；新增测试的两处 `needless_range_loop` 已修复。bridge lib/rebake test/bench 及 cell lib/真实文件集成，以 `-D warnings -A clippy::collapsible_if -A clippy::std_instead_of_alloc` 定向诊断通过；这不是严格门禁通过，未增加源码 allow。改动 Rust 的定向 rustfmt 与 diff whitespace 检查通过。

Apple M2、`aarch64-apple-darwin`、rustc 1.96.1、默认优化 bench profile、bench debug=0、incremental=0。每例先逐样本验证完整目标数据，再执行 8 次预热与 128 次测量，nearest-rank 分位数；计时包括 API 的预检、staging 分配、确定性整数场求值、替换/提交及旧样本与 staging 销毁，排除初始建图、校验、输出及最终快照销毁。起始 edge=8，3 个 mip；初次更新已分配所需父 brick，计时期间都是 replacement（inserted=0）。

| 起始 mip0 驻留 brick | 编辑覆盖 | 重烘 brick / 样本 | Scratch / 最终快照逻辑字节 | P50 / P95 / P99（ns） |
|---:|---|---:|---:|---:|
| 64 | 单格及父级 | 3 / 1536 | 3192 / 68688 | 31208 / 35541 / 35625 |
| 4096 | 单格及父级 | 3 / 1536 | 3192 / 4261968 | 31208 / 32500 / 35625 |
| 32768 | 单格及父级 | 3 / 1536 | 3192 / 34080848 | 31208 / 33000 / 35834 |
| 64 | 完整 X 行及父级 | 112 / 57344 | 119168 / 116528 | 1163125 / 1175750 / 1182541 |
| 4096 | 完整 X 行及父级 | 7168 / 3670016 | 7626752 / 7454768 | 75524291 / 79003667 / 80703875 |

本负载下固定局部编辑的 P50 均约 31.2 µs；大范围重烘随求值数量增长，最大案例约 75.5 ms，不能放入实时帧热路径。结果包含刻意便宜的量化场回调，不代表真实几何求值、job 调度、GPU、RSS/VRAM、多硬件或产品帧时。实现依赖树查找仍有对数驻留规模项，不能从这三个测量点声称严格 O(1)。

### 2026-10-21 分离式重烘验证与任务基准

独立 target `/private/tmp/prism-world-sdf-task-20261021.h5UGnl`，offline、单编译任务、dev/test debug=0、incremental=0：`cargo test -j1 -p prism_world_physics_bridge -p prism_world_physics_cell_runtime --features prism_world_physics_bridge/tasks --lib --tests --offline` 共 85 项通过（bridge lib 11、重烘 18、task 9、连续采样 10、cell lib 22、codec 11、文件集成 3、生命周期 1），0 失败、0 忽略。真实 worker 测试核对 thread identity 和 worker index，并以 channel gate 阻塞 source，同时验证旧查询、pending poll、drop 不等待；另以无关作业占住同池唯一 worker，验证已完成结果可独立收集。过期结果、取消边界、panic 后 pool 复用均有回归。

默认关闭 tasks 的 `--no-default-features --lib --tests` 共 39 项通过。`--features tasks,prism_tasks/single --lib --test sdf_rebake --test sdf_sampling --test sdf_rebake_task` 共 48 项通过；其中 3 个仅适用于真实 worker 的 channel gate 测试在断言 inline 模式后直接返回，不能把这一配置作为后台执行证据。`--features tasks` doctest 共 3 项通过，含 2 项可执行示例和 1 项禁止 pool 提前销毁的 compile-fail 示例；同 feature 的 `cargo rustdoc --lib -- -D warnings` 通过。生产 adapter 只复用调用者的 pool，测试提供可控的本地 pool；本片未修改 cell、solver 或其他并行模块。

最终严格 `cargo clippy -j1 -p prism_world_physics_bridge --all-targets --features tasks --no-deps --offline -- -D warnings` 仍在库阶段被上述 4 处既有告警阻断。包含最终第 9 项任务测试的定向诊断命令 `cargo clippy -j1 -p prism_world_physics_bridge --lib --test sdf_rebake --test sdf_rebake_task --bench sdf_rebake --bench sdf_rebake_task --features tasks --no-deps --offline -- -D warnings -A clippy::collapsible_if -A clippy::std_instead_of_alloc` 通过；这是命令行排除既有 lint 类后的诊断，不是严格门禁通过，源码没有增加 allow，也未修改其他 agent 的文件。

Apple M2、`aarch64-apple-darwin`、rustc 1.96.1、默认优化 bench profile、bench debug=0、incremental=0；共享 pool 配置 1 个真实 worker。每例 8 次预热、128 次测量，每次停止计时后逐值比较完整 grid 与同步路径 oracle，并检查工作量和 inserted=0；nearest-rank 分位数。初始 grid、pool 与 oracle 构造、结果校验及最终快照销毁均不计时。prepare 包括全部预检/目标缓冲分配；spawn_to_finished 包括 spawn 分配、排队、取消检查、完整求值以及 owner yield-poll 的观测成本；commit 包括结果收集、身份检查、写入和旧样本释放。任务句柄自身的最终销毁在 commit 计时外。

| 起始 mip0 驻留 brick / 编辑 | 重烘 brick / 样本 / scratch 字节 | Prepare P50 / P95 / P99（ns） | Spawn-to-finished P50 / P95 / P99（ns） | Commit P50 / P95 / P99（ns） |
|---|---|---|---|---|
| 64 / 单格及父级 | 3 / 1536 / 3192 | 250 / 1000 / 2000 | 45959 / 93875 / 113792 | 1375 / 8834 / 9292 |
| 32768 / 单格及父级 | 3 / 1536 / 3192 | 1708 / 3875 / 5500 | 102167 / 175292 / 184792 | 1833 / 22584 / 24292 |
| 4096 / 完整 X 行及父级 | 7168 / 3670016 / 7626752 | 558750 / 1348000 / 2123958 | 99136458 / 164211583 / 201487750 | 581834 / 1027083 / 1470500 |

完整行案例 worker 观测 P50 约 99.1 ms、P99 约 201.5 ms，而 owner prepare/commit 各自 P99 仍约 2.124/1.471 ms，说明求值离开 owner 不等于已满足产品帧预算。调度与取消检查有额外开销；此前同步基准和本轮负载均受并行任务影响，不能据此推导加速比或驻留规模无关性。本基准是 CPU 主机阶段墙钟，使用刻意便宜的量化 source，不含真实几何、文件 IO、GPU、RSS/VRAM 或产品多消费者总预算。

### 2026-10-21 cell 后台装配验证

独立 target `/private/tmp/prism-world-cell-rebake-20261021.SotWJk`，offline、单编译任务、dev/test debug=0、incremental=0。关闭 tasks 的 bridge/cell 联合 `--no-default-features --lib --tests` 共 76 项通过；启用 cell `tasks` 的 bridge/cell 联合 `--lib --tests` 共 91 项通过（bridge 49、cell 42），包括真实 worker reload/取消、失败 reload 保留、runtime-scoped ID 与跨 cell 驻留竞争原子拒绝。`tasks,prism_tasks/single` cell 库与专项集成共 27 项通过；其中 2 个依赖真实 worker gate 的测试在 zero-worker 模式按设计返回，其余 inline 语义实际执行。bridge 新增 `try_discard` 覆盖 pending、成功、重复收集、source 错误、原始 panic payload 与完成后取消。文档测试中 bridge 3 项及 cell 新示例 1 项通过。另行重跑 `cargo test -j1 -p prism_world_runtime --lib --offline` 108 项和 `cargo test -j1 -p prism_world_gpu --features gpu --lib --offline` 123 项全部通过；这不构成本轮 GPU shader 实测。

cell 库 `cargo clippy -j1 -p prism_world_physics_cell_runtime --features tasks --lib --no-deps --offline -- -D warnings` 通过。bridge 的库与 task test 在命令行排除既有 `collapsible_if`、`std_instead_of_alloc` 后通过；该结果是新增告警诊断，不是 bridge 严格 lint 门禁通过。依赖 `prism_physics_core` 仍报告既有警告。本片只读复审未发现阻断缺陷；复审核对 terminal 销毁后释放额度、成功后失效、两阶段总驻留预算、错误优先级、pool 借用及无可变入口绕过计费。

### 2026-10-21 cell 后台轮询基准

`aarch64-apple-darwin`、默认优化 bench profile、debug=0、incremental=0、offline、单编译任务；共享 pool 只有一个真实 worker，首任务用 channel gate 阻塞 source，其余任务排队。任务表全部占满，轮询最后一个 ID，覆盖线性查找的最差索引；每项 8 批预热、128 批测量，每批 256 次 `try_commit`，下表为批平均 ns/poll 的 nearest-rank 分位数。计时包含 pending 结果检查，排除 start、求值、发布、构造与销毁。

| 未收集任务数 | 样本 / 逻辑 scratch 字节 | Pending poll P50 / P95 / P99（ns） |
|---:|---:|---:|
| 1 | 8 / 56 | 7 / 8 / 8 |
| 64 | 512 / 3584 | 48 / 48 / 97 |
| 1024 | 8192 / 57344 | 359 / 441 / 1482 |

每项同时断言轮询不释放额度、`cancel_all` 不提前释放额度、gate 放行后逐个收集 cancellation 且 usage 归零、完整旧快照保持不变。这个基准验证真实 worker 下 pending 路径，不衡量单次尾延迟、source 成本、终态收集/驻留发布、RSS/VRAM、GPU 或产品帧预算，也不承诺任意容量下固定成本。
