# World 扇区代理所有权与增量驻留

日期：2026-10-15。实现位于 `prism_world_gpu::{ProxyRegistry,GpuSectorProxies}` 和 `prism_world_runtime/proxy_host`。这是已实现的派生数据生命周期，不另立 scene 内容真相，也不将它称为完整 GPU world scene。

## 从流送到绘制

`StreamScheduler → CellFrameCoordinator → header 校验 → ProxyCellHost → SceneHost`。

`ProxyCellHost` 先调用宿主 `CellProxyDecoder`，再通过注册表验证和预留派生代理，最后调用原始 `SceneHost::load_cell`。原 host 返回错误或 panic unwind 时，私有 guard 删除本次新增的派生扇区；原 host 的部分修改不受补偿，不能将其称为跨 scene/GPU 原子事务。成功驱逐先调用原 host，成功后才删代理；驱逐失败或 unwind 保留代理。panic-abort 不运行析构回滚。adapter 析构只结束借用，不擅自驱逐内容。

decoder 收到校验后、已去除 header 的原始 payload；格式和内容规则由项目提供。decoder 的临时分配和计算工作由宿主限制，注册表预算不是解码器的内存上限。`proxy_host` 只启用 CPU 注册表与 std，不启用 wgpu 或创建设备。

绘制方持有 `GpuSectorProxies`：编辑注册表后调用 `upload`，再 `write_camera → encode → 下游 draw_indirect → submit`。下帧的上传、camera 写入和 encode 必须发生在本帧提交之后，不要求主机等待 GPU。已有 pending edit 时，`resident`、camera 写入和 encode 均拒绝操作，防止旧 GPU 数据被误作当前世界。整体替换借出的注册表会被私有 Arc 身份校验拒绝，即使新旧容量相同也不能绕过同步；应重建 GPU owner。保留旧身份 token 防止分配地址重用造成 ABA。

## 身份、容量与卸载

- `ProxyHandle` 是本 registry 内的 `(u32 slot,u64 generation)`，不是跨 registry、跨存档或网络身份。代际只增不回绕，到 `u64::MAX` 永久退役该槽；可复用容量可能减少。
- dense slot 是当前 GPU source slot，不是持久身份。删除 sector 对其成员逐一 swap-remove，并修正被移动成员的 handle 映射。幸存 sector 的 handle 不变，其 dense 位置可能变化。
- 同一 sector 内 proxy ID 必须唯一；不同 sector 可重复。更新不能改 ID。全部浮点值必须有限，半径、几何误差与最大距离非负。返回的校验、容量和 Vec 预留错误发生在修改之前。
- 代理数量和扇区数量独立有界。`new(capacity)` 默认最多 `capacity.max(1)` 个 sector；`with_sector_capacity(proxy_capacity,sector_capacity)` 支持独立预算。空 sector 也消耗一个扇区名额；零扇区预算拒绝全部插入，移除归还一个名额。零代理容量仍可记录预算许可的空 sector。
- CPU 核心向量在创建时 fallible reserve，此后不超出代理逻辑容量。sector map 节点按需普通分配，GPU feature 下的身份 Arc 也普通分配，OOM 可 abort；不承诺 map 元数据或物理 RSS 的精确硬上限。容量不足只报错，不自行选择驱逐或降质策略。

## 增量上传和复杂度

插入验证对本次 K 个 ID 排序，复杂度 `O(K log K + log S)`；移除为 `O(K + log S)`；handle 查找/更新为 `O(1)`。这里 S 为已注册扇区数。更新即使值相同也保守标脏，不做浮点近似比较。

相邻 dirty rows 合为一个区间，区间内转置为三列 SoA，调用既有 `write_range`。全部区间和 active count 同步成功后才清除标记。一次改变后的 `upload` 扫描 active N 并转置 K 行，清理标记亦扫描 N，即总计 `O(N + K)`，不是总计 `O(K)`；R 个非空区间产生 `3R` 次列写入，逻辑载荷为 `48K` 字节。改变 count/camera 另按驻留 API 写入 128 字节 uniform。尾删即使没有 dirty row 也必须更新 count；全部卸载会在下次 encode 内清零 indirect instance count。无变化 `upload` 为 `O(1)`，不扫描、不转置、不写队列。

scratch 预留 `48 × capacity` 字节列载荷，固定 GPU buffers 为 `72 × max(capacity,1) + 144` 字节。另有 registry 向量、sector map、resident 初始化追踪、pipeline/bind group 和 wgpu staging。这些是逻辑载荷/descriptor 字节，不是 VRAM、RSS 或物理传输。`ProxyUploadStats` 明确报告区间数、转置行数、逻辑列字节和当前 count。构造先验证设备限制，再分配容量级 CPU 存储；upload/encode 不创建 GPU 资源，不提交、等待或回读。

## 回归证据

2026-10-15，Apple M2、Metal、IntegratedGpu、无额外必需 device feature。严格真实 GPU 测试不允许无 adapter、CPU adapter 或 backend 不符时跳过。

`resident_visibility` 的扇区回归执行真实 `draw_indirect` 并比较像素：两个扇区重复 proxy ID、分离 dirty ranges、删除中间成员后的交换压紧、幸存 handle、旧 handle 拒绝、重用槽、追加成员、尾删/全卸载、零容量、空 sector 预算和 buffer 身份不变。coordinator 回归真实执行 stream demand、manifest header、宿主 decoder/SceneHost，再通过同一 registry 上传并间接绘制；无需求后的策略驱逐让旧像素消失，trace 可重放。内存 provider 用例与真实文件归档用例分开；后者还验证非零文件 offset、7 字节分块跨 header、checksum/provider 拒绝和 header 版本拒绝均不进入 decoder/host，只有有效 cell 占用唯一 sector 名额。文件兼容 provider 每次 poll 仍为有界阻塞读取，不将它称为 GPU 直传或异步 reactor 同场验收。两者均使用最小 authoritative host，不是产品 ECS/render graph 接入。

CPU 回归包含 24,000 步独立 sector/dense 参考模型、数值错误的完整状态不变性、ID/预算错误优先级、空 sector flood、代际耗尽退役与核心向量反复满载复用。host 回归覆盖 decoder 错误/unwind、host load 错误/unwind 的派生补偿、host evict 错误/unwind 的状态保留、独立容量与空 sector、错误类型约束及真实 coordinator header 拒绝。

标准复现入口：

```sh
WGPU_BACKEND=metal PRISM_REQUIRE_GPU=1 cargo test --offline -p prism_world_gpu --features gpu --test resident_visibility --test gpu_parity --test wgsl_compile -j1 -- --nocapture
cargo test --offline -p prism_world_gpu --features gpu --lib -j1
cargo test --offline -p prism_world_runtime --features proxy_host -j1
cargo test --offline -p prism_world_runtime --features reactor,proxy_host -j1
```

本轮早期根 workspace 曾被并行任务未完成的 `tools/ecs-perf` manifest 阻断；该阻断随后由并行任务解除，`cargo metadata --offline --no-deps` 已成功。未修改该目录或根 workspace 规避。GPU 复现使用 `/private/tmp/prism-world-sector-validation-tj4W0J/Cargo.toml`：真实 crate 路径依赖、原始 test/bench 源码显式 targets，库原始 `src/lib.rs` 注册为 `world_gpu_unit`，避免无关 terrain→Mesh→asset/ECS dev-dependency 闭包。runtime feature 矩阵使用正常根 workspace 与独立 target。这些不是复制模块、mock GPU 或全 workspace 通过的声明；临时 consumer manifest 不是交付 API，长期入口是上述仓库命令。

最终本轮 GPU harness 为库原始源码 37 项、`resident_visibility` 6 项（5 项真实设备、1 项 shader validation）、parity 1 项、独立 visibility WGSL validation 1 项，合计 45 passed、0 failed、0 ignored。关闭 GPU feature 后原始库源码 32 项通过。harness 全目标（含新增 CPU bench）clippy `-D warnings` 通过；正常根 workspace 的 `cargo clippy --offline -p prism_world_gpu --features gpu --lib -- -D warnings` 亦通过。这不是全部 workspace 的 clippy 声明。

正常 workspace 的 runtime 矩阵均显式 `--no-default-features`，offline、jobs=1、debug=0、incremental=0，独占 target `/private/tmp/prism-world-runtime-build-sdndBX`：

| feature | passed / failed |
|---|---:|
| 无 | 6 / 0 |
| `std` | 53 / 0 |
| `proxy_host` | 73 / 0 |
| `tasks` | 69 / 0 |
| `tasks,proxy_host` | 89 / 0 |
| `reactor` | 73 / 0 |
| `reactor,proxy_host` | 93 / 0 |

无 feature、`proxy_host`、`reactor,proxy_host` 三组 runtime 全目标 clippy 均通过 `-D warnings`。不同 feature 组合重复执行公共测试，不把上述行数相加称为独立用例数。

## CPU registry release 微基准

源码 `pkg/prism_world_gpu/benches/registry.rs`。Apple Silicon/Darwin arm64，rustc 1.96.1；真实 crate path dependency、`default-features=false`，没有 stub、GPU、IO 或 host 工作。每操作 16 预热、128 样本；构造+插入单独 8 样本（P95/P99 均为最大值）。最近秩分位数；pending 每批 4096 次，update 每批 64 次，报告批次摊销 ns/op，不能把其分位数当作逐调用尾延迟。创建为单 sector，稳态为三个 sector；删除/重新插入的是只有一个代理的 sector。

| 操作，P50 / P95 / P99 ns/op | 10,000 代理 | 100,000 代理 | 1,000,000 代理 |
|---|---:|---:|---:|
| 构造+全部插入 | 71,958 / 72,250 / 72,250 | 793,583 / 1,528,250 / 1,528,250 | 49,707,125 / 150,097,500 / 150,097,500 |
| pending=false | 0.49 / 0.50 / 0.63 | 0.48 / 0.50 / 0.50 | 0.49 / 0.50 / 0.51 |
| pending=true | 0.49 / 0.50 / 0.51 | 0.48 / 0.51 / 403.97 | 0.49 / 0.51 / 0.52 |
| 全 dirty prefix 扫描 | 2,000 / 2,083 / 2,084 | 36,209 / 1,909,250 / 9,266,833 | 363,791 / 8,526,083 / 36,494,542 |
| 单代理 update | 4.56 / 4.56 / 5.22 | 11.06 / 11.72 / 12.38 | 11.06 / 11.72 / 11.72 |
| swap-remove | 42 / 125 / 209 | 83 / 291 / 875 | 250 / 1,167 / 1,375 |
| swap 后 reinsert | 84 / 250 / 1,166 | 125 / 375 / 1,417 | 416 / 2,042 / 2,500 |
| tail-remove | 83 / 125 / 125 | 42 / 292 / 833 | 375 / 1,375 / 1,750 |
| tail 后 reinsert | 208 / 250 / 584 | 84 / 542 / 1,375 | 583 / 2,125 / 16,541 |

输入生成、断言、精确多重集合/handle 验证、checksum 和析构均在计时外。CPU 公共 API 不能 acknowledge，因此扫描测的是全 dirty prefix，不是 upload 后 clean/sparse 状态。共享机器尾延迟存在明显扰动，保留全部异常值，不作为隔离性能基线或产品帧时结论。创建开销包含向量预留与返回 handles；其他行不含 fixture setup。只测逻辑数量，不测 RSS/物理显存。

仓库复现：`cargo bench --offline -p prism_world_gpu --no-default-features --bench registry -j1`。本轮使用 `/private/tmp/cpu-registry-benchmark-0w97YI/Cargo.toml` 独立 consumer，仅依赖真实 CPU crate 并指向原始 bench 源码，以免编译无关 GPU/terrain dev dependencies；release clippy 已通过。其独占 target 已清理，manifest/lock 保留可重建。

## 保持独立验收的内容

这条链路不生成 GPU 散布候选，不做密度淡入、HZB、风或交互场，也不实现多材质/多网格 draw 分桶。位置由 decoder 提供一致的 camera-relative 空间，不自动进行全局 i64 坐标转换。产品 scene/ECS decoder、renderer graph、完整资产包、异步 reactor 与 GPU 的同场产品验收、物理显存峰值和多后端测试仍须分别完成。之前 resident 百万实例 encode/submit 基准不包含本注册表与 host 开销，不能移作本链路的吞吐数字。
