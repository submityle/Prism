# World 毫米坐标驱动 SDF focus

本切片连接设计 §17 的世界位置与已有 runtime→GPU clipmap 上传链路，保持
`SparseSdfBrickGrid` 为样本源。宿主每次提供 `[i64; 3]` 全局毫米坐标，无需先缩窄成
体素坐标；不硬编码相机、角色或 renderer，不改变 Scene/override 权威。

## 坐标与迟滞契约

`WorldSdfFocusConfig` 固定 origin、mip0 voxel 毫米尺度和 brick size（1..=32）。
brick extent `B = u64(voxel_mm) * brick_size`；相减先扩展到 i128，逐轴计算
`raw = floor((position_mm - origin_mm) / B)`，checked 转为 i32。负数使用 Euclidean
floor，而非向零截断。相反 i64 极值不会在相减时溢出；raw 超出 i32 时返回明确错误，
不钳制、不 wrap，迟滞不能掩盖越界。

`WorldSdfFocusDriver` 独立处理 XYZ 轴：当前轴 center 为 c 时，在
`[c*B-h, (c+1)*B+h)` 保持原值，离开该半开区间才跳至 raw。`0 <= h < B`，首次
成功发布前不使用迟滞。失败不会提前设置 center，重试使用同一规则。该定义是毫米区间
迟滞，不是整数 brick deadband；一次 teleport 直接计算目标，不逐 brick 追赶。

driver 的 `update_with` 只在 center 改变时调用一次 downstream；回调成功后才提交
focus。回调必须自行保证错误原子性与匹配 planner 的独占所有权，任意回调内部已发生的
修改不能由 driver 回滚。panic 保留 focus，但也不能回滚回调副作用。

## GPU owner 与实际覆盖

`WorldSdfGpuFocus` 私有持有 driver 和 `WorldSdfGpuClipmap`，只暴露只读引用。
`update_focus(grid, position_mm)` 在每次调用时检查 grid 的 brick size/mip levels，
包括 stationary no-op；mapping、planner 或上传失败均保留 focus、CPU/GPU ring 窗口、
revision、dirty 及已有 GPU 样本。

现有 ring 是每 mip 一个 XZ 正方形和**一个 Y 平面**，并非完整三维窗口。为保证
actual focus 所在 brick 仍在窗口内，GPU owner 在分配前强制验证：

- Y 迟滞必须为 0，避免 focus 换层后还保留旧平面。
- X/Z 任一迟滞非零时 radius 必须至少 1；margin 小于 B，所以每轴最多落后 raw 一个
  mip0 brick，各更粗 mip 也不会超过一个 brick。
- 零迟滞允许 radius 0。通用 CPU driver 仍可表达 Y 迟滞；调用方可用
  `validate_single_slice_coverage` 验证与单平面 planner 的配对。

这是窗口覆盖，不保证 source 存在、查询所需邻域完整或距离/梯度可用；缺失样本仍为
Unknown。mip0 Y 改变不一定改变所有粗 mip 的 Y；改变的层替换整层而非 XZ 边条。

`mark_dirty`、`flush_dirty`、`submit_queries` 保持上传 owner 的预算与同提交消费
契约。stationary focus 不清 dirty，不自动 flush；宿主必须独立服务编辑。pending
dirty 时查询 fail closed。成功只表示 queue writes staging/提交，不表示设备已经完成。
对象变化自动观察、bake 完成协议和设备恢复不由这个 owner 实现。

## 性能与资源

坐标 mapping 与 retained focus 的操作数固定，不依赖驻留 brick 数、跳跃距离或 dirty
数量；driver 本体只有 config/center，无堆容器或分配调用。GPU stationary 路径不调用
planner、上传或 consumer，也不改变 revision。changed focus 保留既有 planner 和 GPU
staging 的分配、排序与预算成本，不能宣称整链路零分配。

CPU release 微基准位于 `tests/sdf_focus_performance.rs`，每场景 16 轮预热、128 轮
4096 次更新，记录 batch 平均的 nearest-rank P50/P95/P99；retained、crossing、teleport
使用相同空 downstream；retained 断言 downstream 调用数为 0，另外两个场景断言每次
都发布新 center。crossing 越过迟滞上端点，而不仅是 raw brick 边界。它不测 scheduler、
GPU、单次调用尾延迟或产品帧时，也没有 allocator 运行时计数。

2026-10-21 Apple M2、release、本机存在其他并行工作时最后一次实测，单位为
batch 摊销的 CPU ns/update：

| 场景 | P50 | P95 | P99 |
|---|---:|---:|---:|
| retained | 21.912 | 55.837 | 81.431 |
| crossing | 20.365 | 54.484 | 55.166 |
| teleport | 20.813 | 21.037 | 23.275 |

同轮较早运行的 retained P99 为 23.000 ns；这里保留较晚运行的尾部抖动，不能以此
推导产品负载加速比或实时 deadline 保证。GPU 测试断言逻辑 buffer 字节总额不变，
不等于证明没有同尺寸重分配、峰值物理显存或整条链路零分配。

## 验证入口

```bash
export CARGO_TARGET_DIR=/private/tmp/prism-world-focus-verify.4zQWan
export CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0
cargo test --offline -j1 -p prism_world_runtime --lib
cargo check --offline -j1 -p prism_world_runtime --no-default-features --lib
cargo test --offline -j1 -p prism_world_runtime --features gpu_proxy --lib
cargo test --offline -j1 -p prism_world_gpu --features gpu --lib
cargo clippy --offline -j1 -p prism_world_runtime --features gpu_proxy --lib --test sdf_focus_gpu --test sdf_focus_performance --no-deps -- -D warnings
WGPU_BACKEND=metal PRISM_REQUIRE_GPU=1 cargo test --offline -j1 -p prism_world_runtime --features gpu_proxy --test sdf_focus_gpu --test sdf_clipmap_gpu -- --nocapture --test-threads=1
WGPU_BACKEND=metal PRISM_REQUIRE_GPU=1 cargo test --offline -j1 -p prism_world_runtime --features gpu_proxy --lib gpu_proxy_cell_host_tests -- --ignored --nocapture --test-threads=1
CARGO_TARGET_DIR=/private/tmp/prism-world-focus-perf.4zQWan CARGO_PROFILE_RELEASE_DEBUG=0 cargo test --offline -j1 --release -p prism_world_runtime --test sdf_focus_performance -- --ignored --nocapture
```

## 2026-10-21 验证结果与范围

初次验证时主工作树 HEAD 为 `cd9e374273ba1d0733d2428437bddebfd39603d7`；交付前在
`bb6619e8b8e8d8f908bea864018154680fec339d` 重新执行下表的编译、单测、Clippy 和实卡
入口，结果一致，并显式补跑 7 项默认忽略的宿主测试。两次均含共享未提交改动，不能
称作干净 checkout 验收。使用上文独立 target、离线、禁用 incremental/debug info：

| 入口 | 结果 | 证据范围 |
|---|---|---|
| runtime `--lib` | 148 passed | 含 14 项 focus CPU 测试 |
| runtime `--no-default-features --lib` check | 通过 | no-default 编译，不是测试执行 |
| runtime `--features gpu_proxy --lib` | 168 passed，7 ignored | 本入口默认忽略 7 项，下列显式命令另行执行 |
| runtime `gpu_proxy_cell_host_tests --ignored` | 7 passed，无跳过 | 跨沙盒硬件 adapter 门禁，覆盖 load/reload/evict 和 host error/panic |
| runtime `gpu_proxy` lib / focus 两项 integration 的 Clippy | `-D warnings` 通过 | 不代表整个 workspace 的 Clippy |
| `sdf_focus_gpu` | 7 passed，无跳过 | Apple M2 / Metal / IntegratedGpu，pipeline + dispatch + 独立 readback |
| `sdf_clipmap_gpu` | 4 passed，无跳过 | 同一真实 adapter 上复验既有上传与 consumer 顺序 |
| `sdf_focus_performance` release ignored | 1 passed | 仅上文 CPU driver 微基准 |
| `prism_world_gpu --features gpu --lib`（主工作树） | 编译失败，exit 101 | `prism_asset/src/scheduler.rs:806,944,971` 的 `AtomicUsize` 未导入及 `Flight.consumers` 不存在；未修改该 agent 文件 |

GPU 专项先在沙盒内遇到 `metal found no adapters`，没有按成功或跳过计数；随后经授权
跨沙盒执行，强制硬件 adapter，无软件回退。7 项 focus 测试覆盖负坐标/非零 origin、
多 mip、独立 X/Z 半开迟滞、Y 换层、stationary grid mismatch、mapping/ring 越界、
首次 303/304 字节边界、teleport 336/304 字节失败原子性与重试、pending dirty 查询
拒绝、35/36 字节 dirty 上传及 32 字节删除失效，还包括携带 dirty 的成功滚动：保留
窗口内编辑、丢弃退出编辑、上传进入数据及后续 bounded flush。混合窗口外 mark
请求原子拒绝，不默默过滤。新 focus 测试的 GPU poll 和 map callback 各限制为 30 秒。
独立 readback 在后续上传/提交全部暂存后才读取，避免同步等待掩盖队列顺序问题；
回滚范围是同步返回错误，不包含异步设备丢失或任意回调 panic 的副作用。

此前同轮另有隔离源码副本中的 GPU lib **123 passed**：asset 等依赖来自已提交
`d7eeb443c4cc6fbb28e3e32d82b540a5a03e8e31`，runtime/gpu/residency 刷新自当时 HEAD
的已提交版本，并叠加本切片源码。该副本使用最小 runtime/gpu workspace、独立 lockfile，
不是当前主树原样重现，也没有把其 123 项算作上述主树命令成功。主树 GPU lib 的依赖
编译阻断仍须由对应模块解决。本切片已通过的 runtime GPU 路径不依赖那条失败的 dev
dependency 编译闭包。

## 产品边界

该 owner 可直接接收宿主每 tick 提供的相机/其他焦点位置，关闭了手工 world-mm→brick
映射接缝；仍不是设计 §17 全部交付。renderer 相机 tick 的产品装配、对象自动脏传播及
bake 编排、physics solver/GI/renderer/nav 真实消费者、跨消费者累计帧预算、三维窗口、
多设备及设备恢复仍须独立端到端验收，不以本切片或微基准替代。
