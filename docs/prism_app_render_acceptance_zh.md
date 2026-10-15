# Prism App / RenderApp 本机验收记录

更新时间：2026-10-16。本文只记录当前工作树在本机可复现的 App 与 RenderApp 证据；不把设计目标、被跳过的硬件测试或单机样本扩展成跨平台结论。`prism_replication`、`prism_asset` 不在本次修改范围内。

## 当前已覆盖

- `prism_app`：无头 Runner、固定步/生命周期、子应用提取与流水线、回滚相关单测均可在 crate 隔离条件下执行。
- `prism_render_app`：null backend 下的 plugin → extract → render schedule → frame graph → submit 闭环，以及资源、队列、管线、恢复和 transform resident 测试均可执行。
- App runner 对拍：同一确定性内容在两个 headless runner 实例中运行，比较生命周期计数、固定步数、overstep 和 trace digest。
- CPU pipeline benchmark：比较 serial、dedicated render 和 task lane；只测合成 CPU 工作负载与调度/流水线指标，不代表 GPU、窗口 present 或游戏内容性能。

## GPU resident columns / material fallback 验收边界

本轮真实验收切片集中在 `prism_render_app`、`prism_render_scene` 和
`prism_render_material`，不把 ECS/world/audio/tasks/replication 的其他改动纳入
结论。

### resident column 的 mapping、同步和生命周期

- 一个 source world 的一个 archetype 对应一个独立 GPU buffer；row 地址是
  `archetype -> chunk -> row`，不会把不同 archetype 的同一 component id 复用到一条
  buffer 上。消费者必须使用同步时返回的 `rows`、`stride` 和 `live_bytes()`，不能读取
  shrink 后的尾部容量。
- 脏判断组合使用 chunk incarnation、structural epoch、live row count 和 column
  revision。这样能覆盖同 tick 重复写、历史 tick 写入、archetype migration、同长度重建
  chunk，以及 tail chunk 重建；clean frame 不发 queue write，dirty chunk 只写对应 span。
- 增长会分配新 buffer 并复制/初始化当前 live rows；缩容保留 high-water capacity。
  `ResidentEcsColumn` 克隆持有 `Arc<OwnedBuffer>`，所以 reset、增长、cache drop 后仍可
  被已记录的消费者读取，直到最后一个 view 被释放。
- extraction 在同一 render runtime 的 queue 上完成 upload；真实 readback 使用
  `copy_buffer_to_buffer`、`Device::poll(wait_indefinitely)`、`map_async(Read)` 和再次
  poll，测试随后显式 `poll_wait()`。这证明的是本次队列提交和读回的同步闭环，不是
  跨队列或跨设备同步保证。
- 设备丢失时清空可绑定 cache 并发布结构化 `DeviceLost`；replacement device 的下一次
  健康 extraction 完整初始化，而不是复用旧 device 的 buffer。旧 view 仍归旧 device
  所有，必须等记录的 work retire 后释放。

### scene/material fallback

- material bind group 在支持 bindless texture 时创建 generation table 和 texture/sampler
  arrays；不支持时使用显式 non-bindless layout。固定的 material/virtual-texture
  fallback 资源从 frame zero 可绑定，未初始化的动态 generation 保持无效 sentinel，
  不能把“未分配 slot”误认为有效材质。
- material capability negotiation 对 required capability 返回结构化 unsupported/error；
  仅 used 的可选扩展会被投影掉并报告 ignored。未知 capability 名称在 schema/registry
  路径拒绝，不静默假设支持。
- 这些 fallback/negotiation 结论来自现有单测、ABI/schema/parity 测试和代码路径；本轮
  没有把每一种 vendor-specific bindless 能力都宣称为真实硬件覆盖。

### 本轮最小 GPU gate

默认运行时没有合适 wgpu adapter 时，硬件测试打印 `SKIPPED` 并通过；设置
`PRISM_REQUIRE_GPU=1` 后同样情况必须失败，错误为 `required GPU: no suitable wgpu
adapter was found`。因此默认通过不等于真实 GPU 已验收。

resident-column gate：

```text
CARGO_TARGET_DIR=/private/tmp/prism-render-validation \
CARGO_BUILD_JOBS=1 \
cargo test --locked -p prism_render_app --all-features --test ecs_columns \
  -- --test-threads=1 --nocapture
```

本机允许访问 Apple M2/Metal adapter 时结果为：`16 passed, 0 failed, 0 ignored`。

强制门禁：

```text
PRISM_REQUIRE_GPU=1 \
CARGO_TARGET_DIR=/private/tmp/prism-render-validation \
CARGO_BUILD_JOBS=1 \
cargo test --locked -p prism_render_app --all-features --test ecs_columns \
  -- --test-threads=1 --nocapture
```

没有物理 adapter 时该命令会因 required-GPU 错误失败，不会 silent pass。本次真实
adapter 为：

```text
name: Apple M2
device_type: IntegratedGpu
backend: Metal
subgroup_min_size: 4
subgroup_max_size: 64
transient_saves_memory: Some(true)
```

该 adapter 上 `ecs_columns` 结果为：`16 passed, 0 failed, 0 ignored`。实际覆盖
same-tick/backdated writes、dirty-only upload、partial tail、archetype migration、
recreated chunk、growth/shrink/swap-remove、snapshot restore、replacement device、
retained consumer/readback、12-byte compact layout 和 buffer release。

## 本机命令与结果

### App 单元/集成测试

```text
CARGO_TARGET_DIR=/private/tmp/prism-app-acceptance \
CARGO_BUILD_JOBS=1 \
cargo test --locked -p prism_app --all-features --lib
```

结果：`477 passed, 0 failed, 0 ignored`。

### RenderApp 软件/null 路径

```text
CARGO_TARGET_DIR=/private/tmp/prism-render-app-acceptance \
CARGO_BUILD_JOBS=1 \
cargo test --locked -p prism_render_app --all-features --lib
```

结果：`230 passed, 0 failed, 1 ignored`。唯一 ignored 项是需要真实 OS
window/compositor 的 `real_window_presentation_acceptance_entry`。

### RenderApp 强制真实 GPU 门禁

```text
PRISM_REQUIRE_GPU=1 \
CARGO_TARGET_DIR=/private/tmp/prism-render-validation \
CARGO_BUILD_JOBS=1 \
cargo test --locked -p prism_render_app --all-features --lib
```

允许沙盒外访问本机图形设备后，Metal/wgpu 路径实际启动，结果为
`230 passed, 0 failed, 1 ignored`。其中
`recovery::hardware_tests::gpu_recovery_rebuilds_resources_and_restores_pixels`
已验证设备替换、材质 buffer 全容量字节、replay buffer、像素和新 device generation
均恢复。

### WGPU backend 与 GPU-driven

```text
PRISM_REQUIRE_GPU=1 \
CARGO_TARGET_DIR=/private/tmp/prism-render-validation \
CARGO_BUILD_JOBS=1 \
cargo test --locked -p prism_render_driver_wgpu --all-features
```

结果：库测试 `27 passed, 0 failed`，真实 GPU 集成测试 `5 passed, 0 failed`。
adapter inventory 记录为 Apple M2 / IntegratedGpu / Metal；CPU、virtual 和未知
adapter 不会被计入物理 GPU 证据。

```text
PRISM_REQUIRE_GPU=1 \
CARGO_TARGET_DIR=/private/tmp/prism-render-validation \
CARGO_BUILD_JOBS=1 \
cargo test --locked -p prism_render_gpudriven --all-features
```

结果：`18 passed, 0 failed`；GPU culling、indirect draw、multi-draw、图像 parity
和重复提交稳定性均通过。

### Headless runner parity

```text
CARGO_TARGET_DIR=/private/tmp/prism-app-acceptance \
CARGO_BUILD_JOBS=1 \
cargo run -p prism_app --features winit,determinism \
  --example runner_parity -- --headless-only
```

结果：`frames=128 fixed_steps=177 final_hash=e679c6d9b125aadd trace_hash=49b156c395f0513f ... parity=true`。

### CPU pipeline benchmark

```text
CARGO_TARGET_DIR=/private/tmp/prism-app-acceptance \
CARGO_BUILD_JOBS=1 \
cargo bench -p prism_app --bench runtime_pipeline \
  --features multi_thread,pipelined-tasks
```

本次机器样本：macOS/aarch64，8 logical workers，`--locked`，合成 workload。输出验证 `output_parity=true`、`frame_sequence=true`；dedicated/task lane 的相邻帧 overlap ratio 分别为 `0.998054` / `0.995512`，serial 为 `0`。frame P50/P95/P99 分别为 dedicated `1033000/1055416/1072625 ns`、task lane `1031500/1063083/1125917 ns`；schedule scaling 的 P50（1/2/4/8 workers）为 `4089375/3159750/2121667/1460375 ns`。benchmark 默认不启用阈值，显式配置后输出 `performance_thresholds_passed=true`；这些数值仅用于当前机器和当前构造负载的回归参考，不能外推为跨机或生产性能结论。

benchmark 支持通过 `PRISM_PIPELINE_MAX_FRAME_P99_NS` 与
`PRISM_PIPELINE_MIN_OVERLAP_RATIO` 启用显式门禁；本机以 `1500000` ns / `0.90`
运行时 `performance_thresholds_configured=true` 且
`performance_thresholds_passed=true`。这些门槛是本次验收配置，不是跨机器或生产统一标准。

## 尚未闭环的缺口

### GPU 证据边界

GPU 验收测试统一使用 wgpu adapter gate：只有 `IntegratedGpu` 或 `DiscreteGpu`
才会进入“真实硬件 GPU”切片。`Cpu`、`VirtualGpu`、`Other` 会打印 adapter
名称、backend、device type、vendor/device、driver 并标记为
`SOFTWARE/VIRTUAL` 或 unknown，不会被计入 GPU 通过；没有 adapter 时普通运行明确
输出 `SKIPPED ... no adapter`。设置 `PRISM_REQUIRE_GPU=1` 后，缺少 adapter 或只
发现非物理 adapter 都会使测试失败。

null backend 只覆盖 resident columns、dirty upload、device replacement 的状态机和
事务语义；software adapter 只可作为软件执行/生命周期证据。它们都不能证明物理
GPU 执行，也不能推出 GPU 性能收益。wgpu readback 在提交、poll、map callback 成功
后才读取映射范围；map 失败或 callback 缺失会显式失败，不再伪造成功。

1. `real_window_presentation_acceptance_entry` 仍是 `#[ignore]`，需要真实 OS window 和 display compositor；本次没有伪造窗口验收。
2. `runner_parity` 本次只运行 `--headless-only`，没有执行 Winit 窗口分支。
3. 真实 Metal/wgpu 的普通绘制、材质、mesh、transform、GPU-driven 与 device-loss recovery 已在本机门禁通过；仍缺真实移动设备、其他后端和长时间运行证据。
4. benchmark 没有跨机器、长时间、真实游戏场景、present latency、功耗或 GPU 帧时间门槛；不能据此声称“超过 UE”或达到固定 FPS。
5. 全仓测试不作为本记录门禁；`prism_replication` 的 `security.rs` 重复导入问题由 Dalton 处理，本次未修改或绕过它。

## 尚未证明的内容

- 目前只有 Apple M2/Metal 的真实 adapter 证据；没有移动设备、独立 GPU、Vulkan、DX12、
  GLES/WebGPU 浏览器 backend 或其他 vendor 的结果。
- 没有长时间 soak、并发多 queue、跨设备资源共享、真实窗口 present latency、功耗或
  P50/P95/P99 GPU frame-time 数据。上述测试不能支持“高端性能”、UE parity、固定 FPS
  或跨平台吞吐声明。
- bindless fallback 的 capability/ABI 路径已被代码和契约测试覆盖，但本轮没有为每个
  backend 建立独立的真实 shader execution parity 样本；需要时应继续以具体 adapter
  feature set 和 readback 证据补充。

