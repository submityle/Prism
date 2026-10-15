# Prism World 系统实现记录

日期：2026-10-15

本文把 `prism_world_system_design_zh.md` 的设计要求映射到当前仓库中已经存在的实现和验证结果。设计文档是架构输入；本记录只把有源码和测试证据的部分标为完成。

## 已交付

### workspace crate

- `prism_world`：组合三个 policy core 的 `WorldRuntime`，只输出 frame plan，不拥有 scene/ECS/GPU 真相。
- `prism_world_stream`：i64 全局毫米坐标、负坐标 sector 换算、Morton locality hint、cell manifest、需求集、优先级、预算、预取、迟滞驱逐、可回放 VM trace。
- `prism_world_residency`：A/B/C 三档状态、物化预算、确定性 promotion/demotion、HLOD 连续 alpha 与 dither、QoS 滞后调速、稀疏 SDF brick 和确定性 PCG/cache key。
- `prism_world_persist`：PersistentId、规范化 OverrideDelta、append-only log、确定性 LWW CRDT、snapshot/timeline、undo/redo、little-endian codec 和 checksum。

所有新增 world core 都是 `no_std + alloc`、`forbid(unsafe_code)`，不依赖 `bevy_*`，不包含 TODO、占位实现或 `unimplemented!`。

## 验证结果

```text
cargo fmt -p prism_world_stream -p prism_world_residency -p prism_world_persist -p prism_world -- --check
cargo clippy -p prism_world_stream -p prism_world_residency -p prism_world_persist -p prism_world --all-targets -- -D warnings
cargo test -p prism_world_stream -p prism_world_residency -p prism_world_persist -p prism_world
CARGO_TARGET_DIR=/private/tmp/prism-world-target cargo check \
  -p prism_world_stream -p prism_world_residency -p prism_world_persist -p prism_world \
  --no-default-features
```

结果：格式检查通过，Clippy 零告警，四个 crate 共 19 个单元测试通过，隔离 target 的 no-default-features 检查通过。

## 接缝边界

- `WorldRuntime::tick` 只生成 stream/residency policy plan；宿主负责调用 `prism_scene` 的 spawn/despawn/materialize/dematerialize 契约。
- `CellManifest` 只保存 scene key、版本、bounds 和派生 hash；它不复制 scene 内容，也不成为第二真相源。
- `AppendOnlyLog` 是内存确定性核心；文件系统容器、崩溃恢复、压缩、签名和网络传输仍应由上层 provider 实现。
- `SparseSdfBrickGrid`、PCG helper 和 HLOD 是 CPU 可验证 primitive；当前没有把它们冒充成 GPU clipmap、完整 PCG 节点图或真实渲染管线。

## GPU 验证门禁

World core 不需要 GPU。接入 GPU proxy/cull/LOD 时，必须同时记录 adapter 信息、所需 limits/features、shader/pipeline 创建结果和 CPU golden readback。严格模式使用：

```bash
WGPU_BACKEND=metal PRISM_REQUIRE_GPU=1 cargo test -p prism_render_driver_wgpu --test gpu -- --nocapture
```

仅获得 adapter 或默认测试 skip 不构成 world GPU 完成证据；性能报告需要单独记录 CPU/GPU P50/P95/P99、上传字节、显存和 pass 时间。

本次在受控环境执行严格 Metal 门禁的结果是：`PRISM_REQUIRE_GPU=1` 下 wgpu 没有返回 adapter，4 个基础 GPU 测试均按严格模式失败。该结果只说明当前执行环境不能提供 GPU 证据，不把 GPU 后端标记为通过，也不影响上面的 CPU/no-std 核心验证。
