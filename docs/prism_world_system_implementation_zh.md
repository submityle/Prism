# Prism World 系统实现记录

日期：2026-10-15

本文只把已经存在源码、测试和提交记录的部分标为已交付；设计文档中的路线图、指标和效果目标仍然不能替代实现证据。

## 已交付切片

### M0-M2：空间、流送和常驻

- `prism_world_stream`：`i64` 毫米坐标、负坐标 `div_euclid/rem_euclid`、sector/Morton locality、cell manifest、需求集、预测预取、优先级、预算、迟滞驱逐和可回放 frame trace。
- `prism_world_residency`：A/B/C 三档状态、物化预算、确定性 promotion/demotion、HLOD 连续 alpha/dither、QoS governor、稀疏 SDF brick 和 PCG 图求值/脏传播/cache key。
- `prism_world`：组合 policy core 的 `WorldRuntime`，并提供 feature-gated `SceneEcsHost`；cell 加载通过真实 `prism_scene::spawn_typed_plan`/`SceneSpawn::attach_instance`，卸载通过 `prism_scene::despawn_instance`，驻留升降级通过 typed provider 回调，不直接写第二份 scene 真相。

### M4：GPU proxy visibility contract

- `prism_world_gpu` 已纳入根 workspace。
- CPU contract、SoA little-endian upload layout、WGSL compute shader、readback 四字记录和 CPU golden 对拍已具备。
- WGSL parse/validation、CPU tests 和 `cargo check --features gpu` 已通过。
- 严格门禁命令：

```text
WGPU_BACKEND=metal PRISM_REQUIRE_GPU=1 cargo test -p prism_world_gpu --features gpu --test gpu_parity -- --nocapture
```

本 checkout 的受控环境结果为失败：`No suitable graphics adapter found`，其中 Metal 没有返回 adapter。该结果是环境限制，不构成 GPU 通过证据；默认无 GPU 测试只允许显式非严格模式跳过。

### M5：持久化

- `prism_world_persist`：`PersistentId`、规范化 `OverrideDelta`、append-only log、确定性 LWW CRDT、snapshot/timeline、undo/redo、little-endian codec/checksum。
- `std` feature 提供 `FileJournal` 和 `SnapshotStore`：版本头、长度/序号/payload/checksum、批量 append、`flush + sync_all`、重启恢复、损坏/截断尾部裁剪、乱序/重复拒绝、同目录临时文件原子替换和父目录同步。
- 相关提交：`4d0fbfebe`；Scene/ECS 接缝提交：`34656b305`；GPU workspace 集成：`c94d5cf13`。

## 当前验证

通过：

```text
cargo fmt -p prism_world_stream -p prism_world_residency -p prism_world_persist -p prism_world -- --check
cargo test -p prism_world_persist --features std
cargo test -p prism_world --features scene_host
cargo test -p prism_world_gpu
cargo test -p prism_world_gpu --features gpu --test wgsl_compile
cargo check -p prism_world_gpu --features gpu
CARGO_TARGET_DIR=/private/tmp/prism-world-target cargo check -p prism_world_stream -p prism_world_residency -p prism_world_persist -p prism_world --no-default-features
```

核心 world policy 与 durable/scene tests 当前共 12 项持久化测试、5 项 `prism_world` 测试、4 项 GPU CPU contract 测试和 1 项 WGSL validator 测试通过。

`cargo clippy -D warnings` 在包含 `prism_world` 依赖闭包时会进入仓库其他并行改动的 `prism_asset`/`prism_ecs`，当前被这些既有告警阻断；world 自身新增代码已按相应 crate 的局部检查验证，不能把依赖闭包失败误报为全 workspace 通过。

## 尚未交付的边界

以下项目仍在实现或需要独立硬件/端到端证据，不能用 policy primitive 冒充完成：

- M3：真正与 world stream 连通的异步 IO/provider、取消和背压闭环。
- M6：terrain 多表示、cube-sphere 行星 tile、增量重烘端到端接入。
- M7：侵蚀/破坏/生态/水文的 world 调度和 SDF 一场多用接入。
- M8：多人 AOI/authority/sector handoff 网络消费层及轨道到地表端到端验证。
- GPU proxy 的真实 adapter、dispatch、readback 和 P50/P95/P99 性能数据仍需在有 GPU 的环境执行。

这些边界不是占位 API；每一项必须以可运行算法、明确契约、单测和真实调用路径交付后才从本节移除。

## 不变量

- world 不复制 scene 内容；可变权威状态是 override delta，派生索引/代理/SDF/PCG 结果可删除重建。
- policy core 保持 `no_std + alloc`、`forbid(unsafe_code)`；文件系统和 GPU 能力只在明确 feature/provider 边界出现。
- 所有日志、排序、PCG、HLOD 和 readback contract 都有稳定顺序或稳定 key。
- 任何声称 GPU 或性能完成的结论必须同时记录 adapter、pipeline、dispatch/readback、P50/P95/P99、上传字节和显存/资源预算。
