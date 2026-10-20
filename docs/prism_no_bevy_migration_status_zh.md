# Prism 无 Bevy 迁移状态

截至 2026-10-20，Prism 自有 `pkg/prism_*` 依赖闭包已完成以下迁移：

- `prism_math` 删除 `compat_bevy` 与旧兼容测试，提供 `prism_math::ops`。
- 13 个音频 crate 的数学依赖改为 `prism_math`，并保留 `std`、`no_std`、`serialize` 语义。
- `prism_render_material` 删除 Bevy bridge，改用 Prism 数学与材质契约。
- `prism_render_visibility` 删除 Bevy bridge，视图/场景数据只接受 Prism ABI。
- `prism_physics_gpu` 的 CPU 参考关节约束改用 `prism_math::ops`。
- `prism_utils`、`prism_tasks`、`prism_time`、`prism_transform`、`prism_reflect` 删除未被消费的 `compat-bevy` feature、兼容模块、导出与专属测试；保留原生容器、线程池、时钟、变换和反射契约。五份 design docs 已改用原生 API 迁移说明，不再承诺只替换 import 即可接入。

当前唯一仍直接声明 `bevy_*` 依赖的 Prism crate 是 `prism_render_scene`。它正在进行渲染场景运行时迁移，完成前不能宣称 Prism workspace 已经无 Bevy。仓库内 `crates/bevy_*` 是尚未删除的旧上游源码树，不能作为 Prism 新 API 的依赖入口。

## 验证

此前 `cargo metadata --locked --no-deps --format-version 1` 的依赖扫描结果只有 `prism_render_scene`。本次对 `pkg/**/Cargo.toml` 直接声明的 `bevy_*` 依赖做静态复核，仍只命中该包；未将静态扫描当作完整 workspace 编译验收。

基础兼容层清理前，全仓 Rust 源码与 Cargo manifest 搜索未发现五个兼容层的外部消费者；清理后再次搜索 `compat-bevy`/`compat_bevy`、`bevy_prelude`、`CompatTaskPool`/`CompatScope` 与 `BevyGlobalTransformExt`，无残留声明或引用。

2026-10-20 基础包隔离验证记录（macOS，Rust/Cargo 1.96.1）：临时 workspace 位于 `/private/tmp/prism-foundation-compat.qFtoL5`，仅包含五包及其本地依赖闭包共 11 包；复制当前 manifest、沿用 workspace lint，源码目录链接到本仓库。使用独立 lockfile 和 target，未修改仓库 workspace 或 lockfile。两条命令均从该临时 workspace 运行，退出码均为 0：

```sh
export CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0
export CARGO_BUILD_JOBS=2
export CARGO_TARGET_DIR=/private/tmp/prism-foundation-compat.qFtoL5/target
cargo test --offline -p prism_utils -p prism_tasks -p prism_time -p prism_transform -p prism_reflect --lib -- --test-threads=2
cargo check --offline -p prism_utils -p prism_tasks -p prism_time -p prism_transform -p prism_reflect --no-default-features --lib
```

默认 lib 测试共 **1077 项通过**：`prism_utils` 177、`prism_tasks` 292、`prism_time` 222、`prism_transform` 179、`prism_reflect` 207。第二条命令是五包联合检查，依赖的 feature 可能统一并重新启用默认 feature，不能视为逐包隔离的 `no_std` 证明。改动 Rust 文件的 `rustfmt --check` 与限定写集的 `git diff --check` 均通过；未运行 all-features、集成测试、doctest、bench、clippy 或完整 workspace 验收，不推导其他消费方已通过验收。

验证结束后仅清理本任务 target：`cargo clean --manifest-path /private/tmp/prism-foundation-compat.qFtoL5/Cargo.toml --target-dir /private/tmp/prism-foundation-compat.qFtoL5/target`，报告移除 230 个文件、69.7 MiB，随后确认 target 目录不存在。临时 workspace 的小型 manifest、lockfile 和源码链接保留供复核。

渲染可见性迁移此前实际新跑的是默认 feature 下 **124 项 lib 测试**，不能扩写为三档 feature 检查或无分配性能验证；本次仅纠正文档，未重跑该包。GPU 物理数学迁移此前只确认独立 target 的 `cargo check --offline -p prism_physics_gpu --lib` 通过，**CPU/WGSL parity、真实 GPU 执行及跨后端确定性均未验证**。两者详见各自迁移文档。

完整 workspace 验收仍须等待 `prism_render_scene` 迁移完成后执行。
