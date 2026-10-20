# Prism World SDF Physics Runtime

本文记录 World sparse SDF 到 physics 查询消费者的运行时边界。它是派生数据接缝，不改变 `prism_scene` 作者真相，也不宣称已经把 SDF 接入刚体 solver、渲染或 GI。

## 已交付

- `prism_world_physics_bridge` 为 sparse brick 增加显式 `origin_mm`、`voxel_size_mm`、`distance_scale_mm` 与 `revision`，拒绝零尺度。
- `WorldSdfPhysicsBridge::stage` 在发布前检查 brick、sample 和逻辑字节预算；`publish` 对版本保持单调，失败保留旧快照。
- `prism_world_physics_cell_runtime::SdfPhysicsCellHost` 将解码、每 cell staging、聚合驻留预算和 `SceneHost` 生命周期接成一个提交点；reload、decode、revision、预算或 host 失败都不替换旧快照。
- 查询入口按 cell 返回 `Value`、`Miss` 或 `Unknown`。未驻留 cell 和缺失 brick 都是 `Unknown`，不会被解释为空场。
- 物理查询快照保留量化 grid 与世界尺度，逻辑 footprint 计为 48-byte header、16-byte brick metadata 和 2-byte sample；这不是 allocator RSS 或 GPU 显存峰值。

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

## 验证边界

应至少运行：

```text
cargo test -p prism_world_physics_bridge --lib --offline
cargo test -p prism_world_physics_cell_runtime --lib --offline
cargo test -p prism_world_physics_bridge --test sdf_sampling --offline
cargo test -p prism_world_physics_cell_runtime --test sdf_sampling_lifecycle --offline
cargo bench -p prism_world_physics_bridge --bench sdf_sampling --offline
```

这些测试证明 cell 生命周期到 physics 查询的原子接入，不证明 `prism_physics_core` 已使用该场生成 SDF 接触，也不证明 `prism_render_scene`、GI 或 GPU renderer graph 已消费同一份 World SDF。后续消费者必须复用 `WorldSdfSnapshot` 的尺度、revision 和 `Unknown` 语义，不能另建未版本化的平行场。

### 2026-10-21 验证记录

独立 target、offline、单编译任务、dev/test debug=0、incremental=0 下，bridge 库 11 项、连续采样 10 项、cell runtime 库 22 项和生命周期集成 1 项全部通过（共 44 项，无忽略）。连续采样包含最小正 subnormal 与最大有限梯度的归一化回归。另行回归 `prism_world_runtime --lib` 108 项与 `prism_world_gpu --features gpu --lib` 123 项全部通过，历史 `LayeredCellManifest::admitted_layers` 编译错误未复现。依赖 `prism_physics_core` 仍报告既有警告，本记录不宣称全 workspace 无警告。

改动 Rust 文件的定向 rustfmt 与 diff whitespace 检查通过。`cargo clippy -p prism_world_physics_bridge -p prism_world_physics_cell_runtime --all-targets --no-deps --offline -- -D warnings` 未通过：bridge 原有代码报告 1 处 `std_instead_of_alloc` 和 3 处 `collapsible_if`，在 cell 检查前终止；没有为本片改动无关代码或放宽 lint。

Apple M2、默认优化 bench profile、bench debug=0、incremental=0，预先逐位置验证 4096 个确定性位置的解析距离/梯度与缺失语义，再运行 128 批、每批 4096 次查询。下表分位数是各批平均 ns/query 的分布，不是单次查询尾延迟；没有独立预热批。计时包括查询及成功结果检查，不包括建图、位置生成、正确性 oracle、输出或销毁。

| 负载 | P50 / P95 / P99（ns/query） | 快照逻辑字节 |
|---|---|---|
| 64 个 edge=8 brick、32768 个样本、混合内部/边界位置 | 172.729 / 183.268 / 196.371 | 66608 |
| 空 grid、首角点缺失即返回 Unknown | 31.820 / 34.587 / 56.773 | 48 |

这是 CPU 查询微基准，不是 GPU timestamp、RSS、实际显存、产品帧时或多消费者负载验收；空 grid 的快速失败也不代表最后一个角点缺失的成本。
