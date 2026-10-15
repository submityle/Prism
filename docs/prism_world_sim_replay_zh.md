# Prism World 仿真 delta 确定性回放

`prism_world_sim` 生成的 `SimulationDelta` 既是当前 tick 的持久变更，也是
时间轴、FileJournal 和网络重放的输入。codec 与 persist bridge 负责稳定保存和
恢复 bytes；`WorldSim` 现在同时支持严格正向应用与事务式反向回滚，形成可验证的
状态闭环。

## `WorldSim::apply_delta`

应用只接受 `current_tick + 1`，避免重复应用、跳过事务或把旧网络包写入新状态。
函数先验证全部记录，再进行任何写入：

- flow 必须引用现有 hydrology cell、按稳定 source 坐标排序且每个 source 只能出现
  一次；目标邻居、可用水量、四分之一限流、坡度/渗透率限流和目标剩余容量都必须
  与 hydrology 方程一致。多个 flow 只为实际触碰的坐标维护有界调整表。
- voxel 坐标不能重复，material 和 `before_mass` 必须与当前状态一致，且
  `before - after == removed`；`removed` 还必须重新满足 erosion 的水量、硬度和
  每格上限，水量不足不会被静默截断。
- ecology 坐标不能重复，`before_biomass` 必须匹配，`after_biomass` 不能超过
  cell capacity；同时校验 replay 后的水分、stress、邻域拥挤和固定点增长方程，
  不能只伪造一个合法范围内的 biomass。
- fragment 记录必须保持正质量、stable key 唯一，并进入 `WorldSim::fragments`；
  这样正向重放和反向回滚都不会丢失或重复生成碎片状态。

验证失败不会改变 tick、hydrology、voxel、ecology 或 fragments。成功后只更新真实
字段并把 tick 推进到 delta 携带的值，因此从同一初始状态执行一步和应用该步 delta
得到位级相同的稀疏状态。体素 delta 同时保存 hardness，即使质量降为零、体素从
稀疏表移除，回滚也能恢复完整 cell。

## `WorldSim::reverse_delta`

反向操作只接受当前 tick 等于 delta.tick 的记录。它先验证 flow 的逆向水量、体素
的 `after_mass/material/hardness`、生态的 `after_biomass` 和 fragment 的完整
payload，再一次性发布逆向调整；任一失败都不改变 tick、字段或 fragment 表。
成功后恢复所有 `before_*` 状态、移除本 delta 生成的 fragment，并把 tick 回退一格。
反向路径按 delta 实际触碰坐标建立有界调整表，不克隆完整 sparse world。

codec revision 已升至 2，voxel record 从 26 字节扩展为 28 字节以携带 hardness；旧
revision 不会被静默解释为可逆记录。

## 资源边界

回放路径只为 delta 实际触碰的 flow 坐标建立 `BTreeMap` 调整表，生态、体素坐标
分别用有序集合拒绝重复记录；不会为完整 sparse world 做临时 clone。编码后的 delta
仍应先通过 `decode_delta_with_limits` 的字节和记录预算，再交给
`WorldSim::apply_delta` 做状态一致性验证。

## 验收

```text
cargo fmt --manifest-path pkg/prism_world_sim/Cargo.toml -- --check
cargo test --manifest-path pkg/prism_world_sim/Cargo.toml --no-default-features
cargo test --manifest-path pkg/prism_world_sim/Cargo.toml --features persist
cargo clippy --manifest-path pkg/prism_world_sim/Cargo.toml --all-targets --all-features -- -D warnings
```

这不是 GPU stencil、SDF 重烘或产品 renderer 接线；这些仍由各自的 world
边界独立验收。本切片关闭的是仿真 delta 的确定性正向回放与反向回滚缺口，不代表
整个仿真、GPU 水文或产品级时间轴已经完成。
