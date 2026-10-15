# Prism World 仿真 delta 确定性回放

`prism_world_sim` 生成的 `SimulationDelta` 既是当前 tick 的持久变更，也是
时间轴、FileJournal 和网络重放的输入。此前 codec 与 persist bridge 能保存和
恢复 bytes，但没有把 delta 安全地应用回权威 `WorldSim`；本切片补上这一条
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
- fragment 记录必须保持正质量；fragment 是外部派生事件，不被错误地伪造为
  `WorldSim` 内部第二份存储。

验证失败不会改变 tick、hydrology、voxel 或 ecology。成功后只更新真实字段并把
tick 推进到 delta 携带的值，因此从同一初始状态执行一步和应用该步 delta 得到
位级相同的稀疏状态。

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
边界独立验收。本切片只关闭仿真 delta 的确定性状态回放缺口。
