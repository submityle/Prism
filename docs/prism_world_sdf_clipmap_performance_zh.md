# Prism World SDF Toroidal Clipmap CPU 性能基准

## 目的与边界

该基准比较 `prism_world_residency` 中 `CpuSdfClipmap` 与
`SdfToroidalClipmap` 的 CPU `update_center` 更新耗时。它覆盖负坐标、多 mip
小步滚动和彼此分离窗口的大跳跃，并逐样本断言两个实现产生相同数量的进入与离开逻辑坐标。

基准使用 Rust 的 `cargo bench` 默认优化 profile；不使用 debug profile。计时只包围
一次 `update_center` 调用，planner 首次构造和固定预热不计入样本。结果是本机进程的
CPU 墙钟观测值，可能受调度、温度、功耗和其他主机负载影响；**不是 GPU 执行时间、产品
帧时间或跨机器性能承诺**。该基准不创建 GPU 资源，不测 SDF 烘焙、上传、采样、渲染、内存
驻留或端到端工作流。

## 复现

在仓库根目录执行：

```sh
cargo bench --offline -p prism_world_residency --bench sdf_clipmap
```

无第三方 benchmark 依赖。每个 planner、每个场景固定预热 128 次并采集 1,024 个更新样本。
P50/P95/P99 使用 nearest-rank 口径：排序后取 `ceil(N × p) - 1` 下标。时间百分位按耗时排序；
delta 百分位独立按 delta 个数排序。另报告各时间百分位样本所对应的 delta 数。delta 定义为
各 mip 新进入和新离开坐标数之和；这是逻辑工作量，不等于唯一物理槽位数或实际 GPU 上传字节。

## 固定配置与 workload

| 项 | 固定值 |
| --- | ---: |
| mip levels | 6 |
| 每 mip ring radius | 24 bricks |
| 每 mip footprint | 49 × 49 = 2,401 cells |
| 全部 mip layout cells | 14,406 |
| `max_update_bytes` | 8,388,608 bytes (8 MiB) |
| 预热 / 测量 | 128 / 1,024 次更新，每 planner、每场景 |
| 计时范围 | 单次 `update_center`，不含初始构造 |

配置预算对 sorted planner 和 toroidal planner 都设为 8 MiB；该值是两种 API 各自的逻辑
规划预算，不代表真实分配峰值或等价内存上限。两者均成功完成所有样本。

- `small_roll_negative_multimip`：从 `(-8192, -17, -4096, mip 0)` 开始；第 `n` 步中心为
  `(-8192 + n, -17, -4096 + floor(n / 4), mip 0)`。测量前先执行 128 步，所有坐标保持负值；
  X 每步移动，Z 每四步移动，使各 mip 的 floor 缩放边界按确定序列变化。
- `large_jump_disjoint_windows`：第 `n` 步中心在
  `(-1,000,000, 113, 750,000, mip 0)` 与
  `(1,000,000, 113, -750,000, mip 0)` 间交替。窗口不相交，每次更新都需处理完整进入和离开集合。

两类 planner 在同一 workload 的相同中心序列上独立运行。benchmark 对 1,024 对样本逐一
断言 entering+exiting 坐标数量一致；断言通过只能证明此 workload 下的逻辑 delta 数匹配，
并不替代 clipmap 正确性测试对坐标集合内容的验证。

## 实测结果

采集日期：2026-10-21。环境：macOS Darwin 27.0.0，Apple Silicon ARM64（`arm64`），
`rustc 1.96.1 (31fca3adb 2026-06-26)`；使用 `cargo bench` 优化 profile。结果单位为
微秒/次更新。

| 场景 | Planner | P50 | P95 | P99 | delta 坐标数 min / P50 / P95 / P99 / max |
| --- | --- | ---: | ---: | ---: | --- |
| 小步负坐标、多 mip | `CpuSdfClipmap` | 2279.875 | 5075.500 | 11448.750 | 98 / 98 / 778 / 1068 / 1164 |
| 小步负坐标、多 mip | `SdfToroidalClipmap` | 0.416 | 1.125 | 1.625 | 98 / 98 / 778 / 1068 / 1164 |
| 不相交窗口大跳跃 | `CpuSdfClipmap` | 1074.584 | 2430.167 | 5008.042 | 28812 / 28812 / 28812 / 28812 / 28812 |
| 不相交窗口大跳跃 | `SdfToroidalClipmap` | 22.458 | 58.000 | 99.958 | 28812 / 28812 / 28812 / 28812 / 28812 |

逐样本进入/离开 delta 数量相等断言通过。小步更新的样本包含不同 mip 边界跨越次数，因此
delta 分布不是常数。大跳跃每次更新的 28,812 个坐标是 6 mip × 2,401 cells × 进入/离开
两侧的完整逻辑集合。

上述倍率只适用于这次单机微基准与这组参数。小步 workload 中 toroidal planner 的边界枚举
随变化边缘和输出 delta 增长；大跳跃仍必须枚举完整输出集合。测量没有隔离主机调度和热状态，
尤其 P99 不应被解释成稳定上界。结果也不说明消费者实际复用了槽位、及时回收 SDF 数据或已
接通 runtime/GPU 路径。

## 验证记录

- `cargo fmt --package prism_world_residency -- --check`：通过。
- `git diff --check`：本次 GPU toroidal 实现、residency planner、benchmark 和文档改动通过。
- `CARGO_TARGET_DIR=/private/tmp/prism-world-sdf-toroidal-20261021 cargo bench -j 1 -p prism_world_residency --bench sdf_clipmap --offline`：release benchmark 完成，逐样本 delta parity 断言通过；最新观测数据已替换上表旧样本。
- 未运行 GPU benchmark、运行时集成测试或产品场景帧时间测试；它们不在该 CPU harness 的测量范围内。
