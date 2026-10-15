# World 作者密度与种类场

日期：2026-10-16

## 所属层与确定性契约

`prism_world_residency` 的 `ScatterDensityField` 是 scene 规则派生的、可丢弃重建的不可变描述，不是编辑器文档、第二份 scene 真相、持久对象身份或另一套随机数状态。调用者提供 `ScatterFieldConfig` 与带稳定 ID 的 `DensityBrush`；构造器验证所有输入（包括不生效的刷子/噪声参数），按唯一 brush ID 升序原地排序。相同笔刷集合的输入 Vec 顺序不影响结果。种类是项目定义的 `u32` 资产表索引，没有植被/角色等内建分类。

全域位置使用有符号 `i64` 毫米，噪声/刷子/选取只用整数。算法版本 `SCATTER_FIELD_VERSION=1` 与调用者 `config.version` 分开；改变任何生成位模式或取整规则必须提升算法版本。以下规则覆盖负坐标、超过 f64 精确整数范围的坐标以及 i64 两端：

- 单 band 三维 value noise，每格 8 个角点，PCG 派生 UNORM16 值；Euclidean 除法确定格坐标和非负余数。格内权重先向下量化到 0..65535，再做整数 smoothstep，按 Z→Y→X 固定顺序三线性插值。
- `ONE=65535`。lerp 用非负加权和 `(a*(ONE-w)+b*w+ONE/2)/ONE`；smoothstep 用 `(t²*(3*ONE-2*t)+ONE²/2)/ONE²`，均为 nearest-half-up，不经浮点中间值。
- 可选 domain warp 使用另三条独立噪声通道。各轴位移为 `(2*noise-ONE)*warp_amplitude_mm/ONE`，有符号除法向零取整。先在 `i128` 中将位移加到原位置，再查密度噪声；不在世界边缘夹坐标或将位置 wrapping 回世界。虚拟格坐标取低 64 位进入 hash，明确定义为每轴 2^64 格的哈希周期。
- 密度从 `base_density` 加上以 `amplitude` 为幅度的有符号噪声，再夹到 UNORM16。`amplitude=0` 跳过噪声和 warp，但输入仍参与配置比较且 cell size 仍须非零。

这里没有将既有 `prism_math` 浮点 Perlin/Simplex 噪声的输出冒充整数全域确定性结果；作者场使用独立版本化整数 kernel。它是单 band value noise，不承诺分形/蓝噪声频谱、地形投影或艺术效果验收。

## 刷子合成

球形刷子含整数中心、内外半径、UNORM16 opacity、密度操作及可选 species。外半径须正，内半径不超过外半径；精确外边界权重为零。内半径内为满径向权重；两半径相等时为硬刷子。中间区域对 `(outer²-distance²)/(outer²-inner²)` 的 UNORM16 向下量化值做 smoothstep，再以 nearest-half-up 乘 opacity。该衰减按**平方距离**定义，不是线性距离/sqrt 衰减。

距离差先扩宽到 i128，单轴平方用 u128。每轴平方与剩余外半径平方先比较、再累积，不允许世界两端三轴平方求和溢出。opacity 为零可提前返回，但构造时仍验证其半径/操作。

按 brush ID 顺序应用：`Replace(v)` 用有效权重 lerp；`Add(delta)` 要求 delta 在 ±65535 内，带符号加权除法向零取整，再饱和到密度范围。相同 ID 拒绝，不能依赖 Vec 的偶然顺序。种类是离散值：有效权重至少 32768 时替换 species；用 `Add(0)` 可实现只刷种类。种类边界是明确的硬阈值，不声称已实现概率混合、多种资产比例或连续材质过渡。

## 候选选取与缓存

`candidate_rank` 从 field seed 及 `ScatterCandidateKey` 的全部结构字段派生独立 u32 rank，与请求范围、dense/source slot、刷子顺序、密度值无关。密度 0 全拒绝，65535 全接受；其他值采用 `rank < density*65537`。固定 candidate key 与 field seed 时，增大密度不会删除已选成员。rank 仅决定分布，不是无碰撞身份。噪声 seed 同时也是 rank namespace；换 seed 不承诺集合包含关系。

`scatter_with_density_field` 先调用现有 scatter，再按全局位置采样并过滤源幸存者，保留完整 key、原位置、词典序及对齐的 density/species 列。它是源集合的子集，保持已有最小距离及任意半开范围拆分/合并/逆序求值不变性。不重新填补密度拒绝的点，也不重新判断这些点是否阻挡过原始候选。

`FieldScatterCacheKey` 包含原完整 `ScatterCacheKey`、field 算法版本、完整不可变 field 描述；比较配置与每个笔刷的值，不使用 digest 或 Arc 指针当身份。全部 inactive 输入也参与失效。field clone 为 O(1) Arc 共享；独立描述之间的相等比较最坏 O(B)，不能宣称缓存比较始终 O(1)。派生结果可以随时销毁重建；外部地表、资产映射或 override 若影响消费者，仍须由消费者维护相应完整失效描述。

## 预算和热路径

构造预算分别限制 brush 数与调用者移交 Vec 的 `capacity*size_of::<DensityBrush>()`，包括 spare capacity。排序原地进行，不复制第二个 brush buffer；Arc 元数据采用普通分配/OOM 处理，不属于可恢复预算错误。

线性路径每个 field sample 的逻辑 work 为 `1 + B + 8*noise_enabled + 24*warp_enabled`：1 表示选取 rank，B 包括全部刷子；噪声 amplitude 为零时不计 noise/warp。它是保守逻辑探针数，不是 CPU 指令数或毫秒上限。该路径逐点遍历 B 个刷子，采样为 O(B+噪声固定成本)。另有显式 `ScatterBrushIndex` CPU BVH 路径，复用本文整数 kernel，先空间裁剪再按稳定 ID 合成，使用独立索引/scratch 预算与节点/刷子 work；契约与稀疏/重叠对照见 `prism_world_scatter_brush_index_zh.md`。两条路径均不承诺百万笔刷实时编辑。

field scatter 在枚举前检查 `primary_candidates*work_per_sample` 的 checked 乘法与显式工作预算。源 scatter 自己的 work/count/byte 限额仍在密度筛选前生效，低密度不能绕过源预算。返回的 `field_work` 为实际源幸存者数量乘单样本逻辑 work，可能低于预检上界。

源 point Vec 原地 retain，不分配副本；属性 Vec 一次 fallible exact reservation，按全部源幸存者预留，之后只在预留容量内 push。总输出预算同时计算 points 与 attributes 的实际容量载荷，预留前后各检查一次；失败整体返回错误，不交付截断前缀。筛选后不 shrink，空结果也可能保留容量。

该总输出限额**不是峰值分配上限**：源 scatter 会先按自身预算分配 point buffer，再进行合计检查；分配器超额给出的属性容量只能在预留后检查并拒绝。Vec/Arc header、allocator 元数据/对齐、栈、调用者持有的其他缓存及 RSS 不计入。每次独立 `sample` / `candidate_rank` / `accepts` 不分配、不修改状态、不保存隐式逐点缓存。

## 复现入口与限定性能

```sh
cargo test --offline -p prism_world_residency --no-default-features
cargo clippy --offline -p prism_world_residency --all-targets -- -D warnings
cargo bench --offline -p prism_world_residency --bench scatter_field
```

本轮 no-default 完整回归为 65 项单测与 1 项文档测试通过，0 失败/忽略；其中作者场新增 30 项单测，包含独立宽整数噪声/刷子 oracle、固定 golden、全部 UNORM16 权重与密度的穷举、跨 i64 边缘 warp、每个 inactive 缓存字段、预算失败和三轴任意分区双向求值。all-target clippy `-D warnings` 与 owned 文件格式检查通过。独立复审未发现阻塞问题；这些验证不替代下述尚未接入的产品链路。

bench 对 10,000 个真实 source scatter 三维 jitter 点，执行 8 次预热、64 次测量，nearest-rank P50/P95/P99。样本循环含 field sample、rank 选取和结果 digest；全链路另测源 scatter、原地过滤、属性写入，包含输出分配、不含输出 drop。配方/笔刷构造在计时外。所有迭代校验 accepted/digest 或完整输出统计一致，分配计数器验证独立采样为零分配、全链路恰为两个 Vec 分配。

2026-10-15 Apple M2/macOS release，其他并行任务运行，尾部存在明显干扰；这些是 CPU 主机墙钟，不是 GPU 时间、画面质量、产品帧时或严格优化前后 A/B 证据：

| 10k 任务 | accepted | 逻辑 work/sample | P50/P95/P99 µs | 计时内分配 |
|---|---:|---:|---|---:|
| constant，sample+rank | 4,990 | 1 | 200.459 / 217.209 / 234.625 | 0 |
| noise，sample+rank | 5,109 | 9 | 782.250 / 800.750 / 818.834 | 0 |
| warp，sample+rank | 5,124 | 33 | 2685.917 / 2823.125 / 3189.875 | 0 |
| warp+32 个重叠刷子，sample+rank | 5,124 | 65 | 5392.875 / 32179.334 / 55792.958 | 0 |
| scatter+warp+32 个重叠刷子全链路 | 5,124 | 65 | 5878.709 / 12008.041 / 17783.000 | 2 |

全链路返回 Vec 容量载荷 880,000 字节，而非仅按 5,124 个保留点计费。噪声角点共用 hash 前缀已在循环外复用，每 band 从重复的 24 次 mix 降到 14 次，无额外缓存；优化前后 benchmark digest 与接受数量一致，不能将受干扰的两次墙钟差直接宣传为加速比。

## 尚未扩大的边界

本切片是完整可调用的 CPU 作者场及 source scatter 过滤。CPU 笔刷空间索引另见 `prism_world_scatter_brush_index_zh.md`。GPU twin 与固定容量笔刷上传另由 `GpuScatterField` 实现，精确 ABI、五 storage binding 要求及设备验证见 `prism_world_gpu_scatter_field_zh.md`；可选 canonical 区间树见 `prism_world_gpu_scatter_brush_index_zh.md`。普通 `GpuScatter` 仍只生成源候选。已有 resident coverage/dither 是另一层显示策略，真实 heightfield 精确高度/离地带/坡度稳定过滤另见 `prism_world_heightfield_scatter_zh.md`。编辑器工具、scene schema、曲率/数据层/跨层/SDF 过滤、朝向生成、风/交互、蓝噪声目标与产品 renderer graph 仍需各自实现和验收，不能由本场的局部测试替代。
