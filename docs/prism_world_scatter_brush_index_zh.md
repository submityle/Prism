# World CPU 作者刷子空间索引

日期：2026-10-16

## 权威与用途

`prism_world_residency::ScatterBrushIndex` 是 `ScatterDensityField` 的可重建 CPU 查询加速结构。它共享完整不可变场配方，不引入第二份 scene 真相、持久对象身份、编辑器状态或随机状态。原 `ScatterDensityField::sample` 保留为线性基线；调用者显式选择索引路径，GPU 作者场仍使用自己的线性笔刷扫描。

索引不改变 field 算法版本、candidate rank、密度选取或输出缓存身份。完整 field 配方（包括零 opacity、未覆盖查询位置、关闭噪声等 inactive 输入）仍由 `FieldScatterCacheKey` 保留并按值比较。索引拓扑、分配地址和 scratch 容量不进入语义缓存键。

## 查询与合成顺序

构造器对中心坐标跨度最大的轴作中位数空间划分，中心相同时以 canonical brush index 打破平局，建立确定性 BVH。叶子对应原场中按唯一稳定 ID 排序的刷子；连续前序节点与 escape index 支持无递归、无查询栈的包围盒裁剪。

包围盒通过扩宽整数计算 `center ± outer_radius`，再夹到 i64 世界域，不能以 wrapping 或浮点转换丢失世界角点覆盖。包围盒只是保守候选测试：最终球形距离、平方距离 smoothstep、UNORM16 opacity、Add 的向零取整、Replace 的 nearest-half-up 与 species 阈值仍调用原作者场的同一整数 kernel。包围盒外边界的假阳性允许进入精确球形测试，不会改变外球面权重为零的规则。

遍历收集 canonical brush index，原地 `sort_unstable` 升序后合成。Replace/Add/离散 species 的操作不交换；直接按 BVH 遍历顺序合成会改变结果，因此排序是必要工作，而非可省略的实现细节。查询 scratch 是调用者显式持有的固定容量 Vec，不保留跨点缓存；不同索引之间可复用足够大的 scratch。

查询成本为 `O(visited_nodes + K log K + K + fixed_noise)`，K 是通过叶子 AABB 的刷子数。稀疏分布可以有效裁剪；全部重叠时访问整棵树并排序，可能比线性路径慢。该接口不承诺任意配方加速、百万刷子实时编辑或严格 CPU 时间上限。

## 显式资源与工作预算

`ScatterBrushIndexBudget` 分别限制 brush 数和索引拥有的 Vec 容量载荷。`allocated_bytes()` 报告实际索引容量载荷，共享 field 自身刷子载荷另计。`new_scratch(max_scratch_bytes)` 独立预留足以容纳全部 B 个 canonical index 的 scratch，并检查实际容量，而不是按当前某一次查询的 K 估算。

构造使用 checked 数量/字节运算、fallible reservation、预留前及实际容量检查；采样阶段不隐式增长 scratch、不分配、不改写 field。Vec header、allocator 元数据/对齐、构造/查询栈、共享 field 的 Arc 元数据与物理 RSS 不属于这些载荷限额。scratch 可在错误后复用，但其内部临时内容不保证保持原值。

`IndexedFieldSample` 返回完整 `sample` 及实际 `node_tests`、`brush_tests`、`field_work`。逻辑 work 包含原场的基础 rank/noise/warp 单位，加实际节点及精确刷子测试；`max_sample_work()` 保守覆盖基础 work、全部节点与全部 B 个刷子。排序比较、AABB 内部轴比较、整数乘除和 CPU 指令没有逐条计费；不能将该逻辑预算宣传成毫秒或指令硬限额。

独立 `sample` 按实际访问增量扣费：先检查完整 scratch 容量，再在基础采样、各节点和各精确刷子执行前收费。小于最坏上界的预算仍可完成稀疏查询；不足时不返回半份 sample，scratch 可以改变，后续仍可复用。完整 scatter 接口的枚举前最坏准入是另一层更保守的契约。

`scatter_with_indexed_density_field` 在源候选枚举前检查 `primary_candidates × max_sample_work()`，并保留源 scatter 自身的候选、数量、字节限额。低密度或空间裁剪不能绕过源/最坏场工作准入；实际 `field_work` 统计源幸存者的真实索引逻辑 work，通常不等于线性路径的 `field_work`。

## 完整散布输出

索引完整散布仍先生成源，再稳定过滤源幸存者；不重新填洞、不移动位置、不重新解释已拒绝点的阻挡作用。因此保持完整 candidate key、词典序、最小距离、半开分区合并与请求顺序不变性。

源 points Vec 原地压紧，attributes 一次预留源幸存者容量；返回两列严格对齐。两 Vec 的实际容量继续共同计入 `FieldScatterBudget.max_output_bytes`，不因筛后点少而少计 spare capacity。错误整体拒绝请求，不返回预算相关的截断前缀。与原线性接口相同，总输出载荷限制不是构造过程的峰值 RSS 上限。

## 复现入口

```sh
cargo test --offline --locked -p prism_world_residency --no-default-features
cargo clippy --offline --locked -p prism_world_residency --all-targets -- -D warnings
cargo bench --offline --locked -p prism_world_residency --bench scatter_brush_index
```

benchmark 对 B=32/1024/10000 各测稀疏和全部重叠配方，启用 noise/warp 和非交换操作。前两档每批1024查询，10000刷子档每批256查询；8预热、64测量，交替线性/索引测量顺序。每次计时外逐点比较完整 density/species，计时内 digest 防止消除，分配计数器要求两条查询热路径都为零分配。构造时间单独报告，不含 field 构造；scratch 创建、oracle、输出校验、drop 不计入查询时间。

另测 B=1024 的稀疏完整散布：10,000 primary、spacing=100 mm、min_distance=0、区域 `[0,0,0]..[1000000,100,100]`，相同 8 预热/64 测量与交替顺序。每对计时外核对全部 points、attributes、完整 cache key 与 source candidate counter；计时内包含源生成、场过滤与输出分配，断言每次恰为两个分配。drop、oracle、索引和 scratch 构造不计入完整散布时间。

这些是 CPU 查询批次或明确列出的完整 scatter 墙钟；查询批次没有包含 rank 筛选。两类测量都没有包含 GPU、渲染或编辑器帧，不代表多厂商设备、真实百万实例产品场景或画面质量验收。

## 本机 release 实测

本次 no-default 完整回归通过 99 项库测试、16 项公开 API 集成测试、1 项文档测试，0 失败/忽略；all-target strict Clippy（`-D warnings`）、限定文件 rustfmt 与 diff 检查通过。8 项索引白盒测试检查树结构/轴平局/escape/容量/增量预算；16 项公开测试独立核对 baseline、全域边缘、非交换顺序、缓存依赖、分区与整链预算。并行只读源码审查未发现具体阻断项；性能证据仅指下述独立运行。

2026-10-16，Apple M2/macOS、rustc 1.96.1，仓库 bench profile。下表单位 µs，P50/P95/P99 采用 64 样本的 nearest-rank；宿主有其他并行任务，尾延迟明显受干扰，不将这些结果解释为隔离实验或严格产品时限。全部逐点 oracle 与零分配断言通过。

| 刷子数 / 分布 | 每批查询 | 线性 P50 / P95 / P99 | 索引 P50 / P95 / P99 |
| --- | ---: | --- | --- |
| 32 / 稀疏 | 1024 | 366.166 / 815.250 / 1086.708 | 315.958 / 696.125 / 847.833 |
| 32 / 全重叠 | 1024 | 439.333 / 1085.375 / 1133.125 | 610.250 / 1399.125 / 1959.958 |
| 1024 / 稀疏 | 1024 | 2499.000 / 10102.708 / 31004.625 | 347.625 / 881.208 / 2288.625 |
| 1024 / 全重叠 | 1024 | 5342.167 / 12326.583 / 54745.792 | 10483.166 / 25146.125 / 192362.000 |
| 10000 / 稀疏 | 256 | 5158.000 / 11220.250 / 15476.208 | 88.834 / 199.500 / 216.458 |
| 10000 / 全重叠 | 256 | 12430.333 / 14407.459 / 25359.750 | 24671.042 / 26519.542 / 27970.959 |

| B | 节点 | 索引容量字节 | scratch 字节 | 稀疏 / 全重叠构造 µs | 稀疏节点 / 精确刷子探针（整批） |
| ---: | ---: | ---: | ---: | --- | --- |
| 32 | 63 | 4288 | 256 | 2.708 / 22.167 | 11264 / 1024 |
| 1024 | 2047 | 139200 | 8192 | 137.167 / 110.458 | 21504 / 1024 |
| 10000 | 19999 | 1359936 | 80000 | 1407.084 / 1777.292 | 7092 / 256 |

全重叠每点访问全部 `2B-1` 节点和 B 个刷子，还要排序，因此本次全部重叠索引 P50 均比线性更慢。空间索引是显式选项，不自动替换线性基线。稀疏/全重叠三档整批 `field_work` 分别为 46080/131072、56320/3178496、15796/7688192；工作单位与 CPU 时间不可互换。

完整散布接受 4750 点，两条路径均为两个分配、880000 字节返回容量载荷；点/属性/cache/source counter 完全一致。线性 P50/P95/P99 为 21017.125/22018.583/25264.375 µs，索引为 3504.917/3692.458/3754.917 µs；线性/索引实际 `field_work` 为 10570000/526481。该负载关闭最小距离邻居探测，不能据此估计开启该探测时的整链性能。

## 仍独立的边界

本切片仅实现 CPU 作者刷子空间查询及其 source scatter 组合。GPU 刷子空间索引、增量编辑重拟合、产品 scene/ECS/render graph 接线、贴地放置、曲率/数据层/跨层规则、风/交互和蓝噪声目标须各自交付。CPU 索引可以随配方重建，不将全量构造冒充增量编辑。
