# Prism World 系统实现记录

日期：2026-10-16

本文只把已经存在源码、测试和提交记录的部分标为已交付；设计文档中的路线图、指标和效果目标仍然不能替代实现证据。

## 本批交接与验证范围

2026-10-15 按共享磁盘协调要求停止共享 target 构建；本批只整理 `prism_world_io`、`prism_world_terrain` 与本文，测试使用独立的 `/private/tmp` target，未清理或覆盖共享 target，未 push。

- 地形曲面提取使用 `CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=1 CARGO_TARGET_DIR=/private/tmp/prism-world-terrain-target cargo test --offline -j 1 -p prism_world_terrain --no-default-features`：59 passed、0 failed；随后启用 `mesh` feature 重跑为 62 passed、0 failed，包含原生 Mesh 转换回归。
- 新增 threaded 文件 provider 与 16 项回归测试源码，使用 `CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=1 CARGO_TARGET_DIR=/private/tmp/prism-world-io-target cargo test --offline -j 1 -p prism_world_io` 完成 28 passed、0 failed（其中 threaded 回归 16 项）。
- 本批运行的是指定文件的 `rustfmt --edition 2024 --config skip_children=true --check` 和 `git diff --check`，仅验证格式、解析及补丁空白，不等同于类型检查、clippy 或测试。
- `prism_world_io` 全部 target clippy 与 `prism_world_terrain` no-default 全部 target clippy 已通过；terrain `mesh` feature clippy 在编译依赖 `prism_asset` 时因磁盘耗尽中断。性能/GPU 验证及全 workspace 测试仍未运行；本批使用 `/private/tmp` 隔离 target，没有清理或覆盖共享 target。

后续异步 runtime 接线已补齐至固定 worker 文件到 `SceneHost` 的链路，并接入任务池 Future、共享取消和动态 PriorityCell，详见 M3、`prism_world_async_runtime_zh.md` 与 `prism_world_shared_priority_zh.md`；跨 LOD 地形接缝和产品性能目标仍按下文边界保留。本记录不构成完整系统验收。

## 已交付切片

### M0-M2：空间、流送和常驻

- `prism_world_stream`：`i64` 毫米坐标、负坐标 `div_euclid/rem_euclid`、sector/Morton locality、cell manifest、需求集、预测预取、优先级、预算、迟滞驱逐和可回放 frame trace。当前 scheduler 还记录已提交的 `scene_key/content_version/derived_hash`；同一 sector 的 manifest revision 变化会重新进入既有 load admission，而不会被“按 sector 已驻留”短路。
- `prism_world_residency`：A/B/C 三档状态、物化预算、确定性 promotion/demotion、HLOD 连续 alpha/dither、QoS governor、稀疏 SDF brick 和 PCG 图求值/脏传播/cache key。
- `prism_world_runtime` 的 cell reload 缺口已补齐：`CellFrameCoordinator` 用 manifest revision 识别 replacement，仍复用既有 `SceneHost::load_cell` 边界，成功后提交新 manifest；header、provider、取消、背压或 host 失败时恢复旧 manifest 的 resident 状态。该片只覆盖 cell payload replacement，不冒充 `prism_scene::SceneReloadPlan` 的模板级增量 ECS 传播，也不提供 host 内部原子回滚。
- `prism_world_runtime` 新增 engine-neutral `RuntimeDataLayers` 与 `LayeredCellFrameCoordinator`：在既有 `CellFrameCoordinator`/`SceneHost` 之上显式区分 unloaded、loaded、activated，按稳定 `DataLayerId` 排序并取 cell 声明层与 runtime-enabled 层的确定性交集；layer-set host 调用成功后才提交本地状态，失败保留旧状态，可在驻留 eviction 后重新加载并重新激活。该 slice 不解析 layer payload，不修改 terrain、GPU、ECS partition 或 render。
- `prism_world`：组合 policy core 的 `WorldRuntime`，并提供 feature-gated `SceneEcsHost`；cell 加载通过真实 `prism_scene::spawn_typed_plan`/`SceneSpawn::attach_instance`，卸载通过 `prism_scene::despawn_instance`，驻留升降级通过 typed provider 回调，不直接写第二份 scene 真相。
- `SceneEcsHost` 的 ownership registry 已覆盖真实 ECS despawn hook panic：token/binding 在 despawn 成功返回前保持可见；shutdown 逐 cell 提交，失败 cell 与待处理 cell 可由兼容 provider 接手后重试。`WorldRuntime::apply_scene_plan` 只在同一应用内跳过已成功卸载目标的后续 tier action，仍是 fail-fast、非事务接口。

### M4：GPU proxy visibility contract

- `prism_world_gpu` 已纳入根 workspace。
- CPU contract、SoA little-endian upload layout、WGSL compute shader 和 CPU golden 对拍已具备。GPU 物理输出为四个规范 `u32` 与一个未量化 `f32 screen_error`（20 字节步长），原始误差直接来自 GPU，不由 CPU 重算或从量化值还原；cluster SoA 保留几何误差位模式。
- WGSL parse/validation、CPU tests 和 `cargo check --features gpu` 已通过。
- 严格门禁命令：

```text
WGPU_BACKEND=metal PRISM_REQUIRE_GPU=1 cargo test -j 1 -p prism_world_gpu --features gpu --test gpu_parity -- --nocapture
```

2026-10-15 跨沙盒实机结果通过：Apple M2、Metal、IntegratedGpu、无额外必需 feature；真实创建 pipeline、dispatch、map/readback，并逐项比较规范四字与原始屏幕误差。测试覆盖 5 个可见/不可见对象、129 个对象的 3-workgroup 尾部、LOD 阈值、量化半步/饱和与空输入。5 个对象上传 368 字节、读回 100 字节；一般非空上传量 `128 + 48×N`、读回量 `20×N`。测试严格拒绝缺少 adapter/device、CPU adapter 或未遵守 `WGPU_BACKEND`，不把跳过算通过。telemetry 耗时是主机墙钟，不是 GPU timestamp，也不构成吞吐或 P50/P95/P99 性能达标证据。

### M4：固定容量 GPU 驻留与间接绘制

`GpuResidentVisibility` 已提供一次资源分配、验证后增量写入、显式 dense prefix、camera/count 上传与纯 encode。GPU 输出唯一 source-slot compact 流和 `[vertex_count, visible_count, 0, 0]`，真实间接实例 draw 已通过像素验证；每个非空 64-lane 工作组只预留一次全局输出区间。生产 encode 不分配 GPU 资源、不上传、不 submit/wait/map/readback。逐帧必须先提交再上传或 encode 下一帧，不能累积多个不同 camera 的 encoder 后一起提交。

2026-10-15 Apple M2/Metal 严格回归：库单测 7 项、visibility shader validation 1 项、parity 1 项、resident 3 项通过，harness all-target 与正常 workspace GPU lib clippy 通过 `-D warnings`。百万实例/75% 可见、release、8 预热+64 样本，主机 record+finish P50/P95/P99 为 21.167/33.125/35.625 µs，submit+wait 为 1371.750/1511.000/1619.583 µs；逻辑 buffer 72,000,144 字节，首次上传 48,000,128 字节，计时帧无上传/回读/绘制。不是 GPU timestamp、物理 VRAM 或完整产品帧时。API、复现方式、三档实测与未集成边界见 `prism_world_gpu_resident_zh.md`，不据此宣称 M4 全部完成。

### M4：扇区所有权与宿主生命周期

`ProxyRegistry` 提供固定代理容量、独立 sector 数量预算（含空 sector）、sector 内唯一 ID、代际 handle、交换压紧和 dirty range；`GpuSectorProxies` 持有同一注册表、预留 SoA scratch 和固定 resident GPU 资源，无变化 upload 为 O(1)，改变后的扫描/转置/清标记总计 O(N + K)，逻辑列上传为 48K 字节。未上传编辑和整体替换 registry 均拒绝 render/camera/encode，尾删和全卸载仍同步 count。返回的校验/预算错误不改变注册表；map/Arc 的普通分配 OOM 不属于可恢复错误承诺。

`prism_world_runtime/proxy_host` 在 header 校验后借用项目 decoder、原 `SceneHost` 与同一注册表，先派生 admission 后调用权威 host。load 错误或 unwind 仅补偿新增派生 sector，不能回滚 host 的部分修改；evict 成功后才删代理，失败/unwind 保留。它不另立 scene 真相、不规定 payload 格式、不启用 wgpu。真实 stream demand/coordinator→decoder/host→上传→间接绘制与策略驱逐像素回归已接通；合约、测试方法与性能边界见 `prism_world_sector_proxies_zh.md`。这不是产品 ECS decoder/render graph 的完整接入。

### M4：GPU 密度覆盖与距离淡出

`DensityPolicy` 以 `(proxy_id, cluster_id, seed)` 的稳定 PCG rank 计算平滑 coverage，不依赖可变 dense slot；可选距离 fade 与 quality coverage 相乘。构造器校验有限 normal/零参数，明确拒绝非零 subnormal；CPU/WGSL 使用相同 f32 缩放 norm、退化 width 与结果规范化规则。普通 resident 不增加资源；可选路径额外 `4×max(capacity,1)+16` 逻辑 buffer 字节，策略变化只写 16 字节 uniform，不遍历/重传列。原几何 record ABI 不变，compact 取几何可见与正 coverage 交集。下游按 compact source slot 读取 coverage，并可调用导出的 4×4 ordered-dither WGSL；部分 coverage 不被材质消费时不构成连续淡出。

2026-10-15 Apple M2/Metal 原始源码 harness 最终 96 项通过、0 失败/忽略（库 79、density 8、resident 6、parity 1、WGSL 2）。真实 draw_indirect 读取 GPU coverage 并逐像素对拍；另测两 seed、129 尾部、交换压紧、全卸载、固定资源、极小 normal range 与大坐标。实机回归发现并修复窄 transition 消减、极小距离 subnormal 余量及 smoothstep 中间平方冲零；明确检查微小正 coverage 仍在 compact 集合内。release 百万实例 density 模式输出 416,295 个 source slot，record+finish P50/P95/P99 为 48.750/57.125/64.125 µs，submit+wait 为 2058.625/2107.625/2136.208 µs；不含上传、回读或绘制，不是 GPU timestamp/产品帧时或严格 A/B 比较。API、三档双模式实测、harness 隔离说明和边界见 `prism_world_density_zh.md`。整数 GPU 候选生成另见下节；作者密度场、风/交互、HZB 与产品 renderer graph 尚须继续完成。

补充验证：no-default 原始库源码 73 项 CPU 测试通过；原始源码 harness all-target clippy 和正常 workspace GPU lib clippy 均通过 `-D warnings`。这两项局部结论不代表全 workspace 或完整产品集成通过。

### M4：可再生 CPU 散布 kernel

- `prism_world_residency::scatter` 在全局整数毫米三维格中每格生成一个 hash-jitter 候选；显式 seed/version/salt 与完整结构化 candidate key 决定位置。半开位置范围负责唯一归属，负坐标使用 Euclidean 除法；任意拆分、合并和请求顺序不会改变成员集合。完整 cache key 包含 kernel 版本、配置与范围，丢弃后可重建，不充当持久对象身份。
- 可选最小距离采用 jittered lattice 上的 Matérn type-II 式筛选，重建未请求邻格的原始候选，以 `(priority hash, full key)` 排名消除边界依赖；不是 Poisson 输入、蓝噪声保证、最大填充或递归 dart throwing。距离比较使用宽整数，等于半径允许保留。
- 预检工作量上界为 `N × (2 × ceil(radius/spacing) + 1)^3`，无筛选时为 `N`；输出数量和 Vec 容量载荷字节另有硬限制。checked 算术、一次 fallible exact reservation，超限整体失败，不返回随预算变化的截断前缀；字节限制不含分配器元数据、栈或外部消费者副本。
- no-default 测试 35 项（其中散布新增 20 项）、all-target clippy `-D warnings` 和格式检查通过，覆盖全局参考筛选、跨 cell 最小距离、未请求邻格阻挡、任意分区双向请求、缓存重建、预算与 i64 极值。这是 CPU 派生数据 kernel；其整数 GPU twin 与限定百万候选基准见下节，仍未完成产品散布渲染/地表投影及完整性能验收。

### M4：整数 GPU 散布生成

`GpuScatter` 复用 residency revision 1 的配置、范围、完整缓存键及 O(1) 工作预检。WGSL 以 u32 low/high pair 精确实现全局有符号 i64 网格、PCG jitter、请求外原始候选排名和严格最小距离比较，无浮点或 shader-i64 feature。固定容量约束所有 primary slot，不按幸存数量截断；逻辑 buffer 为 `68×max(capacity,1)+208` 字节。请求变化只上传 192 字节；encode 清计数并记录生成/compact，无分配、上传、submit/wait 或回读。消费端通过 compact source slot 直接读取 64 字节整数记录及 indirect args；队列生命周期与既有 resident 相同，必须先提交再更新请求。

2026-10-15 Apple M2/Metal 全套原始源码 harness 104 项通过、0 失败/忽略（库 82、scatter 5、density 8、resident 6、parity 1、WGSL 2）；no-default 库 73 项通过；harness all-target 与正常 workspace GPU lib clippy 均通过 `-D warnings`。GPU 覆盖 i64 两端八角、u32 spacing 极值、未请求 blocker、任意分区双向集合、63/64/65/128/129 尾部、资源复用、失败保留请求，并直接消费 records/compact 做真实间接绘制像素校验。二维 dispatch/填充编号溢出以 synthetic limit 单测检查，未宣称实机跨二维边界。release 百万无距离筛选候选 encode+finish P50/P95/P99 为 130.416/196.583/394.708 µs，submit+wait 为 3572.834/4616.916/4775.042 µs；10k/半径 17mm 筛选后 2,736 点，submit+wait 为 1519.625/1720.250/2515.084 µs。计时内无上传/回读/绘制，不是 GPU timestamp、产品帧时或蓝噪声画质验收。完整契约、四档数据与验证入口见 `prism_world_gpu_scatter_zh.md`。

### M4：CPU 作者密度与种类场

`ScatterDensityField` 是不可变、可丢弃重建的派生描述，使用全局整数毫米、三维 value noise 和三通道 domain warp；跨 i64 边缘的 warp 先扩宽再采样，仅虚拟格 hash 有明确的 2^64 周期。唯一稳定 brush ID 决定 Replace/Add 的合成顺序，球形刷子按平方距离 smoothstep 与 opacity 衰减；species 在有效权重至少 32768 时替换。候选 rank 只依赖完整 candidate key 与 field seed，固定 seed 时密度选择单调，0/65535 端点精确。全结构缓存键不使用摘要充当身份，field clone O(1)、独立配方比较最坏 O(B)。

`scatter_with_density_field` 原地过滤已有源散布，不重填空洞，保持源最小距离和半开范围的分区/逆序求值不变性。构造预算包括原 brush Vec 的 spare capacity，场工作上界在枚举前检查；合计输出预算计算两个 Vec 的实际容量载荷，不是峰值分配/RSS 上限。每次 sample/rank 无分配；该线性路径逐点采样成本为 O(B+固定噪声成本)，显式 CPU 空间索引路径见下节。

no-default 完整回归 65 项单测与 1 项文档测试通过，其中作者场新增 30 项；all-target clippy `-D warnings` 通过。覆盖独立整数 oracle/golden、全部 UNORM16 权重与密度、i64 边缘 warp、完整缓存字段与三轴任意分区双向求值。Apple M2/macOS release 的 10k 点、8 预热/64 样本：constant sample+rank P50/P95/P99 为 200.459/217.209/234.625 µs，warp+32 重叠刷子为 5392.875/32179.334/55792.958 µs；完整源生成+过滤+属性为 5878.709/12008.041/17783.000 µs、恰为两个 Vec 分配、880,000 字节输出容量载荷。并行负载干扰明显，不是 GPU 时间、产品帧时或严格优化 A/B。API、精确取整与完整基准见 `prism_world_scatter_field_zh.md`；该段是 CPU 基线验收，GPU twin 的独立验收见下节，地表过滤和产品 renderer 仍未由这些测试覆盖。

### M4：CPU 作者刷子空间索引

`ScatterBrushIndex` 共享完整不可变 `ScatterDensityField`，以确定性空间 BVH、前序 escape 节点和显式可复用 scratch 裁剪笔刷。i128 包围盒端点夹到 i64 世界域；候选 brush index 在合成前恢复 canonical ID 顺序，复用基线整数 noise/warp/距离/取整/species kernel，不改 field 算法版本或完整缓存身份。

索引与 scratch 分别限制实际 Vec 容量载荷，fallible 预留，采样不分配。`scatter_with_indexed_density_field` 在枚举前以 primary×最坏单样本 work 准入，输出原地稳定压紧、两列容量共同预算；点/属性/cache/source counter 与基线相同，实际 `field_work` 另按节点/精确刷子探针计数。复杂度为 O(访问节点+K log K+K+固定噪声)，全部重叠时可能慢于线性扫描；不是增量编辑、GPU 索引或产品帧时承诺。API、预算、独立对照与 release 复现入口见 `prism_world_scatter_brush_index_zh.md`。

2026-10-16 no-default 通过 99 库+16 集成+1 文档测试，all-target strict Clippy 通过。Apple M2 release：10000稀疏刷子/256查询批次的线性与索引 P50/P95/P99 为 5158.000/11220.250/15476.208 和 88.834/199.500/216.458 µs；全重叠索引更慢，三档负载均保留线性基线。1024稀疏刷子/10000 primary的完整散布为 21017.125/22018.583/25264.375 与 3504.917/3692.458/3754.917 µs，两边4750点、两个分配、880000字节输出容量并且完整输出一致。宿主并行负载、无最小距离邻居探测，不扩张成产品帧时结论。

### M4：GPU 作者密度与种类场

`GpuScatterField` 在 `GpuScatter` 后记录精确整数 field pass，复用 CPU revision 1 的 noise/domain warp、brush ID 顺序、平方距离 smoothstep、signed Add 取整、species 阈值和独立 rank；无 float 坐标、shader-i64 或 optional feature。生产路径无 CPU 逐点生成/采样/回读。显式五 storage binding，因此新增严格 required-limits context acquisition；原 downlevel 四 binding 默认值保持不变。

source 64 字节 record 保留源 accepted；新增 source-slot 对齐16字节属性和最终 compact/indirect。下游必须消费场输出，不能只用源 accepted。复合 owned 逻辑 GPU buffer 为 `88×max(capacity,1)+64×max(max_brushes,1)+352` 字节，另一次 fallible 预留 host 刷子 scratch。源/场工作、容量、brush 数与 dispatch 均在任何 queue write 前验证，可恢复错误保留旧请求。变更驱动上传 source192字节、field uniform128字节、brush64×B字节；不变刷子不随源范围重传。encode 无 owned allocation/upload/submit/wait/readback，调用者仍须先提交当前 encoder 再更新下次请求。

2026-10-15 Apple M2/Metal 原始源码隔离 harness 完整定向回归 113 项通过、0 失败/忽略（库84、scatter5、field7、density8、resident6、parity1、WGSL2）；no-default 库73项通过，harness all-target 与正常 workspace GPU lib Clippy 均通过 `-D warnings`。新场测试逐 slot 对 CPU 全记录/属性/rank，对 compact 唯一集合与 indirect count，并在同一 encoder 内直接消费 GPU records/attributes/compact/indirect 作实际像素验证。覆盖世界角点、u32 极值、最大 brush 半径、signed rounding、尾部/空请求/重复 encode/无 wait 连续提交和失败保留完整旧请求；非完整 workspace 或多后端验收。release 基准覆盖10k/100k三维完整 voxel、constant/noise/warp/warp+32重叠刷子，每档8预热/64样本，全部576次派发逐点对CPU。100k constant 的 submit+wait P50/P95/P99 为911.333/965.750/1211.250 µs；warp+32刷子为6437.792/6715.792/7302.500 µs，owned GPU逻辑buffer为8,802,400字节，host刷子scratch为2,048字节。计时排除上传/回读/校验/绘制，不是GPU timestamp或产品帧时。完整八档分位数、资源与digest见 `prism_world_gpu_scatter_field_zh.md`。

### M4：GPU 作者刷子有序 AABB 索引

`GpuScatterField::new_indexed` 与 `GpuScatterHeightfield::new_indexed` 显式启用canonical ID连续区间中点树，不移植CPU空间中位数树；叶序保持非交换Replace/Add/species合成顺序，signed i64保守AABB与前序exclusive escape支持无栈裁剪，无每点scratch/排序。节点64B，尾随实际brush列表打包在原storage buffer内，不增加binding或optional feature，不改算法版本、输出ABI或完整缓存身份。原`new`仍是线性基线。

构造O(B)、平衡递归栈O(log B)，查询最坏O(B)；资源和host scratch预算包括完整树，单点work另加实际节点数，全部请求预检仍在queue write前完成。只在完整brush列表改变时重建上传；base/noise/source变化不重传树，不是增量refit。indexed作者场→真实heightfield→managed完整键删除组合已与公共CPU链路对照。

2026-10-16 Apple M2/Metal冻结源码独立harness定向164项全通过，0失败/忽略；no-default库另73项与仅gpu feature索引11项通过，正常workspace GPU+terrain库与harness all-target严格Clippy通过。六类release A/B含预热共432次dispatch逐slot/属性/compact/indirect完整对照；256/1024顺序稀疏的总窗口P50线性/索引为1238.959/596.833与1444.375/492.750µs，乱序稀疏和全重叠均退化，因此不默认启用。该窗口是encode+finish+submit+wait主机墙钟，非GPU timestamp或产品帧时；完整资源、分位数、ABI及边界见`prism_world_gpu_scatter_brush_index_zh.md`。

### M4：真实 heightfield 查询与 CPU 地表过滤

`prism_world_terrain::PlanarHeightfieldSurface` 直接查询真实烘焙 tile，显式平面 origin/spacing，XZ 半开归属，沿既有 mesh 三角拓扑进行 i128 精确插值并返回几何法线。默认构建无需 mesh/引擎；可选 `scatter` 仍为 `no_std + alloc`，组合源散布/作者场后按闭区间地表高度、垂直离地带及有理坡度稳定删点，保留源 key 重建、最小距离及 XYZ 分区不变。不是贴地投影；窄带可无点，不作补点密度保证。

地形工作以 primary×4 逻辑读数在枚举前准入，实际查询 field 幸存点；复用两 Vec 原地压紧、容量继续计入原输出预算。完整缓存 key 通过 Arc 共享不可变 tile 快照，比较实际样本/spec/revision/embedding 和所有过滤参数，不用 tile/revision 或摘要替代身份。默认 78 项回归、scatter feature 93 项回归均通过，0失败/忽略；两配置 normal workspace all-target Clippy `-D warnings` 通过。新增查询覆盖非共面/对角线、半开边界、超2^53负坐标、i64/u32极值和解码超i64；过滤覆盖精确闭端点、双轴坡度、属性对齐、源距离/key重建、三轴逆序分区、完整缓存失效、公开 rebake 发布新快照及失败原子性、预算/溢出/空请求/缺覆盖。

2026-10-16 `aarch64-apple-darwin`、macOS27.0.1、rustc1.96.1 release，8预热/64样本，主机墙钟 nearest-rank：100k真实查询含逐结果校验 P50/P95/P99=1705.166/2613.750/3803.084 µs、0分配；10k源+noise/warp作者场+地形过滤为3514.916/3930.916/5631.916 µs，6132个场幸存点筛到1250点，24528次逻辑地形读取，两列容量880000字节、恰2次分配，共享heightfield8450字节另计。每次完整结果在计时外逐点/属性对oracle，计时排除构造/校验/销毁；首次fixture未对齐格被预算拒绝后修正并加入候选数断言。不是GPU时间、峰值内存、无干扰A/B或产品帧时。完整方法与边界见 `prism_world_heightfield_scatter_zh.md`。

### M4：CPU 刷子索引与真实 heightfield 组合

`scatter_with_indexed_heightfield` 显式接收已构建CPU空间BVH及有界可复用scratch，先执行地形完整覆盖/过滤参数/primary×4读取预算预检，再调用索引密度源，最后与线性入口共用精确地表过滤和原地两列压紧。完整点/属性/cache/source candidate counter/地形读数与线性相同，仅实际field_work不同；算法版本不变，支持直接组合完整键删除。新增Index错误保留原始索引错误链，不隐式创建/增长scratch或引入逐点临时输出。

正常workspace scatter配置109项库测试+新增11项公开集成测试通过，no-default配置78项通过，0失败/忽略；两配置all-target严格Clippy与本轮文件格式/diff检查通过。覆盖线性全输出对照、noise/warp/非交换ID顺序、稀疏与全重叠work、极端整数、三轴分区/距离、严格预算/溢出/错误顺序与错误链、完整缓存/删除组合、scratch跨索引/失败复用；独立源码审查未发现可复现缺陷。索引依赖residency另通过99项库测试+16项公开集成测试+1项文档测试，不代表全workspace验证。

2026-10-16 Apple M2 release，1024刷子/10000 primary、真实65² ramp，每类8预热对+32测量对交替AB/BA，全部160次计时调用在计时外全量核对输出/cache/counters与容量。稀疏线性/索引P50/P95/P99为23322.709/24135.250/24262.958与1049.458/1170.041/1262.417µs；全重叠为84574.709/89875.833/93354.000与131862.875/136042.167/138091.750µs，因此索引仍为显式可选。两Vec恰2分配、880000字节容量，索引139200字节/scratch8192字节/地形8450字节另计。无noise/warp、最小距离0，主机墙钟含源/场/地形/输出分配，不是GPU timestamp、RSS、产品帧时或完整系统验收；复现与预算见`prism_world_heightfield_scatter_zh.md`。

### M4：真实 heightfield 的 CPU 二维贴地放置

`scatter_on_heightfield` 是独立于三维稳定过滤的新入口：canonical grid Y=0的全局XZ jitter提供二维候选，无随机Y slab；水平Matérn-II筛选包含请求/表面外以及随后被terrain/field拒绝的原候选。真实三角面高度与整数offset相加后整体取整，半毫米远离零，投影溢出整体报错；地表高度/坡度仍按原精确样本过滤，作者场在最终取整偏移位置采样。`HeightfieldPlacementKey`显式包含placement算法版本和完整source key，`rebuild(surface,offset)`重建投影几何，不修改旧ScatterPoint身份或既有API。

任意非对齐XZ半开分区和请求顺序不改变成员，水平最小距离保证投影后至少同样的三维欧氏间距，但不保证测地线均匀、最大填充、Poisson或蓝噪声。完整缓存含四种算法版本及全部配置/region/filter/terrain/field；候选/四角/作者work上界在分配前检查，两列各一次fallible reserve、实际capacity计费，只对最终接受行检查count上限。CPU `no_std + alloc`，无engine/mesh/GPU依赖或逐点分配。

2026-10-16 Apple M2/macOS27.0.1/rustc1.96.1 release，真实65² ramp、10000候选、三枚非交换作者刷子，半径0/150mm各8预热+32样本；全部80次计时调用均完整逐点对独立整数参考算法、重建key、检查属性/cache/counters与全部输出对水平距离。两档P50/P95/P99为1036.750/1085.833/1094.375与6245.959/8721.583/10955.792µs，最终4571/683点，实际candidate探针10000/122011；两Vec capacity共1840000字节、恰2分配，场/高度共享载荷另计。保留尾延迟，不推导无干扰跨档加速比或产品帧时。方法、资源、语义及复现见`prism_world_heightfield_placement_zh.md`；该源生成片不包括GPU/行星放置、曲率/数据层/跨层规则或产品renderer接线，独立placement持久删除由下述M4/M5后续切片交付。

本片最终正常workspace回归：`scatter`为109库+26新放置集成+11既有索引集成，no-default为78库，均0失败/忽略；两配置all-target Clippy `-D warnings`及定向rustfmt通过。另对既有GPU地形过滤在Apple M2/Metal以原生产crate/原测试文件的最小验证包连续两轮通过8项，含真实indirect draw像素；正常workspace尝试被并行修改的`prism_tasks::arena`缺少`Layout`导入阻断，隔离验证使用自身锁依赖。没有修改资产/GPU源码，不将该结果当作新放置GPU twin或完整workspace通过；验证细节见上述放置文档。

### M4：真实 heightfield 的 GPU 二维贴地放置

`GpuHeightfieldPlacement` 是 CPU `scatter_on_heightfield` revision 1 的固定容量 GPU twin，
独立执行二维 XZ source、水平 Matérn-II、真实三角投影、未偏移高度/坡度过滤、最终 XYZ
作者场和 finalize。source/projection/field/finalize 通过有序 dispatch 连接；source record、
sample、attribute、compact 和 indirect 均保持 source-slot 对齐，生产 encode 不做 owned
分配、等待或读回。WGSL 只用 core u32 limb；Naga ABI 闸门验证 192B/128B uniform、64B
record/sample、binding access、入口和无 float/wide scalar。

投影高度加 offset 后整体半值远离零取整，投影溢出在过滤前检测；任何溢出都会由 finalize
清空 indirect count 和所有 live acceptance，禁止部分成功。完整 source key、terrain/field
快照、filters、embedding 和配置参与缓存依赖；构造和 request preflight 先验证完整 brush/
surface、设备 binding、work budget 与固定 capacity，逻辑 buffer 为
`148C+64B+4S+468`（C/B/S 至少为1）。

2026-10-16 Apple M2/Metal 严格回归 7/7 通过，覆盖三角 oracle、signed normal、坡度/闭区间、
半毫米跨零、最终 Y field、刷子/noise/warp、非对齐分区、i64 边缘、缓存/事务、整体溢出和
真实 indirect draw 像素；无设备 Naga 闸门 6/6 通过。release 8 warmup+64 samples 的
10k/100k×0/150mm 结果、完整资源和复现命令见 `prism_world_gpu_heightfield_placement_zh.md`。
这只完成二维平面 GPU kernel，不代表 GPU placement deletion、行星/曲率/数据层、风/交互、
产品 scene/ECS decoder/renderer graph 或多厂商性能已完成。

### M4：真实 heightfield 的 GPU 精确地表过滤

`prism_world_gpu/terrain_scatter` 的 `GpuScatterHeightfield` 组合整数源生成、作者场与地形过滤，直接读取真实 baked 高度样本，不在 CPU 逐点求值或回读生产输出。六 storage binding、128B 地形 uniform、i32 样本；有符号 i64 世界坐标与更宽有理高度/离地带、精确双轴坡度由核心 u32 limb 完成，不需要 shader-i64 或其他 optional feature。与 CPU revision1 三角/闭端点/坡度逐项一致，只删点、不移动/补点，保持源最小距离、key 重建和 XYZ 分区成员不变。

全部资源在构造前验证，owned GPU 逻辑 buffer 为 `108C+64B+4S+496` 字节（C/B/S 为至少1的源/刷子/样本容量）。完整请求/读数/源/场预检在任何 queue write 前进行，可恢复错误保留旧请求、结构 key、固定资源与 GPU 输出。地形 uniform 变化上传128B；真实样本变化上传4N字节，只有 embedding、过滤或 metadata 变化不会重传相同样本。encode 无 owned 分配/上传/submit/wait/map/readback；必须先提交当前 encoder 再改请求。原64B源记录不改写，消费者必须使用最终16B属性、compact及indirect。

2026-10-16 Apple M2/Metal 正常 workspace package 定向回归126项通过、0失败/忽略（库87、density8、parity1、resident6、field7、scatter5、heightfield8、terrain draw1、WGSL3），no-default库另73项通过。八项新增实机回归涵盖全部slot CPU oracle、compact唯一性、真实draw_indirect像素、有理/极值/坡度溢出、缓存与rebake、任意XYZ分区、尾部/空请求/连续无wait提交和失败保留旧输出。隔离原始源码 harness all-target Clippy `-D warnings`通过；正常 workspace all-target Clippy 在共享依赖 `prism_asset` 的164项现有lint/doc错误处失败，不视作通过，也不修改其他工作。

release两轮每档8预热/64测量，共288次dispatch全部逐点对CPU。10k/100k三维候选分别从6132/61122场幸存点筛为1250/6193点；逻辑GPU buffer为1097460/10817460B，host高度scratch16900B，共享原高度8450B，首次显式上传含indirect初始化17396B。回归后复测的submit+wait P50/P95/P99分别1442.333/2362.833/4674.250µs和7522.584/15448.750/25020.584µs；首轮同样保留于 `prism_world_gpu_heightfield_scatter_zh.md`，不挑选较好尾延迟。计时不含上传/回读/CPU检查/绘制，不是GPU timestamp、物理VRAM或产品帧时；贴地放置和完整产品散布仍另行验收。

### M4/M5：可再生散布的完整键持久删除

`ScatterDeletionSet` 以不可变完整结构键集合提供稳定原地过滤；`ScatterDeletionBinding` 将整组键存于调用方选择的作者节点 PersistentId/FieldId 的 override Bytes，而不是散布 dense slot、截断 hash 或第二份内容数据库。格式 `PSDL` revision1 为16B头+每键52B，严格数值排序/唯一、完整键重建及独立 wire/count/Rust容量准入；错误不降级为空。单键实际修改O(D)拷贝、过滤O(N log D)，原points/attributes指针容量与源key/距离语义不变，不补点。runtime输出保留node/field/完整集合额外cache依赖；整字段提交沿用Timeline真实inverse及字段LWW，明确不是逐键并发union。

真实FileJournal关闭重开→回放→丢弃并重新生成缓冲→过滤，以及持久undo/redo回归已通过。并行流送修改前默认runtime 78、no-default 30、proxy_host 98测试通过，residency 91单测+1doctest通过；最后三项回归加入前全feature曾通过115项。默认/no-default all-target严格lint及完整示例随后通过（51点、删除/undo/redo为1/0/1、3条事务回放）。并行流送修改曾因未匹配新增`StreamAction::Reload`阻断重跑，该阻塞现已修复；2026-10-16独立target、offline/locked定向lib重跑默认79、全feature119项，均0失败/忽略，不扩大为当前脏工作区全配置或全workspace通过。完整可执行示例以clone staging、落盘成功后才发布内存Timeline；保留进程内撤销栈、不声称重建重启后的可编辑历史。这是明确的失败发布边界，不是产品增量事务或跨资源崩溃原子指令。

2026-10-16 macOS27.0.1/aarch64/rustc1.96.1 release，每项8预热/64样本：10k/100k输入、25%删除，原地过滤主机P50/P95/P99分别791.917/797.167/922.708µs及9758.500/9875.125/11031.166µs；独立encode+decode为106.000/109.209/113.917µs及1083.584/1092.292/1252.833µs。完整结果/指针/容量在计时外逐样本校验，不含生成/clone/析构，不是GPU时间、RSS或产品帧时。wire/实际容量、完整契约及历史/合并/发布成本见`prism_world_scatter_deletions_zh.md`；GPU及地形删除适配见下节，产品接线仍另行验收。

### M4/M5：二维贴地放置的完整键持久删除

`HeightfieldPlacementDeletionSet`只接收独立placement版本与canonical grid Y=0的完整源键，验证不依赖当前地形覆盖或请求成员；不混用原三维删除。不可变Arc快照、构造去重前count/实际capacity预算、无变化O(1) clone和实际修改O(D)拷贝均有回归。`filter_heightfield_placement`先检查列对齐，再稳定原地压紧真实投影点及作者属性，保留两Vec指针/容量、源缓存和工作计数，不重投影、不补点。

runtime显式`placement` feature提供`PlacementDeletionBinding`、完整namespace/set/filter-version缓存依赖及PPLD revision1（16B头+56B完整键）；默认依赖不扩大，启用placement仍支持no_std+alloc。严格规范顺序、完整身份、版本、长度及独立wire/decoded容量验证，不接受PSDL或静默清空。scene字段仍为唯一权威，沿用Timeline真实inverse/字段LWW，不是逐键union；日志与Timeline不自动原子。完整示例以clone staging→真实FileJournal同步→发布，再关闭重开、回放、丢弃并重新生成400点/属性，删除/undo/redo为1/0/1、3条事务全部核对。

2026-10-16正常workspace独立target回归：terrain scatter库132+二维放置集成26+索引地形集成11、no-default库78项；runtime默认79、placement104、no-default+placement55项全部通过，含新增23+25项删除/持久恢复测试。四种定向配置all-target Clippy `-D warnings`通过。Apple M2/macOS27.0.1/rustc1.96.1 release，10k/100k×0/25/100%删除，每操作8预热/64测量，两轮共1728次计时调用及全部解码后过滤对照通过。最终100k/25%过滤P50/P95/P99=10103.084/13572.792/19537.500µs，独立codec为1233.209/2192.375/2230.084µs；100k全删过滤P99=115402.916µs明确保留。空删除先验证列对齐后O(1)返回；两列capacity1840000/18400000B，全删不释放。主机计时排除生成/clone/外部oracle/销毁，不是GPU时间、峰值内存或帧时承诺；两轮全部档位、资源/历史成本及边界见`prism_world_placement_deletions_zh.md`。本片不改变GPU代码，不将CPU持久恢复冒充GPU二维放置、产品renderer接线或整个系统验收。

### M4/M5：CPU 地形删除与 GPU 完整键删除

`filter_heightfield_scatter` 在真实地形过滤后稳定原地压紧 points/attributes；即使空删除或空列也先检查长度对齐，保留原指针/容量、完整地形/作者场/源缓存键与上游工作计数。删除集合（含请求外键）是额外完整缓存依赖；不重烘、不改变位置或补点。normal workspace terrain no-default 78项、scatter 109项（新增16项）全部通过，两配置 all-target Clippy `-D warnings`通过。

`GpuScatterDeletions<U>` 独占 sealed 源/作者场/地形上游，shader只读取最终compact/count并按完整13-word键二分删除，不复活中间被剔除的点。保持 records/attributes 未改，消费者必须使用最终 compact/indirect。新增逻辑buffer为`52*max(Dcap,1)+4*max(C,1)+144`字节；预算覆盖整链、实际key scratch容量和`N*ceil(log2(D+1))`比较界，D=0仍预算遍历。所有层先检后写，错误保留旧请求/输出；encode无owned分配/上传/submit/wait/readback。context clone共享私有Arc设备身份，拒绝跨Instance局部handle相等误认。调用者仍须先submit当前encoder再write下一请求，raw API不自动检测提交epoch。

2026-10-16 Apple M2/Metal、无optional feature，冻结后原始源码隔离harness共139项全部通过（库100、源删除7、组合删除8、既有源5/场7/地形8、WGSL4），含真实draw_indirect像素、完整记录/属性不改写、CPU全键oracle、独立设备拒绝/clone接纳、预算失败原子性和无wait顺序提交。harness all-target与正常workspace terrain_scatter GPU lib依赖闭包Clippy均通过`-D warnings`，正常workspace no-default/terrain_scatter lib check均通过；不是全workspace验收。

release两组完整记录共864次dispatch逐全键/位置对CPU，10k/100k候选×0/25/100%删除、8预热/64测量。冻结后100k/25%删除encode+finish P50/P95/P99=57.500/156.375/330.167µs，含submit+wait为2004.875/2523.542/7724.459µs；整链逻辑GPU8500352B、新增1700144B、key scratch1300000B。首轮和其他档位完整保留于`prism_world_gpu_scatter_deletions_zh.md`，不挑快轮、不作无干扰A/B；计时排除构造/上传/回读/校验，非GPU timestamp、物理VRAM/RSS或产品帧时。没有因此完成风/交互、逐键并发协议或产品renderer接线。

### M5：持久化

- `prism_world_persist`：`PersistentId`、规范化 `OverrideDelta`、append-only log、确定性 LWW CRDT、snapshot/timeline、undo/redo、little-endian codec/checksum。
- `std` feature 提供 `FileJournal` 和 `SnapshotStore`：版本头、长度/序号/payload/checksum、批量 append、`flush + sync_all`、重启恢复、损坏/截断尾部裁剪、乱序/重复拒绝、同目录临时文件原子替换和父目录同步。
- `FileJournal::open` 在读取、初始化或恢复之前取得独占 OS 文件锁，并保持到句柄销毁；第二个句柄明确返回 `Locked`，不支持绕过锁写入或替换锁定路径。写入、flush 或 sync 的 I/O 失败会 poison 当前句柄，即使尾部回滚成功也必须销毁并重开；回滚失败同时保留原始错误和回滚错误。故障注入走同一生产 writer，覆盖 seek/短写/flush/sync 和 truncate/seek/flush/sync 回滚失败。失败后内存 log 保持旧视图，但回滚失败且磁盘已有完整记录时，重开允许恢复这些记录；batch 不是崩溃原子事务。
- 本轮 macOS 验证：`cargo test -j 1 -p prism_world_persist --features std --quiet` 的 18 项测试、真实子进程锁回归 2 项，以及 `cargo check -j 1 -p prism_world_persist --no-default-features` 通过。锁内读测试复用持锁文件句柄并恢复游标；跨进程回归证明父进程持锁时子进程得到 `Locked`，释放后可重开并读取已提交前缀，且文件字节不变。尚未提供 Windows 实机验证。
- 相关提交：`4d0fbfebe`；Scene/ECS 接缝提交：`34656b305`；GPU workspace 集成：`c94d5cf13`。

### M3：异步 IO/provider

- 共享动态优先级：`spawn_future_with_priority` 的初次提交/wake/量子 continuation 与 world 路由/scheduler/native chunk 保留同一个 `Arc<PriorityCell>`，`PrioritizedCellIo` 可直接包裹任务 scope 与 coordinator。IO 按 live lane、静态分数、FIFO 准入；动态队列有界扫描，静态队列保留 heap，活动读取不抢占。原生逐块采样 kernel hint，物理槽等待和 NamedThreads 派发仍 FIFO；任务池显式 priority inbox 始终先于普通队列，并非 IO 的隐式 Normal 排序。生命周期、性能和验证细节见 `prism_world_shared_priority_zh.md`。
- `prism_world_runtime/tasks` 提供 `TaskCellIo` child scope 和 `next_cell_completion`，真实 archive 加载在 `TaskPool` 的 0/1/2 worker 上完成 checksum/header/host 链路。`CancelToken::cancelled()` 可唤醒停驻任务，逐 poll 有界取消，迟到 payload 受抑制，drop 逐个取消未完成请求。零 worker fallback 已修复外部 IO 唤醒只入队不通知 driver 的停顿；runtime tasks 63 项、tasks std 273 项和 11 项文档测试通过。async harness 每 dispatch 最多 32 poll、continuation 进入全局 FIFO；零 worker root 每 32 dispatch 再获 poll，仍为合作式且不保证跨队列严格公平。详细生命周期、panic 清理与 feature 验证边界见 `prism_world_async_runtime_zh.md`。
- `prism_world_runtime/reactor` 提供 `ReactorCellIo`，借用宿主既有 `IoReactor`/`NamedThreads`，IO worker 独占 fd/native future 并在 reap 后关闭；compute 只 poll。取消终态不提前退还原生清理配额，有界 FIFO 同时保留槽与声明字节后锁外唤醒，不自旋重试。真实 native archive 在 0/1/2 compute worker 完成校验和 host 链路，IO 63/runtime 67 测试通过；Apple M2/macOS 的 2 MiB cached archive release P50/P95/P99 为 4.413/4.511/4.525 ms。IO worker 会 park，非全部线程无阻塞；当前 backend 仅 Apple。所有权、2×capacity 准入界、复现与限定测量见 `prism_world_native_io_zh.md`。
- `prism_world_io`：分块读取请求、稳定优先级/FIFO 准入、活动读取 round-robin、单调请求 ID 的 fail-closed 耗尽保护、显式容量背压、可取消 poll 操作、连续 chunk/short-read/checksum 校验和 std 文件 provider。
- 默认构建保持 `no_std + alloc`；`std` 只增加 scheduler 与文件系统 provider，不把 IO、资产数据库或 ECS 引入 world policy core。
- `prism_world_runtime`：真实调用 `StreamScheduler` 生成 load/reload/evict 动作，并通过有界、可取消、优先级排序的 provider 队列提交 cell；完成后执行固定 manifest header 校验，再进入 `SceneHost::load_cell`，失败/驱逐/reload 失败写入可 replay trace。std 构建新增 `IoSchedulerAdapter`，实际桥接 `prism_world_io::IoScheduler` 的 chunk/completion/cancel/backpressure/error 事件，并保留同步 `ChunkProvider` 兼容路径。
- `prism_world_runtime/asset_container` 现提供真实 `AssetContainerCellProvider`：使用已校验的 `RuntimeManifest`/`OverlaySet` 通过 `ManifestReader` 路由 `scene_key`，底层复用 `ContainerReader`/`RangeSource` 的 raw、deflate 与 range-backed 读取；在分配前检查声明长度/上限，拒绝非零物理 offset、路由/身份/范围错误、短读、header/checksum 不匹配和取消，不把部分字节交给 `SceneHost`。这是同步兼容 provider，不宣称可中断正在进行的容器读取、GPU 直传或场景解析。
- `prism_asset_container` 的可选 `mmap` feature 现提供只读 `MmapRangeSource` 与 `ContainerReader::from_mmap_file`：文件以共享不可变映射承载严格 header/ToC/填充/完整 payload hash 校验和后续 range 读取，空文件、usize 转换和源边界均 fail-closed；默认 positional `FileRangeSource` 不变。它完成了 M3 的 mmap-backed container range vertical slice，但 manifest/ToC 元数据仍由现有 reader 解析为受验证索引，不宣称已完成设计目标中的零解析 schema view，也不包含 GPU 直传或硬件解压。
- `FileCellProvider` 复用 `prism_world_io/std` 的真实 `FileProvider`，以 sector 注册完整 manifest/文件 offset/checksum，拒绝过期 manifest 并保留调度优先级。真实临时文件测试覆盖文件→分块 scheduler→header/checksum→`SceneHost`，及短读、打开失败、取消和错误隔离；底层每次 poll 为有界阻塞读取，不宣称已变成线程池异步文件 IO。
- 本批新增独立的 `ThreadedFileProvider` 源码：固定 std worker、预留槽位/队列、受限 chunk、ACK 驱动的后续读取、worker 内文件打开/读取/关闭，以及非 joining 的取消/shutdown。取消的 Ready 缓冲区交由 worker 清理；槽位在结果和文件销毁后才释放。waker 克隆/替换后的销毁及唤醒位于 metadata 锁外，worker/批量 shutdown 隔离唤醒回调 panic。逻辑 payload 约束为 `capacity × max_chunk_size`，不包含调用方已取走的 chunk、线程栈、元数据、分配器开销或 RSS；阻塞 OS 调用只能合作取消，返回前仍占槽位。16 项回归覆盖范围/短读/打开失败、准入、parked ACK、queued/running/ready 取消、重复 shutdown、last-owner drop、唤醒时序及 panic，已与 crate 其余测试合计 28 项通过。
- 新增 `CellIo`/`CellFrameCoordinator` 的 task-context poll 接口；`ScheduledCellIo` 聚合完整 cell，每次最多消费 16 个 chunk，随后唤醒并让出。`WorldFrameCoordinator<P>` 是同步兼容别名，保留 `new/poll_io`；旧路径仍可能阻塞，不把兼容 API 称为异步。同步 runtime 的 ID 也改为单调非零、耗尽显式失败，并用 ID 保持同优先级 FIFO。
- `RoutedCellIo` 在提交前检查完整 manifest 版本，保留注册的 offset/checksum 与调用方优先级；`ThreadedCellIo::threaded` 将其接入独占固定 worker pool，要求 active limit 不超过物理槽位。真实 archive 回归从 stream demand 到 worker 分块、checksum/header 校验和 `SceneHost` 提交，错误内容不会进入 host；不是 scene/ECS 或 GPU 端到端验收。
- `QueuedThreadedFileProvider` 提供取消清理后的有界容量等待：所有 clone 共享 `capacity` 条准入记录与 `capacity` 个物理槽位，FIFO 授予槽位后再唤醒；等待中不保留 payload/file、不使用自唤醒重试。直接 `ThreadedFileProvider::start` 仍保留立即背压契约。IO crate 42 项测试通过，包含阻塞清理后的替换请求、2 MiB 文件校验、多 scheduler、FIFO grant、取消/drop、shutdown 及重入/panic waker。
- `IoSchedulerAdapter` 以带 ID/原始请求的 `Failed` 终态关联错误；scheduler 串行锁内跟踪实际 start/poll 回调，scheduler 的显式取消 ID 优先，provider 不能伪造其他请求的取消 ID。并发 blocking reader 保留各自结果、共享 provider 唤醒可唤醒全部等待者，public poll 不抢占其结果；终态被消费前仍计入 capacity，取消会唤醒停驻 reader。
- adapter public poll 的 waker 由最新 Context 替换，blocking reader 按请求 ID 各有一个注册；锁外执行 clone/drop/wake，锁内 provider 的同步通知延迟到 adapter guard 解锁后派发，隔离回调 panic 并保留重入注册与并发通知。本轮 IO 42 项、runtime 53 项及 no-default runtime 6 项测试通过；两种 feature 的 all-target clippy 通过 `-D warnings`。真实 archive release 测量与限定条件见 `prism_world_async_runtime_zh.md`。
- `IoByteLimits` 为 adapter/file provider 增加独立的单请求和累计声明字节准入，包含 44 字节 header；mutex 内在分配请求 ID/启动 IO 前检查上限及累计溢出，拒绝不消耗 quota/ID。queued、active 和未消费的 Completed/Failed 均保留完整配额，消费 Chunk 不释放；兼容构造器保留无配置字节上限行为，但仍拒绝算术溢出。
- 聚合首次 chunk 时 fallible reserve 完整声明大小，后续 append 也为 fallible；分配失败取消底层操作，待 scheduler 移除后才以原 request ID/分配错误报告终态。累计上限 B 限制逻辑准入，不是 RSS 硬上限：保留 Chunk 副本与聚合结果的逻辑长度可接近 2B，Vec 实际容量/分配器、provider、事件元数据及交还调用方后的缓冲区不在计数内。
- 预算补丁在并行任务验证快照中通过 runtime 全 feature 测试 27 项、no-default 测试 3 项及两种 feature 的 all-target clippy/check/fmt；预算批新增 8 项预算/文件验证，既有 5 项并发/关联测试先前额外重复 100 轮通过。历史收尾重跑曾因并行修改使用 `error.to_string()` 而触发泛型错误类型缺少 Display 的 E0599；后续已恢复 Debug 格式化，并在本批 threaded provider 修改之前通过 runtime 全 feature 27 项测试。该记录不代表本批新 provider 已被编译，也不将当前共享工作区误报为全部通过。上述 runtime 验证不依赖 `prism_asset`。

### M6-M8：地形、仿真与网络边界

- `prism_world_terrain`：cube-sphere 六面映射、四叉树 tile/跨面邻接、连续 screen-space LOD/morph、i64/Q32.32 局部坐标误差界、确定性高度采样和小端体素布局。
- M6 纵切已补齐：`TerrainStreamAdapter` 将 `TileKey` 映射为稳定 `SectorId`，计算保守行星毫米 AABB，生成稳定 `CellManifest`/derived hash，并可直接交给 `StreamScheduler`；12 项 terrain 测试、fmt/check/clippy 通过。
- M6 可执行 heightfield kernel：全量烘焙为带正毫米比例的 signed i16 高度，超范围明确失败；`BakeLimits` 在采样和分配前限制数量/载荷字节，分配失败可返回错误。局部脏区按网格 floor/ceil 保守覆盖，先完成 scratch 采样再提交；失败不改变高度或 revision。`GridChange` 是单个稳定 cache owner 的有序 token，不是跨缓存的全局身份。
- `HeightfieldMesh` 输出本地 XZ 米制索引网格，+Y 三角绕序和有限差分单位法线；更新覆盖脏高度及一圈法线邻域、不重分配原 mesh。载荷为 `32×顶点数 + 24×quad数` 字节。`mesh` feature 输出真实 `prism_mesh::Mesh`，不是渲染占位数据。
- `PlanetHeightfieldMesh` 将已烘焙高度投影为六面 cube-sphere 索引网格；TileKey 整数身份经 f64 UV/归一化投影，减 i64 毫米 origin 后编码为局部 f32 米制位置。半径限 `1..=2^52` 毫米，非正半径+高度、非有限值、载荷超限和超过位置误差预算均失败；误差界包含浮点运算及编码，不包含高度量化、三角插值和 LOD 接缝。测试覆盖六面向外绕序/单位法线、同级相邻 tile 与跨面边界位置、地球半径第 30 层 tile 小于 0.001 mm 的数值误差界及原生 Mesh。`validate_planet_heightfield_edge` 按精确跨面方向和有界完整边比较，在发布相邻 tile 前拒绝 tile/分辨率/毫米高度断裂；`PlanetHeightfieldNormalNeighborhood` 使用邻 tile 内侧一行计算跨 tile 边界法线。新增 `PlanetHeightfieldMesh::update` 消费有序 `GridChange`，只重投影变更矩形并重算一圈法线，身份/邻域校验和完整 preflight 在提交前完成，无 mesh 资源重分配；更新结果与全量重建逐顶点一致。新增 `PlanetMeshPublication` 将多个 replacement/update/remove 组成单次 generation 提交，统一执行 tile/字节/邻边预算和跨 tile 高度校验，失败不改变旧快照。混合 LOD 缝合和 renderer graph 接线仍是独立边界。
- `SdfGrid` 全量烘焙 signed Q16.16 到 X-fast 小端 i32；半开 XYZ 局部区域先 scratch 采样、成功才覆盖并递增 revision。按完整 dimensions、实际 source 调用数、输出+scratch 逻辑峰值字节预检，checked 算术及 fallible allocation 拒绝溢出/分配失败；任何采样失败保留原 bytes/revision。`quantize_sdf_mm` 用 i128 最近舍入、ties-away、超范围拒绝；`VoxelSdfSource` 只将声明为 Q8.8 的距离精确扩展 256 倍，拒绝 density/格式或规格不匹配，不伪造距离变换。空间 embedding 由 source 保持稳定；这一步没有执行曲面提取。
- 新增 `SdfSurfaceMesh` 在同分辨率的固定六四面体格上提取零等值面，规范 sample/edge key 焊接顶点并消除重复端点退化面；半开 cube 区域读取一圈 sample halo，任意等分辨率分区与全量提取的有向三角集合一致。负值在内、零在外，输出外向 CCW 索引与按实际编码三角面面积加权的单位法线；开放区域边界法线仍为单侧平均。i128 先减整数毫米 origin，再作 f64 插值及局部 f32 米制编码，拒绝精度超限、面翻转/塌缩、无定义法线和坐标超界；误差界不覆盖采样、量化或曲面近似误差。
- 曲面构建对 cube/triangle/vertex/逻辑峰值字节设硬限制，checked 算术与 fallible reservation，失败不修改源 grid。非空构建三次遍历，排序/查边复杂度 `O(C + T log T + T log V)`；输出载荷 `24V + 12T`，峰值 `max(60T + 12V, 12T + 48V)`，不含借用 grid、旧 mesh、分配器开销/容量取整和固定遍历栈。no-default 回归 59 项及 `mesh` feature 回归 62 项已通过：覆盖 256 种符号 mask、距离极值、各向异性平面、零点焊接、闭球流形/Euler、任意区域分区、极大共用 origin、精确预算、错误路径与原生 Mesh 转换；这不是 Dual Contouring、Transvoxel 或混合 LOD 接缝实现，也尚未进入 renderer/physics 产品链路。
- 表示选择和重烘规划现按实际 XY 网格覆盖乘 Z 深度计算工作量/字节，覆盖位置敏感的 floor/ceil 边界、溢出、完整规格比较及输入排列无关的脏区合并；本批 terrain no-default 59 项、启用原生 mesh 后 62 项通过。IO 与 terrain no-default clippy 已通过，terrain mesh clippy 受磁盘耗尽阻断；性能/GPU 仍未验收，原生 mesh 的依赖 `prism_asset` 仍有既有文档告警。
- `prism_world_sim`：统一 tick/output/dirty-region budget，确定性局部侵蚀、持久体素/碎片 delta、水文容量约束、D8 流向/汇流量/流域分析、生态 biomass 更新和小端 delta codec。
- 2026-10-16 D8 分析切片验证：默认配置 20 项测试、`persist` 配置 25 项测试和 all-target/all-features strict Clippy 均通过；独立 `/private/tmp/prism-world-sim-d8-20261016` target 验证固定八邻居预算、区域裁剪、同高 tie-break、流域出口、河网阈值、`i32` 坐标边界及失败不变性。该证据只覆盖 CPU 拓扑分析，不代表 GPU 水面、侵蚀反馈或水引擎接线。
- M7 持久接缝已补齐：`simulation_delta_to_transaction` 将 `SimulationDelta` 编码为稳定 opaque `Bytes` forward + 同字段 `Delete` inverse，`WorldRuntime::append_simulation_delta` 自动使用 log next sequence 并写入 append-only log；simulation feature 测试覆盖 round-trip、错误和 facade 入口。
- M7 仿真状态回放已闭合：`WorldSim::apply_delta` 的正向校验与 `WorldSim::reverse_delta` 的反向校验均在写入前完成；`VoxelDelta` revision 2 保留 hardness，`WorldSim::fragments` 保留 stable-key 碎片，使质量归零的体素和已生成碎片可以无损回滚。no-default 22 项、persist 27 项单测和 all-target/all-features strict Clippy 通过。该切片仍不包含 GPU stencil、SDF 重烘或产品级时间轴调度。
- M7 SDF 只读接缝已补齐：`SparseSdfQuery`/`SdfQueryConsumer` 为多个消费者共享同一 sparse grid，明确区分 `Value`、`Miss` 与缺 brick 的 `Unknown`，覆盖负坐标、范围和缺失数据测试；不会把未知场误报为空场。
- `prism_world_network`：复用 stream/persist 身份的 AOI、delta envelope、authority/read-only/shared-LWW policy、sector handoff ledger 和顺序无关 PersistentId 合并；不拥有传输层或第二份世界真相。
- M8 session seam 已补齐：`Session` 提供 token gate、有限 outbound/inbound 队列、序号、累计 ACK、retry deadline、duplicate suppression 和失败状态；`MemoryTransport` 提供 no_std 可运行的双端测试传输。9 项 network 测试及 no-default-features/clippy 通过。真实 socket/加密 transport 仍由宿主提供。
- 这些 crate 已加入根 workspace，均有可运行算法和单测；它们是 world 接缝的 engine-neutral 实现，不宣称已经完成整个 renderer/physics/network transport 端到端产品。
- `SparseSdfBrickGrid` 已支持负坐标采样、跨 mip dirty propagation、删除后的父 mip 失效，以及 brick/region 的保守 i16 min/max 查询；查询返回 `None` 时表示稀疏数据未知，不能当作空场。
- `CpuSdfClipmap` 已补齐固定容量、确定性排序和事务式中心滚动：每个 mip 保持固定 X/Z 方形覆盖，负坐标使用 floor 除法，更新只返回新进入的 brick，坐标/容量/分配失败不会提交半成品布局。它是 SDF clipmap 的 CPU 布局与 residency 输入契约，不包含 GPU 纹理分配、SDF 采样、上传或 renderer graph 接线；这些仍需独立实现和真机验收。
- M7 GPU SDF brick lookup 已补齐：`GpuSdfLookup` 以固定容量上传 `BrickCoord` 排序的完整键、元数据和两枚 `i16`/`u32` packed samples，shader 用负坐标 floor division + binary search 做 `O(log B)` lookup，并以显式 `Value/Unknown` 结果区分缺 brick；上传、query、设备 binding 和 buffer budget 均在 queue write 前预检。WGSL ABI 5/5、residency 109/109、GPU crate strict no-deps Clippy 通过；跨沙盒 Apple M2/Metal 两项真实 parity 通过，证明 pipeline/dispatch/packed 解包/readback。该 slice 仍不等于 renderer、physics、GI、nav 或 runtime clipmap eviction 接线。
- terrain 在最大 quadtree 层级返回显式 `NoChildren`，LOD 非有限值和高度乘法溢出有确定性错误/饱和行为；simulation 的 tick、hydrology 和 delta codec 增加回绕/溢出、canonical record、输入大小和记录数限制；network merge 以 `TransactionId -> PersistentId` 索引去重，避免按对象桶线性扫描。

## 当前验证

本轮 scene/world streaming 切片的定向验证：`rustfmt --edition 2024 --check pkg/prism_world_stream/src/lib.rs pkg/prism_world_runtime/src/lib.rs`，以及 `cargo test --offline -p prism_world_stream -p prism_world_runtime --lib --no-fail-fast`；结果为 `prism_world_stream` 5 项、`prism_world_runtime` 79 项通过。新增回归覆盖 manifest revision 触发 replacement、成功提交新 cell，以及坏 header 后旧 resident 状态可重试。工作树的全量 `cargo fmt --all -- --check` 仍会报告其他并行改动文件，本轮没有格式化或修改这些非目标包。

通过：

```text
cargo fmt -p prism_world_stream -p prism_world_residency -p prism_world_persist -p prism_world -- --check
cargo test -p prism_world_persist --features std
cargo test -p prism_world --features scene_host
cargo test -p prism_world_gpu
cargo test -p prism_world_gpu --features gpu --test wgsl_compile
cargo check -p prism_world_gpu --features gpu
CARGO_TARGET_DIR=/private/tmp/prism-world-target cargo check -p prism_world_io -p prism_world_network -p prism_world_sim -p prism_world_terrain --no-default-features
cargo test -p prism_world_io --features std
cargo test -p prism_world_network -p prism_world_sim --features prism_world_sim/persist -p prism_world_terrain
CARGO_TARGET_DIR=/private/tmp/prism-world-target cargo clippy -p prism_world_io --features std --all-targets -- -D warnings
CARGO_TARGET_DIR=/private/tmp/prism-world-target cargo clippy -p prism_world_network -p prism_world_sim -p prism_world_terrain --all-targets -- -D warnings
CARGO_TARGET_DIR=/private/tmp/prism-world-target cargo check -p prism_world_stream -p prism_world_residency -p prism_world_persist -p prism_world --no-default-features
```

核心 world policy 与 durable/scene tests 的既有验证包含 5 项 `prism_world` 测试、17 项 `SceneEcsHost` feature 测试、11 项 IO 测试、4 项 network 测试、6 项 simulation 测试、7 项 terrain 测试和 1 项 WGSL validator 测试；本轮持久化回归扩至 18 项并新增 2 项真实子进程锁回归，GPU CPU contract 已有 5 项通过。场景扩展仍不是完整流送到 ECS 的产品级端到端验收，不以这些数量代替完整覆盖。

万级 benchmark（本机单次 best，作为回归基线，不是跨硬件性能承诺）：

- stream demand solve：10,000 manifests，1.964 ms，5.09 M actions/s，3,366 次分配。
- residency schedule：10,000 records，0.368 ms，27.18 M records/s，3 次分配。
- world tick：10,000 items，1.818 ms，11.00 M items/s，3,374 次分配。
- persist CRDT merge：10,000 transactions，2.394 ms，4.18 Mtx/s；encode 7.642 ms，decode 10.246 ms。
- 以上由 `cargo bench -p prism_world_stream -p prism_world_residency -p prism_world_persist -p prism_world --benches` 实际执行，结果随 CPU、编译器和负载变化。

2026-10-15 受限环境中的严格 GPU 重跑曾失败：`No suitable graphics adapter found`（Metal 未返回 adapter）。同日经过用户允许的跨沙盒严格重跑已取得上文 Apple M2/Metal 的 pipeline、dispatch/readback 和 parity 通过证据；保留该受限探测失败记录，不将无 adapter 的运行算作通过。

### 地形原生网格与 Metal 实机回归

```text
CARGO_TARGET_DIR=/private/tmp/prism-world-kernel-target cargo bench -p prism_world_terrain --bench heightfield_bake
WGPU_BACKEND=metal PRISM_REQUIRE_GPU=1 CARGO_TARGET_DIR=/private/tmp/prism-world-target cargo test -p prism_world_gpu --features gpu --test terrain_draw --test gpu_parity -- --nocapture
```

`terrain_draw` 在 Apple M2/Metal IntegratedGpu 上将真实原生 Mesh 顶点/索引送入 indexed raster，启用背面剔除，读回 64² RGBA8 像素检查覆盖、UV、法线和高度。81 顶点、384 索引；全量顶点 2,592 字节、索引 1,536 字节；3×3 高度重烘只需更新含法线邻域的 5×5 顶点，五段上传共 800 字节到同一个 GPU buffer。局部上传与全量替换的全部 16,384 读回字节一致，远离脏区的像素不变。64 次测量、8 次预热的主机墙钟 encode/submit/执行/同步/map/copy 为 P50/P95/P99 = 590/839/882 µs；这是 debug、小型离屏测试，不是 GPU timestamp、产品帧时或完整 render graph 集成结论。严格 parity 测试同次重跑通过。

CPU release 基准为确定性平面、16 次预热、每项 128 次测量；下表单位 ns，为 nearest-rank P50/P95/P99，本机本轮结果，不代表跨硬件承诺或峰值内存。局部更新均为 9 个高度样本、25 个 mesh 顶点。

| 网格 | 高度载荷 / mesh 载荷字节 | 全量高度 | 全量 mesh | 局部高度 | 局部 mesh |
|---|---:|---:|---:|---:|---:|
| 129² | 33,282 / 925,728 | 119,583 / 184,458 / 222,334 | 265,625 / 412,167 / 494,209 | 167 / 500 / 1,500 | 333 / 541 / 667 |
| 257² | 132,098 / 3,686,432 | 323,416 / 994,083 / 1,312,042 | 761,709 / 1,625,917 / 2,512,334 | 500 / 2,708 / 3,416 | 292 / 1,292 / 1,417 |

新增行星/SDF release 基准同为 16 次预热、128 次测量，2026-10-15 在其他并行任务存在时实测；不可与上表推导为无干扰加速比。行星测试投影地球半径整面，origin 为面中心，允许误差 2,000 mm 以容纳整面远坐标的 f32 编码；它不是局部毫米精度用例，精度由上文独立回归验证。

| 操作 | 逻辑输出字节 | P50 / P95 / P99（ns） |
|---|---:|---:|
| 行星 mesh 129² 全量投影 | 925,728 | 1,783,125 / 6,531,834 / 9,624,625 |
| 行星 mesh 257² 全量投影 | 3,686,432 | 5,563,375 / 10,421,000 / 15,644,250 |
| SDF 33³ 全量 | 143,748 | 14,667 / 33,625 / 40,792 |
| SDF 33³ 局部 27 样本 | 143,748（另加 108 scratch） | 166 / 625 / 2,500 |
| SDF 65³ 全量 | 1,098,500 | 127,500 / 240,291 / 283,083 |
| SDF 65³ 局部 27 样本 | 1,098,500（另加 108 scratch） | 167 / 833 / 2,291 |

行星法线当前每顶点读取邻域并重新投影，耗时明显高于平面 mesh；数据保留 P99，不把该 CPU 烘焙耗时写成产品帧时目标达标。SDF 基准 source 为廉价轴对齐平面，实际复杂 source 成本需要另测；所有字节数为逻辑载荷，不是 allocator/RSS 峰值。

`cargo clippy -D warnings` 在包含 `prism_world` 依赖闭包时会进入仓库其他并行改动的 `prism_asset`/`prism_ecs`，当前被这些既有告警阻断；world 自身新增代码已按相应 crate 的局部检查验证，不能把依赖闭包失败误报为全 workspace 通过。

## 尚未交付的边界

以下项目仍需要独立硬件或端到端证据，不能用 policy primitive 冒充完成：

- M4 散布：行星/曲率/数据层/跨散布层规则、风与交互、逐键并发删除协议、索引增量编辑、蓝噪声目标、产品 proxy renderer 接线及大规模产品散布性能验收；`GpuPlanetPlacement` 已交付固定容量的 source/projection/finalize kernel、signed-i128 上传 ABI、compact/indirect 输出、Naga/device-free 回归，并已通过 Apple M2/Metal 上包含 placement producer 的完整键 deletion dispatch/readback 回归，但仍没有 CPU 位级 parity、scene/ECS decoder、renderer graph 或行星产品链路。已有内容中立有界 CPU/GPU 源散布、CPU/GPU 作者密度/种类场及各自可选刷子空间/有序索引、真实heightfield的CPU/GPU高度/离地带/坡度精确稳定过滤、独立身份的CPU二维精确贴地放置及其PPLD完整键持久删除、CPU/GPU二维贴地放置 kernel、CPU源/作者场/地形及GPU完整键删除与CPU持久文件恢复、独立 resident 密度 coverage/dither 与真实消费测试，不等于完整产品散布链路。
- M3：资产包解压、真正零解析的 mmap schema 视图、GPU 直传和产品级吞吐仍需实现或独立运行时验证；本批仅新增可选 mmap-backed container range source，保留受验证索引解析。共享动态优先级已接入但不提供抢占或跨队列无饥饿保证。任务池 Future/共享取消、固定 worker 及宿主原生 reactor → pollable coordinator → manifest/payload → `SceneHost` 已有真实文件端到端证据，同步兼容路径仍独立保留。离线缓存文件基准记录于 `prism_world_async_runtime_zh.md` 与 `prism_world_native_io_zh.md`，不能代替冷盘、资产包或产品负载验收。
- M6：多表示产品接线、Dual Contouring/Transvoxel 与混合 LOD 接缝、实际 renderer graph 接入；本批等分辨率 marching-tetrahedra 已通过 no-default 103 项和 `mesh` feature 106 项 CPU 回归，行星跨 tile 法线、高度局部增量投影和 `PlanetMeshPublication` 的跨 tile 原子 CPU 发布也已补齐，但其性能、混合 LOD 接缝和产品接线仍未验收。既有 heightfield 原子局部重烘、cube-sphere 增量投影、Q16.16 SDF 数据烘焙、native Mesh、离屏真实 GPU 绘制和 terrain-to-world-stream manifest 接缝不构成这些剩余缺口的完成证据。
- M7：SDF 与渲染/物理/GI 的多消费者真实运行时接入，以及大规模实时调度；CPU sparse grid、clipmap layout 与 GPU brick lookup contract 已交付，产品多消费者接线仍独立验收；simulation-to-world persistence 接缝已完成。
- M8：真实加密传输、多人 session、跨 sector handoff 的网络系统端到端一致性验证；engine-neutral session/authority/handoff seam 已完成。
- GPU proxy 已有 Apple M2/Metal parity、驻留 compact/真实间接绘制、sector/coordinator/宿主生命周期及限定 release P50/P95/P99 证据；物理显存峰值、多厂商/后端、完整产品性能与产品 scene/ECS decoder/renderer graph 集成仍未验收。

这些边界已有明确契约和可运行基础；只有补齐对应真实运行时证据后，才从本节移除。

## 不变量

- world 不复制 scene 内容；可变权威状态是 override delta，派生索引/代理/SDF/PCG 结果可删除重建。
- policy core 保持 `no_std + alloc`、`forbid(unsafe_code)`；文件系统和 GPU 能力只在明确 feature/provider 边界出现。
- 所有日志、排序、PCG、HLOD 和 readback contract 都有稳定顺序或稳定 key。
- 任何声称 GPU 或性能完成的结论必须同时记录 adapter、pipeline、dispatch/readback、P50/P95/P99、上传字节和显存/资源预算。
