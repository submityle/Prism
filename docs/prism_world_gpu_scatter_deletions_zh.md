# GPU 完整键散布删除

日期：2026-10-16。此处描述内容中立的 GPU 删除切片，不是完整 world 产品验收。
持久化格式、权威字段、真实文件回放与撤销边界见
`prism_world_scatter_deletions_zh.md`；GPU 不拥有第二份持久状态。

## 上游所有权与完整键

`GpuScatterDeletions<U>` 接收并独占已有的 `GpuScatter`、`GpuScatterField`
或启用 `terrain_scatter` 的 `GpuScatterHeightfield`。producer trait 是 sealed，
不能注入任意外部 buffer，也不暴露上游的可变引用或单独 encode 入口。
固定绑定与实际同一 logical device 绑定，不以语义 cache 相等替代资源身份。
`GpuContext` clone 共享私有 `Arc<()>` 身份，独立获取的 context 则始终不同；
不依赖 wgpu Device 的跨 Instance handle 相等。首轮实机回归曾发现独立 Instance
局部 ID 相等导致错误资源通过预检、最终触发 wgpu 断言；现已用该身份修复，并
覆盖独立 context 拒绝与 clone context 实际执行两类回归。

请求分别为 `GpuScatterSourceRequest`、`GpuScatterFieldRequest`、
`GpuScatterHeightfieldRequest`，将源、场、地形及其工作上限分层表达；
传入现有 `ScatterDeletionSet` 和独立删除工作预算。空删除集合也走完整上游。

shader 消费上游最终 compact 的 GPU count 前缀，绝不遍历中间 accepted 标记
使已被密度/地形剔除的点复活。对每个存活 slot 从源记录恢复 grid，并加入
kernel、seed、recipe version、salt、spacing；二分搜索比较完整结构键。
64 位 seed/salt 以 high/low u32 比较，XYZ 有符号 i64 将 high word 翻转 sign bit
后比较；不是 key hash、浮点位置、短 ID 或字节字典序。

只删除、不移动、不补点，保持源 key、整数位置、最小距离和任意 XYZ 分区集合。
源/场/地形缓存键全部保留；删除集合含请求外、其他 namespace 和已被上游剔除的键，
仍是完整缓存依赖。缓存键另含 CPU/GPU deletion revision、key ABI revision
和不含 instance count 的 draw 参数。结构比较不是碰撞摘要或指针身份。

## 固定资源、预算与失败语义

令 C 为上游 primary 容量、Dcap 为删除容量。新增逻辑 GPU buffer 为：

| buffer | 字节 | 用途 |
|---|---:|---|
| full keys | `52*max(Dcap,1)` | 13 个 u32/键，显式打包 |
| final compact | `4*max(C,1)` | 原始源 slot |
| uniform | 128 | N、D、dispatch、完整 namespace |
| indirect | 16 | 保留上游 draw 参数，重置/写入最终 count |

合计 `52*max(Dcap,1)+4*max(C,1)+144`。`max_buffer_bytes` 约束**整个链**，
包括调用方移入的已有上游，不只是新增部分；失败时上游对象因按值传入而被消费，
不承诺构造失败返还对象。各 buffer/storage/dispatch/workgroup 设备限制先检。
pass 需要六个 storage binding、一个 uniform、64 lanes、260B workgroup storage，
无 optional device feature。上下文须显式申请六个 storage binding，普通四 binding
downlevel 默认不变。

构造时 fallible 预留 `Vec<[u32;13]>`；`max_scratch_bytes` 按实际 capacity×52
复查。只有改变的集合重新打包，不在 encode 中分配/打包。共享删除快照在当前
aarch64 的原始 key 载荷为 56D，不是这个 52D scratch；快照、场/地形数据、旧请求、
Arc/Vec 头、driver/allocator 开销另计，不把逻辑载荷称为物理 VRAM/RSS。

删除工作准入使用 primary 上界 N，而不是回读未知的 GPU 幸存数 M：

- `max_input_slots >= N`，包括 D=0 的遍历和 compact 成本。
- `max_keys >= D`，另有固定 Dcap。
- `max_key_comparisons >= N*ceil(log2(D+1))`，checked 计算。

一次 comparison 最多比较固定 13 words，实际工作不超过 M 次搜索加 N lanes 调度；
不会因 D=0 的 comparison 数为零而绕过 traversal 预算。源最小距离邻域、作者场与
地形预算各自先检，不由删除预算替代。

`validate_request` 不上传/修改。`write_request` 在第一次 queue write 前预检整个
链；返回的校验/预算错误保留旧请求、缓存键、upload stats 和旧输出。设备丢失、
wgpu OOM、外部通过 raw GPU handle 别名任意写入不属于可恢复错误保证。
更改请求 uniform 上传128B；更改非空集合上传52D；相等集合不重传键。空集合不
清旧 key buffer，因为 D=0 不读它。stats 只计删除级，不含上游或 driver staging。
完整等价比较最坏 O(D+brushes+surface samples)，不是恒定时间 API。

## encode 与消费

每次 encode 校验已上传的上游 generation/count/draw 参数，再依次记录：

```text
owned upstream encode → copy upstream draw args → clear own count → deletion pass
```

没有 CPU key 扫描、上传、提交、等待、map 或回读，也没有本实现的 owned 资源分配；
不承诺 wgpu/driver 记录命令的内部零分配。每工作组前缀扫描后最多一次全局 atomic
reservation；compact 跨工作组顺序未规定，仅最终 count 前缀有效。

`records()` 和可选 `attributes()` 是**未修改的上游数据**，其 accepted flag 表示
删除前状态，不能用它绕过删除。下游必须使用本对象的 `compact_indices()` 和
`indirect_args()`，按 compact 源 slot 读取对齐的源/属性。`candidate_count()`
是 primary N，不是最终 survivor count；不为 CPU 提供隐式回读的 survivor 数。
重复 encode 重置 count，不累加旧实例。

当前请求 encoder 必须先 submit，再 write/encode 下一请求；顺序提交无需主机
等待。queue write 在提交前执行，不是 encoder 内的位置命令，不能先记录多个
不同请求后一起提交。raw encoder API 无法机械跟踪 submit epoch，这是明确的
调用方契约，不声称已经自动检测该误用。

## 验证入口

```text
WGPU_BACKEND=metal cargo test --offline --locked -p prism_world_gpu --features terrain_scatter --test scatter_deletion --test scatter_deletion_composed --test wgsl_compile -j1 -- --nocapture --test-threads=1
WGPU_BACKEND=metal cargo bench --offline --locked -p prism_world_gpu --features gpu --bench scatter_deletion -j1
```

设备测试严格拒绝无设备、CPU adapter、不匹配 WGPU_BACKEND，不以 skip 算通过。
source/field/heightfield 分别和公开 CPU oracle 比较完整 key/位置/属性；
pixel 用例直接绑定 GPU records/最终 compact 并 `draw_indirect`，不上传 CPU 顶点、
不使用 CPU count 发直接 draw，完成绘制后才回读像素。

2026-10-16 Apple M2、Metal、IntegratedGpu、无额外必需 feature，直接编译原始
源码的隔离 harness 定向 **139 项全部通过**，0 失败/忽略：库100、删除源7、
删除组合8、既有源5、作者场7、地形8、WGSL4。删除库新增12项覆盖完整key显式
打包/结构序、全部缓存依赖、设备身份、generation/count/draw guard、容量/实际
scratch、设备限制、checked工作和二维dispatch边界。设备回归另覆盖空/部分/全删、
63/64/65/129尾部、请求外/其他namespace键、signed XYZ、任意分区、最小距离、
整个上游records/attributes前缀不改写、预算拒绝保留旧结果、固定资源及无wait顺序
提交。另有源预检 epoch 到达 u64 上界时拒绝回绕、保持语义 key 的单测。
WGSL以空optional capabilities验证，禁止float和64位标量/向量/atomic。

隔离 harness 使用路径依赖并指向本仓库原始 lib/test/bench，不是另一份实现，
但不包含正常 package 的全部开发依赖或整个 workspace。正常 workspace 的
`cargo check --offline --locked -p prism_world_gpu --lib` 在 no-default 与
`terrain_scatter` 两配置均通过。冻结后隔离 harness 的 all-target Clippy 以及
正常 workspace `terrain_scatter` 的 GPU lib 依赖闭包 Clippy 均通过 `-D warnings`。
完整 workspace、其他后端和多厂商硬件并未据此验收。

## 限定 release 基准

Apple M2/Metal，8预热+64样本，nearest-rank的1-based索引32/61/64，单位µs。
每个样本完整运行源生成+删除，而非单独删除pass；源radius=0，10k/100k个候选，
按原序号每100个删除前0/25/100个。encode列为encode+finish；complete列从encode
开始至submit+有界wait完成。生成CPU oracle、构造/上传、回读、逐完整key/位置校验
及析构不在计时中；每个预热和测量样本在计时外全量对拍，不只核对count。

第一轮：

| N | D | encode P50/P95/P99 | complete P50/P95/P99 |
|---:|---:|---|---|
| 10,000 | 0 | 182.542 / 304.083 / 1088.583 | 949.333 / 6382.375 / 16988.333 |
| 10,000 | 2,500 | 181.333 / 299.167 / 791.625 | 977.958 / 1684.042 / 3198.209 |
| 10,000 | 10,000 | 128.875 / 223.292 / 639.709 | 794.208 / 1418.417 / 5619.667 |
| 100,000 | 0 | 107.917 / 683.542 / 6322.000 | 1886.792 / 5871.584 / 8034.666 |
| 100,000 | 25,000 | 110.375 / 536.458 / 3311.792 | 2383.542 / 5445.625 / 13528.584 |
| 100,000 | 100,000 | 161.625 / 407.375 / 3359.708 | 2559.500 / 7470.916 / 8833.458 |

冻结源码后的记录复测（另一次中间运行的日志未完整保留，不用于数字结论）：

| N | D | encode P50/P95/P99 | complete P50/P95/P99 |
|---:|---:|---|---|
| 10,000 | 0 | 27.292 / 44.125 / 79.334 | 454.375 / 528.583 / 751.125 |
| 10,000 | 2,500 | 25.125 / 31.166 / 43.875 | 521.250 / 578.708 / 6025.417 |
| 10,000 | 10,000 | 18.916 / 25.375 / 32.458 | 529.625 / 555.500 / 621.458 |
| 100,000 | 0 | 59.209 / 76.583 / 94.208 | 1483.458 / 1528.875 / 1637.167 |
| 100,000 | 25,000 | 57.500 / 156.375 / 330.167 | 2004.875 / 2523.542 / 7724.459 |
| 100,000 | 100,000 | 26.333 / 38.291 / 48.958 | 1950.416 / 2029.791 / 2207.042 |

上述两组完整记录共864次dispatch（含预热），每次均逐完整key/位置对CPU。
资源成本两组相同：

| N | D | 幸存 | 整链逻辑GPU B | 删除级逻辑GPU B | key scratch B |
|---:|---:|---:|---:|---:|---:|
| 10,000 | 0 | 10,000 | 720,404 | 40,196 | 0 |
| 10,000 | 2,500 | 7,500 | 850,352 | 170,144 | 130,000 |
| 10,000 | 10,000 | 0 | 1,240,352 | 560,144 | 520,000 |
| 100,000 | 0 | 100,000 | 7,200,404 | 400,196 | 0 |
| 100,000 | 25,000 | 75,000 | 8,500,352 | 1,700,144 | 1,300,000 |
| 100,000 | 100,000 | 0 | 12,400,352 | 5,600,144 | 5,200,000 |

首次请求上传源192B、删除uniform128B及keys52D，构造时另初始化源和删除indirect
各16B；计时样本无请求上传。大D增加完整键存储和二分搜索成本，即使最终全部删掉
也仍需源生成与查找。两轮实测中的调度/队列尾延迟如实保留，不把快的一轮当无干扰
A/B或性能达标；这些是主机墙钟，不是GPU timestamp、单pass纯执行时间、产品帧时、
物理VRAM/RSS或完整密度/地形链的性能结论。

这不是产品 scene/ECS/renderer graph 集成、逐键并发删除协议、贴地投影、物种 mesh
实例生成、风/交互或蓝噪声验收。持久恢复仍由 CPU binding/既有日志提供；项目取得
恢复的完整集合后调用本请求接口，GPU 并不直接解析磁盘日志或拥有编辑权限策略。
