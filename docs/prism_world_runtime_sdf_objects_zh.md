# World Runtime：对象 SDF 局部合成与 GPU 发布

本切片实现设计 §17 中对象编辑到局部 brick 重烘、再到 GPU 查询镜像的路径。
`WorldSdfObjectStore<T>` 与 `WorldSdfLocalVolume` 可用于 `no_std + alloc`；
`WorldSdfGpuObjects<T>` 由显式 `gpu_proxy` feature 启用。它们维护可丢弃的派生数据，
不写存档、不复制 Scene 作者态，也不为项目规定实体类型或几何来源。

## 数据与效果契约

- 对象由稳定 `u64` identity、世界毫米平移、闭区间 local AABB、不可变 `Arc<T>` source
  组成。AABB 必须包含所有不是 `i16::MAX` 的影响；source 的内部可变性不能绕过 upsert
  改变采样结果。旋转、缩放或任意几何可由 source/evaluator 表达，并非本 owner 的矩阵 API。
- 全场共用 `distance_scale_mm`：返回的一个 i16 单位对应相同物理距离，各 mip 不重新缩放。
  世界/local 采样位置以 i128 计算，负坐标使用 Euclidean floor；brick 坐标超出 i32 则拒绝。
- 每个 mip 独立在格点求值，而不是从低 mip min-reduce。粗 mip 可能漏掉亚体素几何；
  不提供保守占据、几何细节保持、Lipschitz 或 sphere-tracing 安全步长保证。
- 每个 brick 保存排序的 contributor identity。替换、移动和删除都会标脏旧/新 AABB 在所有
  mip 的覆盖。重烘从全部存活 contributor 取 min，所以删除遮挡物可恢复被覆盖对象，
  不是试图从 min 场减去旧值。
- 无 contributor 的 brick 被移除，查询语义为 `Unknown`，不是空空间。仍分配的 brick 内，
  处于所有对象影响之外的样本是 `i16::MAX`。不存在隐式已知空白的背景场。

`WorldSdfLocalVolume` 是可选的现成 source：拥有 X-fastest 的规则 i16 体积、每轴至少两个
样本、正毫米 spacing，六个边界面均为 MAX。构造不复制/分配 sample vector，只在常量空间
检查几何、长度、预算和边界；样本预算不包含传入 Vec 多余 capacity。查询 O(1)、零分配，
执行 i128 三线性插值，三轴合并后仅做一次最近取整，半数远离零。边界/体积外为 MAX，
内部 MAX 角点数值参与插值，不表示缺失。这是量化局部场，不自动校准几何距离。

## 事务、任务与调用顺序

```text
upsert/remove → prepare(budget) → detached evaluate → commit
                                                     ↓
                                         CPU ready + resident GPU dirty
                                                     ↓
                                    update_focus / flush_dirty → submit_queries
```

`prepare` 选取有序 dirty suffix，按数量、求值次数上界和 scratch 字节准入，预分配全部
求值缓冲，只克隆该区域的 source Arc。预算不足返回错误，不偷偷缩小工作。`evaluate`
不分配；错误、取消（evaluator 返回错误）或 unwind 都不改变 owner。宿主可把 detached
work 交给线程池，但必须自行限制同时持有的任务及 source 资源总额。

每个 job 带 owner identity 和全局 revision。任何成功编辑或非空 commit 都使所有旧 job
失效，包括不重叠的 job；同版本不同 owner 也拒绝。并行求值不代表允许不经重新 prepare
就连续提交多个同版本结果。缺失 identity 的 remove 和空 commit 是 no-op；revision
耗尽时实际编辑/非空发布在状态变化前拒绝。

CPU store 的 `grid()` 只供诊断，dirty 期间可能混合新旧结果；`ready_grid()` 在全部 CPU
重烘完成前返回 `PendingBake`。GPU owner 不暴露可变 store/ring，commit 先验证 CPU job、
准入 resident changed-coord scratch、原子标记 GPU dirty，再执行无可恢复失败的 CPU
发布。因此 GPU mark 失败不会留下已发布的 CPU 新版本。窗口外变化不标记 GPU，之后进入
窗口时从最新 CPU grid 上传。CPU 未 ready 时连静止 focus/flush 都拒绝；CPU ready 后
仍有 GPU dirty 则查询拒绝。所有 resident 上传完成才开放查询。

查询通过同一 queue 编码 lookup、consumer 和立即 submit；consumer 必须在回调内消费
或复制结果，不能把共享 result buffer 当成跨提交快照。成功上传表示已入队，不是 GPU
完成。旧查询的独立 witness 可在多轮后续上传后才读回，测试验证其旧版本不被覆盖。

## 预算与成本

CPU limits 分开限制 objects、covered ∪ still-materialized bricks、object-brick links、dirty
数量、一次编辑 old+new 原始覆盖和 edit scratch。移动后尚未删除的旧 grid 仍收费；满容量
时可能必须先 remove+bake，再插入别处。重烘按 brick、保守 contributor×sample 求值上界、
描述符/source snapshot/sample scratch 收费；删除时也计入底层 parent dirty 临时载荷。
Vec 的可恢复 allocation/budget 错误不修改状态；BTree/Arc OOM 仍按 Rust fatal allocation
策略处理，不承诺进程 OOM 回滚。上述逻辑载荷不等于 allocator/RSS 峰值，也不包含 source
payload、树节点开销或所有未收集 detached job 总和。

编辑只枚举受影响覆盖、更新 contributor 索引，但 staging 会复制当前 dirty 列表；成本
不纯粹只与本次 AABB 有关。重烘不会扫描所有远处对象；树查询仍有索引规模的对数成本。
不承诺有序 suffix 在持续编辑下的公平性，宿主必须控制编辑速率和服务预算。

GPU owner 继承现有 **XZ square / 每 mip 单 Y 平面** ring，不是三维 clipmap。CPU/GPU
origin、spacing、brick edge、mip 数必须相同，构造前校验。GPU commit scratch、scheduler
mark scratch、每次上传与 query 容量独立准入，不构成跨消费者累计帧预算。每个 present
brick 上传 `32 + 4*ceil(edge³/2)` 字节，删除仅 32 字节；固定 ring 的逻辑 buffer 字节由
`allocated_buffer_bytes` 观测，不含驱动和 queue staging。

## CPU release 微基准（2026-10-21）

4³ local volume，spacing 25 mm，edge 4、3 mip，对象 x=0↔25 mm 移动；每档 8 预热、
64 样本，setup 不计时。单次 job 4 个 changed bricks；无重叠时 return/outward 的求值上界
为 192/256、逻辑 scratch 936/1120 bytes；8 重叠为 1536/1600、2784/2968 bytes。
return 包含一个删除，不能与 outward 当成同一种工作量比较。基准断言两个方向各自的
逻辑工作量不随 4096 个远处对象增加，且从不调用远处 source evaluator。

下表每格为主机 CPU ns/job，P50/P95/P99（nearest rank）；本机同时存在并行任务，保留
长尾，不当作无干扰 A/B、产品帧时、GPU timestamp 或多硬件性能门槛。

| 远处对象 / 局部重叠 | edit | prepare | evaluate | commit |
| --- | --- | --- | --- | --- |
| 0 / 1 | 1000/2666/374292 | 250/625/1875 | 1708/5167/6959 | 250/708/2917 |
| 4096 / 1 | 2250/2625/6167 | 708/792/1375 | 5000/5583/87625 | 584/834/3541 |
| 4096 / 8 | 2375/8000/3931834 | 1250/2792/5257500 | 15750/82666/583542 | 625/3584/44917 |

## 验证入口

```bash
export CARGO_TARGET_DIR=/private/tmp/prism-world-objects-verify.Q4phmr
export CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0
cargo test --offline -j1 -p prism_world_runtime --lib
cargo check --offline -j1 -p prism_world_runtime --no-default-features --lib
cargo test --offline -j1 -p prism_world_runtime --features gpu_proxy --lib
cargo test --offline -j1 -p prism_world_gpu --features gpu --lib
cargo clippy --offline -j1 -p prism_world_runtime --features gpu_proxy --lib --test sdf_objects_gpu --test sdf_objects_performance --no-deps -- -D warnings
WGPU_BACKEND=metal PRISM_REQUIRE_GPU=1 cargo test --offline -j1 -p prism_world_runtime --features gpu_proxy --test sdf_objects_gpu --test sdf_focus_gpu --test sdf_clipmap_gpu -- --nocapture --test-threads=1
cargo test --offline -j1 -p prism_world_runtime --release --test sdf_objects_performance -- --ignored --nocapture
```

当前记录：默认 runtime lib **183 passed**（其中 object 14、local volume 21），
`gpu_proxy` runtime lib **203 passed / 7 ignored**，`prism_world_gpu --features gpu`
lib **123 passed**，no-default check 通过，release benchmark **1 passed**。限定范围
Clippy（runtime lib、对象 GPU/性能集成测试，`-D warnings`）通过；依赖
`prism_render_graph` 仍打印一个既有 unused-import warning，但未纳入 `--no-deps` lint
范围。Apple M2 / Metal / IntegratedGpu 严格实卡回归：对象 GPU **6/6**、focus **7/7**、
clipmap 上传 **4/4**。受限 sandbox 中同一对象测试因无 Metal adapter 全部未能创建设备，
之后在获准的宿主环境通过；不把 adapter 不可用运行算成通过。GPU 专项测试严格要求物理 adapter，包含重叠恢复/移动/删除、
多 mip、延迟 witness、部分 CPU/GPU readiness、预算拒绝、已有 dirty 时 mark 失败、stale
job、upload/roll 失败、查询容量/mip、off-window 再进入、真实 volume 与 geometry mismatch。

## 独立验收边界

此切片不自动观察 Scene/ECS 对象变更，不提供对象任务池 wrapper、产品相机 tick 装配、
device recovery、真三维窗口、保守 mip、solver/render/GI/nav consumer 或多消费者累计
预算。已有独立 ECS sampling bridge 不是 object-edit observer，也不消除上述边界。
异步 device loss、wgpu validation panic 和进程 OOM 不属于可恢复 Result 事务保证。
这些产品能力需要各自的调用链和实机证据，不能由此局部派生缓存切片宣称全部完成。
