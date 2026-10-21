# World streaming placement runtime slice

本切片覆盖 world streaming placement 的 engine-neutral 契约与可选原生 RHI 适配：

- `prism_world_runtime::WorldCellStore` 维护 resident cell 与 `PersistentId` 的稳定索引。索引支持多 cell、跨 cell `relocate`、cell replacement/eviction，以及有明确 byte/cell/object 上限的 canonical restart snapshot。
- `StableRefSceneHost` 是 `SceneHost` 的生命周期适配器。stable-ref decoder 必须由产品 scene 提供，不假设 cell payload 格式；decode/store validation 失败不会调用 scene host，scene host load/reload 失败保留旧索引，eviction 只有在 host 成功后才删除索引。
- `prism_world_gpu::PlacementRenderGraph` 生成确定性的多 cell `Upload → Project → Compact → DrawIndirect` pass plan，并显式声明 source、placement、compact、indirect、scene 的资源读写关系。相同 cell/revision 或同一 cell 的冲突 revision 会在发布 graph 前拒绝；`PlacementGraphEncoder` 将已验证的 plan 逐 pass 交给 renderer-owned consumer，失败返回精确的 pass index、cell/revision 与 stage，且不分配、不重排、不合并 pass。
- `GpuHeightfieldPlacement::encode` 不再依赖 encoder `clear_buffer`：独立 `reset_main` compute kernel 只原子清零 `indirect[1]` 与 overflow status，保留 indirect 的 vertex/first-vertex/first-instance 三个静态字。非空请求随后执行 source、project、field/compact、finalize；空请求只执行 reset，不伪造 `[1,1,1]` 候选 dispatch，也不记录空 draw。
- `GpuHeightfieldPlacement::registration_snapshot` 暴露同一原生设备上的 11 个 buffer handle、精确逻辑 byte size、reset stage、候选阶段、request generation 与 candidate count。snapshot 只克隆句柄，不分配或复制 GPU 内存；空请求以 `active=None` 表达 reset-only。它是生产 RHI 注册的完整输入面，不冒充持久注册所有权或资源退休机制。
- `prism_world_gpu::build_placement_render_graph` 将 renderer 已注册的真实 RHI buffer、scene texture、pipeline 与 bind group identity 降低为 `prism_render_graph::RenderGraph`。active cell 记录 `reset → source → project → field/compact → finalize` 五个 compute pass 和一个 indirect raster pass；empty cell 只记录一个 reset compute pass。reset hazard 只声明 indirect/status 写入，候选资源由后续阶段声明；scene attachment 通过 SSA handle 串接，多 cell 按稳定 cell/revision 顺序记录。零字节 buffer、零维 dispatch、候选数量与 work presence 不一致、未按四字节对齐的 indirect offset 与同 cell 重复输入都在返回 graph 前 fail-closed。

验证覆盖：runtime 多 cell load、跨 cell resolve/relocation、reload 失败保持旧状态、eviction 后不可解析、磁盘 restart round-trip；GPU plan 的排序、hazard chain、重复/冲突 revision rejection，以及 encoder 对每个 pass 的顺序投递和精确失败定位。Null RHI command-recording 回归分别核对 active 的五次 compute dispatch 加一次 indirect draw，以及 empty 的单次 reset dispatch；Naga 闸门检查 reset 的两个 read-write storage binding、无 uniform、核心 u32 atomic ABI。2026-10-20 在独立离线 target 上，terrain-scatter 库 135/135、WGSL 9/9、`cargo test -p prism_world_gpu --features gpu --lib --offline` 119/119、`cargo test -p prism_world_runtime --lib --offline` 104/104；沙盒外 Apple M2/Metal IntegratedGpu 的 `heightfield_placement` 真实 GPU 集成 10/10 通过。这是历史验证；原生 RHI 注册、提交与退休的后续交付见下节，不能将其扩大为完整产品 renderer 交付。

## 原生 RHI 适配与生命周期

`placement_rhi` feature 提供 `PlacementRhiBridge`，从原始 `GpuContext` 的同一
adapter/device/queue 元组采用一个私有 RHI table，不请求第二个设备、不替换宿主错误或
device-lost callback。注册消耗 producer 并形成稳定 wrapper；不同 context 或同 context 的
另一 bridge 的 producer/scene/draw wrapper 都会在录制前拒绝。

每个 producer 缓存 native buffer、compute pipeline 与 bind group 的注册。同一 native
buffer 在后续请求中复用同一 `BufferId`，防止通过重复注册逃避图的物理别名校验。注册只释放
table entry，不对外部 native buffer 调用 `destroy`；部分 stage 注册的 unwind 路径也有 RAII。
每个 scene 持有 color texture/view，draw pipeline 持有其 pipeline/layout/group layout。
调用方提供 shader，固定 group 0 绑定为 records、terrain samples、attributes、compact；
删除流绑定 filtered compact。只支持单层、单 mip、单采样 2D color target 和无 vertex buffer、
无 depth 的单 color attachment draw。raw shader IDs 必须来自 `bridge.device()`；这些逃生口
不认证任意调用方的资源来源，也禁止调用方提前销毁 native inspection 返回的资源。

`begin_frame → add_heightfield/add_deletions → record → submit` 的 move-only owner 保持
producer 请求的共享借用直到真实提交或取消，提交前不能覆写请求；两个 compile-fail doctest
分别覆盖 declaration 和 recorded 阶段。提交后可以按同一原始 queue 的顺序更新下一请求，
不必 CPU 等待 GPU。声明或 recorded owner 被丢弃时不提交，并先丢弃命令再释放 graph allocations。

native encode 后、进入真实 submit 前，graph allocations 和 frame resources 一起转入有界 pending 队列。
completion callback 只写原子信号，不捕获资源图；`poll` 非阻塞推进，`wait` 等待指定同桥 receipt，
精确 `wait` 成功即确认该 receipt 及前序提交完成，不依赖可能覆盖外部更晚提交的回调；
`drain` 成功后确认全部 pending 并回收。receipt 的 `is_complete()` 只有在 allocation/registration pin
真正回收后才为真，不把“submit 返回”当完成。丢弃 receipt 不等待也不丢失 pending 所有权。
提交期间宿主回调 panic 时，pending 保留资源；未取得 submission index 的包在 `drain` 时等待
整个原始 queue，而不是误走取消释放。桥析构只在 shutdown 时 drain，并捕获回调 panic；
30 秒等待失败或 shutdown panic 时保留未完成资源，不能提前 destroy；panic payload 也保留，
以免任意宿主 payload 的析构再次 panic。
这表示资源不再 in-flight，不表示 device-loss 时的语义执行成功或自动恢复。

pending vector 在构造时按上限预留，退休原位稳定压缩为 `O(in-flight frames)`，不反复移除导致
平方搬移。注册集合固定于 producer 的容量与 stage 数；请求之间只更新 dispatch，不重复注册
持久 compute 对象。每个非空 frame/stream 仍创建 draw bind group，并有 graph/命令 CPU 分配；
没有宣称零分配或产品 P99 达标。候选评估、删除与绘制不走 CPU readback；测试 readback 仅用于观察。

空删除请求也记录 upstream reset → 16-byte indirect copy → deletion count reset 三个 pass，
不录制候选 filter/draw，避免上一请求的 instance count 残留。当前图的首个非空 raster 对
scene version 0 清黑，同帧后续 cell 通过 SSA Load 累加；不能用于保留已有颜色的任意叠加。
全部为空时没有 raster，保持已有颜色；宿主若需清空旧画面须自行清屏或替换目标。

2026-10-22 主工作树复验：runtime lib 183/183、GPU feature lib 136/136、placement_rhi lib
158/158、借用约束 doctest 2/2、WGSL 11/11，以及沙盒外 Apple M2/Metal IntegratedGpu 的
RHI 实卡回归 7/7（无忽略）。后者包括 submit/wait 宿主 callback panic 注入和恢复。
这是共享 dirty checkout 验证，非 clean-HEAD 隔离验证；严格 Clippy 仍有 6 项已记录告警。
完整命令和证据边界见 `prism_world_gpu_heightfield_placement_deletions_zh.md`。
产品 `prism_render_scene` mesh/material/unified visibility 消费、scene/ECS decoder、持久删除的
自动产品消费、多厂商和产品负载的性能/显存/功耗验收仍独立于此适配层。

删除键更新已改为最小单一连续 GPU 槽位区间上传：追加只传尾部、相同前缀截短不传键，
非相邻修改合并一次写入，重新增长会覆盖 stale 尾部。仍须先 submit 再改写；RHI frame
借用对此顺序保持约束。实卡 deletion 转换回归 4/4；确定性上传量、复杂度和未优化 dev
专项基准见上述删除文档，不将其视为产品性能验收。

## 删除流上游驻留与失败恢复

`RegisteredPlacementDeletions` 的相同 placement recipe 可以复用已成功提交的上游输出。
缓存证明属于同一 producer/table 与 request generation，仅结构等价的成功写入可延续，
不依赖 cell revision 或 hash 相同。声明/录制取消不发布；A → B → A 未提交的请求转换
不能复活旧证明；空请求仍执行双计数复位。raw graph builder 不使用此优化。
命中图保留 indirect copy → deletion count reset → full-key filter → indirect draw 四步，
只跳过五步 upstream 重建，所有资源/ABI/别名校验不变。

`PlacementRhiRecorded::reused_upstream_count()` 可检查实际命中数；`try_submit()` 捕获
native validation/internal/OOM error，返回 `PlacementRhiError::Native`，而 `submit()`
保留遇错 panic 行为。进入 native encode 前先撤销旧证明，三类 scope 均无错且 submit
返回后才重新建立；因此热缓存失败、宿主 callback 展开也不能保留旧证明。失败包仍由
pending 保活，无可靠 index 时通过 whole-queue `drain` 退休；退休不等于失败工作的
语义执行成功。scope 使用 RAII 且逆序 pop，不覆盖宿主 callback；feature 显式启用
`wgpu/std` 保证线程隔离与 unwind 检测，不依赖宿主偶然启用。native scope pop 不等待
GPU，复用仍只依赖同 queue 的先后提交，不增加逐帧 CPU fence。

通过 native inspection 逃生口改写上游输出前必须调用
`RegisteredPlacementDeletions::invalidate_resident_upstream(&mut self)`，并在下一次
重建提交前排入外部写入；不能在已重建之后再提交旧写入。mutable borrow 同样受 frame
借用保护，第三个 compile-fail doctest 覆盖该约束。适配层无法检测任意外部 native 改写。

专项 `placement_rhi_reuse` 基准在沙盒外 Apple M2/Metal 上验证固定资源的 10,000 候选、
128 删除键、单槽 56-byte 更新；每模式 8 次预热和 64 次采样，并对两个方向的真实计时
输出核对完整键与每 source-slot 像素。dev profile 的 submit+精确 wait P50 为驻留
1.052 ms、强制重建 2.035 ms；它包含 native encode/error scopes、上传、GPU 工作和退休，
不是纯 GPU 时间或产品性能承诺。完整 P50/P95/P99、逻辑容量、命令和验收范围见
`prism_world_gpu_heightfield_placement_deletions_zh.md` 的驻留复用章节。

本轮最终复验（2026-10-22，共享 dirty checkout）：runtime lib 183/183、GPU feature
lib 139/139、placement_rhi lib 163/163、compile-fail doctest 3/3、WGSL 11/11，
no-default-features lib check 通过；沙盒外 Apple M2/Metal
实卡 deletion 4/4、RHI 13/13，无忽略。RHI 新增覆盖精确 recipe 失效、取消/ABA、
显式 native 写入失效、冷热缓存 validation 错误、热缓存 submit panic 后强制重建，以及
同 device 跨线程 error scope 隔离。上文 136/158/7 等是初版历史计数，不是本轮结果。
原 `admitted_layers` 编译错误未复现，本次不改 runtime；复现命令、基准口径及产品
未覆盖边界见删除文档的“本轮最终验证”。本轮不宣称整个 World 系统完成。

同日继续补验：RHI 实卡 **14/14**，新增/补强无 CPU 等待的非空→空→恢复→复用
双间接计数/完整键/像素快照，以及同 table、同 cell/revision/初始 cache key 的独立
producer 交错提交隔离；runtime **183/183**、GPU feature **139/139** 再次通过。
纯提交 `764aeb7da` 的 tracked-tree 独立源码构建尝试在 workspace 解析阶段被
`prism_render_scene` 请求不存在的 `prism_render_material/bevy` feature 阻断，
尚未进入 World 编译，故不宣称 clean-HEAD 验收通过。未修改其他模块；复现命令和
证据范围见删除文档的“无等待空请求恢复与 producer 隔离补验”。
