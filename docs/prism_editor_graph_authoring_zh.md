# 通用节点图编辑层与 Loom 面板

本切片在现有 typed graph 内核上实现可执行编辑协议和真实 Loom 子树。节点贡献、端口类型键和 opaque payload 均由宿主定义，没有材质/VFX/动画或 gameplay 类型，也没有 Bevy 依赖。

## API 与接线

内核公共 API：

- `GraphEditor::new(GraphAuthoringLimits)`、`execute(&GraphCommand)`、只读 `graph()`、`position/positions`、`next_ids`、`revision` 和历史计数。
- `GraphEdit`：带显式 ID 的 CreateNode、DeleteNode、MoveNode、SetPayload、Connect、Disconnect。
- `GraphCommand`：Edit 批、Undo、Redo；提供 `encode/decode`。
- `GraphNodeDraft/GraphPortDraft/GraphPoint` 和 `GraphAuthoringError`；原内核的 `EdgeId` 也已导出。

UI 共享 roots 由 main 所有，本切片不修改。所需普通依赖为：

```toml
prism_editor_graph = { path = "../prism_editor_graph" }
```

UI lib 接线如下，无新增 dev-dependency：

```rust
mod graph;
mod graph_view;
pub use graph::{
    GraphNodeTemplate, GraphPanelAction, GraphPanelController, GraphPanelError,
    GraphPanelLimits, GraphPanelViewport,
};
pub use graph_view::{
    BoundGraphPanel, GraphNodeVisual, GraphPortVisual, GraphWireVisual,
    GraphPanelViewStats,
};
```

宿主构造 GraphEditor 与 GraphNodeTemplate 列表，交给 GraphPanelController；将 `build().element()` mount/update 到 Loom。按实际 painted key 将 UiEvent 交给 `handle_event`。成功的任意操作都会更新 publication identity，旧 view 与其他 controller 的 view 被拒绝。

## 原子命令、身份与局部逆历史

批命令按顺序执行，最终调用内核 validate：端口方向、精确类型键、输入 cardinality、全部连接索引及 DAG 拓扑必须成立。中途或最终失败时，反向执行本批局部逆日志，恢复节点/端口/边/位置/payload、ID 分配器、revision；既有 undo/redo 不变。这里的原子性是同步内存 API 的 Result 边界，不包含进程终止、OOM 或 panic 的恢复。

创建节点、端口、边使用非零显式 u64 ID，拒绝 u64::MAX、低于高水位的复用 ID、同一节点中不递增的端口 ID。允许跳号。失败批恢复高水位；成功 Undo/Redo 恢复原身份但不倒退高水位，新分支不能复用已成功分配的身份。

逆日志持有受影响记录：删除节点移动其 payload，保留其端口和实际 incident edges；位置/数据替换保留旧值；连接变化只保留边。Graph 本身没有 Clone 实现，编辑/undo 不保存整图快照。一次删除若触及全图，其逆记录自然可能达到全图规模；这由 live graph 与历史预算限制，不能称为固定成本。

历史同时限制 undo+redo 总条数和保守 charge。charge 在 admission 时为正反方向预留数据及结构开销，Undo/Redo 不增预算；超大单条拒绝并回滚，可容纳的新条目依次驱逐最老 undo，成功分支编辑才清空 redo。history_bytes() 是保守协议记账，不是 allocator 的实际 RSS 测量。默认 128 条 / 8 MiB charge，硬上限 1024 条 / 64 MiB；live 默认 4096 nodes / 8192 edges，另有每节点端口、payload、文本和命令字节上限。批执行及拓扑校验成本依赖实际模型与触及记录，不承诺无分配。

GraphEditor 持有 Graph<Vec<u8>>；泛型 Graph<N> 的原有公共 API 保留。可通过只读 graph().evaluate 使用真实端口边执行宿主逻辑。编辑层仅支持 DAG；状态机循环等需要另行定义的域协议。分组内核仍存在，当前编辑协议没有分组/子图命令或任意外部可变 kernel 入口。

## 确定性命令字节协议

v1 为 magic PGRF、little-endian u16 版本 1、操作 tag。Edit 包含 u32 条数和有序 edit；身份 u64，位置为有界 i32（绝对值不超过一百万），文本/payload 为 u32 长度前缀；方向、cardinality 使用固定 tag。端口顺序与 edit 顺序属于语义，编码保留，不依赖 hash map 或浮点表示。

编码逐次 append 检查 command_bytes；解码先拒绝总长度、版本、tag、计数和局部长度，再拷贝相应记录；拒绝空批、无效 UTF-8、零/耗尽 ID、截断和尾随数据。解码只验证结构；依赖已有图的引用/类型/拓扑校验在 execute 完成。

从空 editor 用相同 limits 重放成功命令流，包括 Undo/Redo，得到相同结构、位置、payload、分配高水位和历史状态。失败命令不进入成功日志；重放者若需要审计失败，可以记录并比较 Result。此切片不是图快照格式、磁盘 journal、协同编辑或崩溃持久化系统。

## 实际交互、裁剪与成本

- 贡献模板创建真实节点；输入/输出端口展示其真实类型；先选输出再选输入会执行 Connect。每个实际 wire segment 的 painted key 均绑定同一 edge 的 Disconnect。类型不匹配、输入占用或形成循环时模型和历史不变，待连输出保留。
- 节点 activation/Select 支持 Replace/Add/Toggle/Extend（Extend 在图内等同 Add，不隐含列表顺序）；MoveSelection/DeleteSelection 各产生一个可撤销批。工具列有创建、删除、清选、移动、平移、缩放、Undo/Redo 控件。
- SelectRect 使用 document-space 矩形；pointer host 可调用 document_point 转换，再发送 SelectRect/MoveSelection/Pan/Zoom。矩形查询或选择预算不足时整个选择操作失败，不返回悄然截断的选择。
- canvas 的 size 是纯画布尺寸，左侧工具列宽 160 px。zoom 为 0.25..4，size 每轴 64..4096；所有 viewport/矩形输入拒绝 NaN/Inf，范围有界，缩放保持画布中心 document point。
- 节点、端口和三段正交连线使用实际稳定 key 的保留 Element；每段、每个节点框/端口框做数值裁剪。节点两端均不在屏幕内，只要连线路径穿过视口，仍会显示可操作 segment。文字仅在裁剪后存在足够空间时输出缩短标签，不宣称已有通用文字 scissor。
- 成功模型命令后重建稳定排序的平衡 AABB 索引，当前构建成本 O((N+E) log²(N+E))、索引内存 O(N+E)。平移/缩放与 build 不扫描全图，查询 branch visits、candidate output、node/port/edge materialization 均有独立上限；统计明确返回 truncated。候选含 wire AABB 的保守相交，实际 segment 再裁剪，因此 visits/候选预算可能在视觉对象尚少时耗尽。空间索引不是增量更新实现。

默认每 build 上限为 4096 visits、128 node、512 port、256 edge；每 edge 至多三段。硬上限 8192 visits、256 node、2048 port、512 edge。ports 使用 kernel 的有界每节点端口序列。UI 子树创建和查询仍会分配，未做帧时/RSS benchmark 或 GPU 性能承诺。

## 验证证据与剩余边界

实际运行使用自有隔离 workspace，源码链接回真实指定文件，最小 UI harness 包含现有 action/support 和本切片 graph/graph_view；不是全 UI crate、app 或 native 窗口集成测试。单 job、dev/test debug=0、incremental=0、offline。命令：

```sh
CARGO_TARGET_DIR=/private/tmp/prism-foundation-compat.qFtoL5/target-graph-ui CARGO_BUILD_JOBS=1 CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 CARGO_INCREMENTAL=0 cargo test --offline --manifest-path /private/tmp/prism-foundation-compat.qFtoL5/Cargo.toml -p prism_editor_graph -p prism_editor_ui --test graph_panel --test authoring --test graph_kernel --lib
```

结果：graph 原有 9 lib + 1 integration，新增 authoring 9，全部通过；graph_panel 8/8；harness 既有 action 4/4。authoring 含真实数值拓扑求值、ID/索引恢复、typed/cardinality/cycle 回滚、有界历史、规范字节、截断/损坏/资源拒绝和 300 次确定性命令重放。UI 含真实 Ui mount/update、BackendOp::SetLayout 的节点/端口/线段尺寸与位置断言、typed 交互、800 节点可见查询预算和离屏端点连线。限定 package 的隔离 clippy --lib --tests --no-deps -D warnings 通过；不等价于整个 workspace lint 通过。

材质域 native 接入尚未实现：没有 WESL 编译/往返、预览球、native viewport 的命中/拖动/框选事件翻译或 app/domain 注册。当前语义 UiEvent 与 document-space controller API 可实际编辑模型；鼠标事件路由、拖动过程预览、贝塞尔线、对齐、折叠分组、子图、快照存储、域编译与实时预览是后续工作。没有以静态状态文字或 command plan 代替本切片中的 Create/Delete/Connect/Undo 执行。
