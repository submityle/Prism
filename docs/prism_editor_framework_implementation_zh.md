# Prism 编辑器框架实现状态

本文是 `prism_editor_framework_design_zh.md` 的代码状态补充，只记录当前 checkout 中已经存在并通过验证的实现，不把设计规格或代理分支内容计入完成度。

## 当前已交付

### E1 编辑器内核

`pkg/prism_editor_core` 已加入 workspace，并完成以下无 `bevy_*` 依赖的基础能力：

- `DocumentState`、有序 Layer、Base/Override/Runtime 层类型和脏状态/修订号。
- 类型无关的 `Command`、事务、失败回滚、Undo/Redo、影响域集合和命令合并接口。
- 有序去重的 Entity/Asset/SubObject 选择集及主选择。
- 类型键控的服务注册表，以及可注销的命令/面板贡献注册表。
- 单元测试覆盖层约束、事务往返、选择集去重、服务生命周期和贡献注销。

验证命令：

```text
CARGO_TARGET_DIR=/private/tmp/prism-editor-core-target cargo check -p prism_editor_core
CARGO_TARGET_DIR=/private/tmp/prism-editor-core-target cargo test -p prism_editor_core
```

验证结果：5 个单元测试通过，文档测试 0 个；该实现已提交为 `4e6d3a724`。

### E4 面板数据层

`pkg/prism_editor_panels` 已完成无 UI 状态模型：

- Outliner 稳定节点、父子关系、展开/隐藏状态和窗口化可见节点查询。
- 有界 Console 日志、单调序列号、级别/频道过滤和超容量淘汰。
- 通过 `rustc` 独立测试验证 2 个测试用例；workspace 级 Cargo 验证待其他并行 crate 回传后执行。

### 通用时间线基础

`pkg/prism_editor_timeline` 已完成与具体域无关的确定性时间线模型：

- 整数 tick、稳定轨道/关键帧 ID、Step/Linear 插值和维度校验。
- 按稳定 ID 排序的事件轨道，避免同 tick 事件顺序随插入路径变化。
- 轨道采样返回按 `TrackId` 排序的结果，便于帧缓存和可重复测试。
- 通过 `rustc` 独立测试验证 2 个测试用例；未依赖 `bevy_*`。

### E2 反射编辑模型

`pkg/prism_editor_reflect` 已完成并通过 6 个测试：

- 语义字段路径规范化，支持结构体、tuple、列表、数组、map 和 set 的可寻址投影。
- 基于 `prism_reflect` Inspector 的路径化树投影，保留字段元数据。
- 元数据范围/必填/只读校验，以及字段编辑命令的 apply/undo/redo。
- 只依赖 `prism_reflect` 与 `prism_editor_core`，不依赖 `bevy_*`。

### E8 大世界编辑模型

`pkg/prism_editor_world` 已完成并通过 7 个测试：

- 确定性的 Cell 身份、文件路径、数据层可见性/可编辑性和编辑作用域。
- Residency demand 到 load/unload 计划的确定性编排。
- dirty cell 阻止卸载、过期计划拒绝部分执行、失败转换恢复安全状态。
- 可适配 `prism_ecs`/`prism_scene` 分区契约，但不直接执行运行时或文件 I/O。

### E10 通用节点图基础

`pkg/prism_editor_graph` 已完成并通过 7 个测试：

- 稳定 Node/Port/Edge/Group ID、端口类型和连接基数校验。
- 层级分组、连线删除、节点删除和索引一致性维护。
- 稳定拓扑排序、显式 cycle 错误和回调式确定性评估。

### E5 运行时同步协议与 PIE

`pkg/prism_editor_remote` 已完成并通过 10 个测试：

- 传输无关的 endpoint capability、request/response correlation 和明确错误模型。
- Snapshot、incremental changes、revision conflict 和 history retention。
- 有界确定性二进制 codec，拒绝 trailing bytes 并在分配前执行解码限额。
- 客户端副本从 snapshot 到增量 commit 的收敛流程。
- `jsonrpc` 模块提供有界 JSON-RPC 2.0 请求/响应/通知/批量解析、能力协商、BRP 风格 snapshot/change/apply/subscribe 方法和断档重同步通知；27 个测试通过。
- `pie` 模块隔离编辑快照与运行快照，保留有序运行日志，并以显式 Rollback/Commit 返回结果；提交需再次经过编辑版本检查。
- 该 crate 不依赖 `bevy_*`，也不假装提供 socket/进程管理或真实 `bevy_remote` transport；当前交付的是可接入实际 transport 的协议和隔离内核。

### E6 视口交互契约基础

`pkg/prism_editor_viewport` 已完成并通过 3 个测试：

- 多视口稳定 ID、相机/显示设置和代际号，关闭后不复用身份。
- 异步拾取请求的 request correlation、视口校验和过期代际拒绝。
- Translate/Rotate/Scale Gizmo 会话、整数吸附、取消和一次性累计变换结果。
- 该 crate 不执行 GPU 拾取、射线检测或 Gizmo 绘制；这些仍由 `prism_ui_input`/渲染适配层提供。

### E7 内容管线基础

`pkg/prism_editor_content` 已完成并通过 3 个测试：

- 规范化资产路径、确定性内容浏览器查询、标签/类型/收藏/固定过滤和结果上限。
- 正向依赖、反向引用、删除影响查询、缺失引用和循环校验。
- 有界、可取消、确定性导入作业状态机。
- 该 crate 不执行文件 I/O、缩略图、真实 importer/exporter 或 Cook；这些仍由 `prism_asset` 适配层接入。

### E3 外壳与停靠基础

以下两个 renderer-independent 数据层已完成并通过验证：

- `pkg/prism_editor_dock`：面板贡献映射、实例生命周期、工作区布局校验、Primary → LastGood → Default 恢复和有界双流持久化，9 个测试通过。
- `pkg/prism_editor_shell`：命令面板索引/搜索、上下文键位与冲突检测、菜单/工具栏/状态栏贡献聚合，7 个测试通过。
- `pkg/prism_editor_ui` 已完成 Loom `Element` 装配：Shell chrome、命令面板、Outliner、Console、Inspector、Dock workspace、Viewport host 和完整 workspace；动态列表有硬上限，节点使用稳定 key。
- `pkg/prism_editor_app` 的 `EditorApplication::workspace()` 已把这些 Element 与编辑器模型组合起来；该入口仍是无窗口、renderer-independent 的装配层。

### E9 分析器数据层

`pkg/prism_editor_profiler` 已完成并通过 9 个测试：

- 有界 frame/span 树、稳定帧图拓扑与关键路径。
- 内存预算样本/告警及结构化可视日志轨道投影。
- 不连接 Tracy、GPU 或真实运行时采集；这些仍由诊断适配层提供。

### E11 域装配契约

`pkg/prism_editor_domains` 已完成通用域注册与生命周期层：

- Domain/Capability/Contribution/Profile 的稳定 ID、版本范围匹配和能力冲突检测。
- 面板、命令、图节点、时间线和校验器贡献契约。
- 依赖优先启动、依赖者优先停止、确定性诊断和拓扑排序；13 个测试通过。
- 该 crate 只提供域装配契约，不把 VFX、动画、地形、音频或物理业务规则写进通用编辑器。

### E12 协同与无头应用

`pkg/prism_editor_collab` 与 `pkg/prism_editor_app` 已完成第一批可验证能力：

- 协同层提供 HLC/Lamport 时钟、命令 envelope、因果排序、去重、有限历史、字段级 LWW/tombstone、Presence 和 soft-lock；13 个测试通过。
- 应用层提供有界 headless runner、确定性 snapshot/checksum、`prism_app` Update system 集成和 workspace 组合；4 个应用测试通过。
- 当前没有宣称网络服务器、真实多人 transport、操作系统窗口或 GPU 视口已交付。

## 当前未交付

以下项目仍以代码和测试为准，不能仅因设计文档存在就视为完成：

- 真实 BRP/bevy_remote transport、跨进程连接和运行时进程管理。
- GPU 视口后端、ID-buffer picking、实际 Gizmo/debug drawing、相机/网格 render target 和多 viewport GPU 提交。
- 内容浏览器 UI、大世界编辑器 UI，以及各具体域编辑器的业务贡献。
- 真实多人网络 transport、权限/服务器仲裁和操作系统窗口生命周期。

## 不变约束

- 编辑器专有 crate 不得依赖 `bevy_*`；Bevy 机制只在尚未迁移的引擎边界中保留。
- 通用内核不写死项目 gameplay 类型、固定输入通道、网络 authority、AI 或物理策略。
- 每个新增 crate 必须同时具备实现、测试和文档；不以 TODO、`todo!`、`unimplemented!` 或空壳接口宣称完成。
- 设计规格中的 SHIPPED/PLANNED 标记必须以当前源码、Cargo 依赖和测试证据复核。
