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

## 当前未交付

以下项目仍以代码和测试为准，不能仅因设计文档存在就视为完成：

- 反射 Inspector 到 Loom 控件的最终渲染适配。
- 外壳、停靠、核心面板、运行时桥和 PIE。
- GPU 视口、拾取、Gizmo、内容管线、大世界编辑器 UI。
- 域编辑器、分析器、协同和无头编辑器应用。

## 不变约束

- 编辑器专有 crate 不得依赖 `bevy_*`；Bevy 机制只在尚未迁移的引擎边界中保留。
- 通用内核不写死项目 gameplay 类型、固定输入通道、网络 authority、AI 或物理策略。
- 每个新增 crate 必须同时具备实现、测试和文档；不以 TODO、`todo!`、`unimplemented!` 或空壳接口宣称完成。
- 设计规格中的 SHIPPED/PLANNED 标记必须以当前源码、Cargo 依赖和测试证据复核。
