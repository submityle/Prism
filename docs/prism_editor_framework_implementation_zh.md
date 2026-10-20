# Prism 编辑器框架实现状态

更新日期：2026-10-20。本文补充 `prism_editor_framework_design_zh.md`，只计入当前 checkout 的源码和已执行验证，不把设计目标、契约接口或测试替身视为完整桌面产品。

## 当前结论

可以继续实施，且内核、通用数据层、协议/进程桥、Loom 装配与视口 GPU 基础已落地。当前不是可直接投入生产的完整桌面编辑器；原生窗口、Loom 渲染/输入、真实场景绘制与模型编辑回写仍需组成同一条应用路径。

本轮 16 个编辑器包的全 targets、全 features 测试共 **225 项通过**，包含 Apple M2 / Metal 上的两项强制真实 GPU 测试；默认 features 为 223 项。测试数用于标识验证范围，不代表功能完成百分比。

## 已实现与交付边界

| 包 | 已实现并验证 | 尚未据此宣称交付的能力 | 测试数 |
|---|---|---|---:|
| `prism_editor_core` | Document/Layer、命令事务与失败回滚、Undo/Redo、稳定选择集、类型服务、可注销贡献 | 完整编辑器应用和项目业务规则 | 5 |
| `prism_editor_reflect` | 语义字段路径、Inspector 投影、元数据校验、字段命令、`InspectorSession` 选择/修改/撤销/重做/快照回写闭环 | 原生输入到反射编辑的桌面端到端流程 | 8 |
| `prism_editor_dock` | 面板贡献/实例、工作区校验、Primary/LastGood/Default 恢复、有界持久化 | 多原生窗口拖拽和跨屏交互 | 9 |
| `prism_editor_shell` | 命令索引/搜索、上下文键位/冲突、菜单/工具栏/状态栏聚合 | 宿主键盘焦点和事件路由 | 7 |
| `prism_editor_panels` | 稳定 Outliner 层级、展开/隐藏/窗口查询、有界 Console/过滤 | 生产内容浏览器与域编辑器界面 | 6 |
| `prism_editor_graph` | 稳定图身份、类型/基数校验、分组/索引维护、确定性拓扑与评估回调 | 材质/VFX 等具体节点 UI 和编译器适配 | 10 |
| `prism_editor_timeline` | 整数 tick、稳定轨道/关键帧、插值、维度校验、有序事件与采样 | 序列器 UI 和具体运行时播放适配 | 3 |
| `prism_editor_profiler` | 有界 frame/span、帧图/关键路径、内存预算告警、可视日志投影 | Tracy/GPU/真实运行时采集和分析界面 | 9 |
| `prism_editor_content` | 目录索引、分页/游标、过滤、有界缓存、依赖/反向引用、VCS 元数据、导入导出任务状态 | 文件扫描、缩略图生产、实际 importer/exporter/Cook/VCS provider 和内容 UI | 10 |
| `prism_editor_domains` | 域/能力/贡献/Profile、版本约束、依赖启停、确定性冲突诊断 | 材质/VFX/动画/地形/音频/物理的具体编辑器 | 13 |
| `prism_editor_remote` | 有界 JSON-RPC/二进制 codec、增量副本、PIE、TCP、受控子进程 stdio/TCP 桥及取消/重启/超时 | Prism 引擎专属 adapter 和生产场景运行时接线 | 41 |
| `prism_editor_collab` | 有界因果副本/LWW/tombstone、Presence/租约软锁、TCP 接纳顺序/广播/重连/历史重放/能力拒绝 | 身份认证/TLS、产品级权限/部署和 UI 到命令栈协同接线 | 20 |
| `prism_editor_ui` | Loom Element 装配、有界稳定 key、语义目标/事件到选择/展开/字段编辑/命令/Undo/Redo action 的绑定契约 | 真实设备事件、Loom reconciliation/render 和应用命令执行的端到端连接 | 20 |
| `prism_editor_app` | 有界 headless runner、工作区组合、窗口/surface 生命周期、错误回滚/恢复/退出及 `prism_app` 退出传播 | OS 窗口实现、GPU 呈现实现和完整应用渲染回路 | 16 |
| `prism_editor_viewport` | 稳定场景 ID/增量提交、真实整数 ID target/异步读回、多 viewport target 生命周期/过期拒绝、框选/焦点/Gizmo 契约 | 宿主 geometry/depth 绘制、GPU 矩形采集、平台输入、Gizmo 绘制/命中和编辑回写 | 24（默认 22） |
| `prism_editor_world` | 有界异步 provider 驱动、payload 驻留、分层策略、dirty/租约保护、修订安全保存确认、OFPA/HLOD 计划 | 生产存储 provider/文件格式/崩溃恢复、actor 写盘、HLOD 烘焙/渲染和世界 UI | 24 |

### 运行时进程桥

`RuntimeBridge<A>` 拥有受控子进程和有界 JSON-RPC worker，覆盖启动、握手、stdio/TCP 连接、状态/退出码、重启、取消和超时清理。`EngineRuntimeAdapter` 隔离引擎差异；当前附带 `LoopbackRuntimeAdapter` 和子进程测试 fixture，不是 Prism 场景运行时实现。

请求上限覆盖**执行中加排队中**的未完成工作，不再取决于 worker 调度时机。worker 在交付响应前释放容量；调用者丢弃 `RuntimeRequest` 不会提前释放尚未完成的请求。32 项协议/传输单元测试与 9 项子进程集成测试通过。

### GPU 视口

`SceneIdPass` 将完整 entity/mesh/subobject 身份映射到不复用的 token；快照和 patch 生成稳定增量，拒绝输入不部分修改状态。`SceneIdPassSink` 是 renderer 的 draw-list 边界，不是现成的场景 geometry renderer。

`RenderTargetRegistry` 管理每视口的真实 `Rgba32Uint` target。resize 保留 handle、推进 target generation；close/reopen 更换 handle。异步单像素读回按 viewport、相机/视口代际、场景代际、target 代际和 handle 关联，旧结果不能更新选择。

真实 GPU 测试覆盖整数绘制/清屏 miss/读回/关闭后关联，以及多 target resize/close/reopen。框选目前是有界 request/result 与 ID 解析契约，**未实现 GPU 矩形 copy/reduction**；焦点与 Gizmo 也仍需宿主事件和渲染适配。

### 大世界执行数据层

`CellIoProvider` 返回拥有所有权的异步 Future，`WorldIoDriver` 限制请求数和预留字节数，提供取消/完成事件。`EditorWorldData` 管理 payload 驻留、失败恢复、编辑作用域、分层可见性/可编辑性、租约 token 与 dirty 保护。

保存确认检查请求快照、当前修订、字节、有效锁和 actor dirty 状态，旧完成事件不能清掉新编辑。测试包含异步 Pending/wake/load/save/evict/reload 往返，但使用内存 provider；不把它视为生产文件持久化。OFPA 是稳定 actor 文件计划，HLOD 是 show/hide 代理计划，均不冒充实际序列化或烘焙。

### 应用生命周期与 UI

`EditorApplication` 区分 Headless/Windowed。Windowed 必须显式注入 `WindowBackend` 和 `GpuSurfaceLifecycle`，按窗口→GPU→surface 启动并逆序清理；部分失败同样回滚，清理错误不阻止其他资源释放。窗口数量和每帧事件均有限额，多窗口独立挂起，零尺寸暂停、恢复尺寸后继续，关闭不再多提交一帧。

16 项应用测试使用 recording backend，验证顺序、恢复、预算和退出传播，**不证明原生窗口或实际呈现已运行**。`prism_editor_ui` 的 bound builders 可将语义事件解析为 `EditorAction`；当前应用 workspace 仍是视图组合入口，不是完整输入→action→模型变更→Loom 绘制回路。

## 验证记录与复现

在仓库根目录为本次验证选择独立 target，避免多个任务共用或清理彼此的构建目录：

```sh
export CARGO_TARGET_DIR="$(mktemp -d /private/tmp/prism-editor-verify-XXXXXX)"
export CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0
cargo test -p 'prism_editor_*' --all-targets
PRISM_REQUIRE_GPU=1 cargo test -p 'prism_editor_*' --all-targets --all-features -- --nocapture
cargo clippy -p 'prism_editor_*' --all-targets --all-features --no-deps -- -D warnings
cargo tree -p 'prism_editor_*' --all-features --prefix none --edges normal,build,dev
```

- 默认 features：223 项通过；全 features 且强制 GPU：225 项通过，0 失败，0 忽略。
- GPU 硬件：Apple M2，IntegratedGpu，Metal。两项测试均输出 `EXERCISED`；沙盒内无 adapter 时，`PRISM_REQUIRE_GPU=1` 明确失败，随后沙盒外执行通过。无适配器时的可选跳过不计 GPU 验证。
- 严格 Clippy 范围为编辑器包及其全部 targets/features（`--no-deps`）；不能扩写为全仓库/全部依赖零告警。此前完整依赖 lint 曾遇到 `prism_ecs` 既有告警，未越界修改。
- 编辑器 Cargo 依赖树的 normal/build/dev 路径未发现 `bevy_*`。该结论基于实际依赖树，而不只搜索源码文字。
- 本轮代码改动执行包级 fmt/diff 检查；编辑器范围无 `TODO`/`FIXME`/`todo!`/`unimplemented!` 未完成标记。UI 文本输入的 `.placeholder(...)` 是控件属性，不是实现占位。

## 下一条实施主线与验收

优先完成一条真实桌面垂直切片：

`原生窗口 → Loom reconciliation/render → 加载场景 → 多视口绘制 → 输入拾取 → Inspector 字段命令 → Undo/Redo → 保存/重载`

验收必须同时包含：真实设备事件、真实场景 geometry/depth、闭环命令回写、窗口/surface 恢复、过期拾取拒绝，以及保存/重载一致性。不能用录制后端、fixture 场景协议或 Element 树测试替代这些证据。

之后依次接入生产资产/世界存储、引擎专属运行时 adapter、真实诊断采集、各具体域编辑器和协同产品层。AAA 规模、性能和跨后端一致性仍需指定内容/硬件/分辨率/质量并测量 P50/P95/P99，现有正确性测试不提供这类结论。

## 不变约束

- 编辑器专有包不得依赖 `bevy_*`；历史名称不是实现依赖。
- 通用内核不写死项目 gameplay 类型、固定输入通道、网络 authority、AI 或物理策略。
- 每个可交付切片包含实现、测试和文档；接口契约不冒充其所有适配实现。
- 设计规格中的状态标记以当前源码、Cargo 依赖和测试证据为准；完整系统目标尚未完成。
