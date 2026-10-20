# Prism 编辑器框架实现状态

更新日期：2026-10-20。本文补充 `prism_editor_framework_design_zh.md`，只计入当前 checkout 的源码和已执行验证，不把设计目标、契约接口或测试替身视为完整桌面产品。

## 当前结论

可以继续实施，且内核、通用数据层、协议/进程桥、Loom 装配与视口 GPU 基础已落地。当前不是可直接投入生产的完整桌面编辑器；原生窗口、Loom 渲染/输入、真实场景绘制与模型编辑回写仍需组成同一条应用路径。2026-10-20 已接入原生文档会话：保存/加载通过事件循环异步轮询，保存确认受模型快照围栏保护，加载成功后原子替换 UI，并通过 `SceneDocumentAdapter` 候选/提交边界为宿主提供文档到场景的引擎中立绑定；尚未形成完整原生端到端验收证据。

编辑器包此前的全 targets、全 features 普通回归与强制真实 GPU 回归以独立 target 执行；历史强制 GPU 证据使用 Apple M2 / Metal。当前视口增量已将矩形逐像素读回改为 GPU 去重归约，2026-10-20 重新执行的视口全 features 测试为 37 项通过，包含 8 项真实 GPU 测试。测试总数随切片增量变化，以下以命令输出和对应提交状态为准，不把固定总数当作产品完成度证据。此前 `prism_editor_app` 严格 Clippy(`--no-deps`)也通过，依赖树无 `bevy_*`。这证明当前切片可验证，不等于完整桌面产品或引擎集成已完成。

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
| `prism_editor_content` | 目录/子树索引、末项续查分页、有界缓存与诊断、增量反向依赖、真实异步文件扫描、原子源目录发布、可贡献校验规则与精确快照门禁、原子 VCS 元数据和导入导出任务状态 | 缩略图生产、实际 importer/exporter/Cook/VCS provider 和原生内容流程 | 56（目录/扫描/发布/校验） |
| `prism_editor_domains` | 域/能力/贡献/Profile、版本约束、依赖启停、确定性冲突诊断 | 材质/VFX/动画/地形/音频/物理的具体编辑器 | 13 |
| `prism_editor_remote` | 有界 JSON-RPC/二进制 codec、增量副本、PIE、TCP、受控子进程 stdio/TCP 桥及取消/重启/超时 | Prism 引擎专属 adapter 和生产场景运行时接线 | 41 |
| `prism_editor_collab` | 有界因果副本/LWW/tombstone、Presence/租约软锁、TCP 接纳顺序/广播/重连/历史重放/能力拒绝 | 身份认证/TLS、产品级权限/部署和 UI 到命令栈协同接线 | 20 |
| `prism_editor_ui` | Loom Element 装配、有界稳定 key、语义目标/事件到选择/展开/字段编辑/命令/Undo/Redo action 的绑定契约 | 真实设备事件、Loom reconciliation/render 和应用命令执行的端到端连接 | 20 |
| `prism_editor_app` | 有界 headless runner、工作区组合、窗口/surface 生命周期、错误回滚/恢复/退出及 `prism_app` 退出传播；原生文档会话接线、Ctrl/Cmd+S 保存、Ctrl/Cmd+O 加载、receipt 围栏、引擎中立 `SceneDocumentAdapter` 候选/提交边界、点选/拖拽框选、重载 renderer 失效钩子、单对象 transform 增量更新和 native CLI smoke entry | 完整原生输入→Loom→真实场景→Inspector→保存重载验收，以及具体引擎/项目场景 adapter | 见验证命令 |
| `prism_editor_viewport` | 稳定场景 ID/增量提交、indexed geometry/depth ID pass、真实整数 ID target/异步读回、GPU 矩形去重归约、多 viewport target 生命周期/过期拒绝、Gizmo 屏幕命中与会话 | 引擎材质/场景提取、平台输入、Gizmo 绘制与宿主变换命令绑定 | 37（本次全 features） |
| `prism_editor_world` | 有界异步 provider、版本化 cell 文件持久化、校验/并发锁/修订冲突/阶段文件恢复、payload 驻留、分层策略、dirty/租约保护、OFPA/HLOD 计划 | actor codec/写盘执行、跨 cell/OFPA 事务、HLOD 烘焙/渲染和世界 UI | 41（本次全 targets） |

### 运行时进程桥

`RuntimeBridge<A>` 拥有受控子进程和有界 JSON-RPC worker，覆盖启动、握手、stdio/TCP 连接、状态/退出码、重启、取消和超时清理。`EngineRuntimeAdapter` 隔离引擎差异；当前附带 `LoopbackRuntimeAdapter` 和子进程测试 fixture，不是 Prism 场景运行时实现。

请求上限覆盖**执行中加排队中**的未完成工作，不再取决于 worker 调度时机。worker 在交付响应前释放容量；调用者丢弃 `RuntimeRequest` 不会提前释放尚未完成的请求。32 项协议/传输单元测试与 9 项子进程集成测试通过。

### GPU 视口

`SceneIdPass` 将完整 entity/mesh/subobject 身份映射到不复用的 token；快照和 patch 生成稳定增量，拒绝输入不部分修改状态。`SceneIdPassSink` 是 renderer 的 draw-list 边界，不是现成的场景 geometry renderer。`DesktopScene` 现在提供 `SceneDocumentAdapter`，由宿主把反射文档映射为完整 indexed geometry/selection identity 候选，并在 UI 安装成功后原子提交；它不推断项目业务类型。宿主对已发布对象的变换可调用 `update_object_model`，在线设备路径只重建对应实例 bind group，暂停/未初始化路径延迟到下次上传，并取消旧拾取读回。

`RenderTargetRegistry` 管理每视口的真实 `Rgba32Uint` target。resize 保留 handle、推进 target generation；close/reopen 更换 handle。异步单像素读回按 viewport、相机/视口代际、场景代际、target 代际和 handle 关联，旧结果不能更新选择。

真实 GPU 测试覆盖 indexed geometry、前向/反向深度遮挡、相机/模型/网格变化、清屏 miss、GPU debug-color 预览及多 target resize/close/reopen。`DesktopScene` 的拖拽框选调用 `encode_box_readback`：compute 在实际 `Rgba32Uint` target 的请求矩形内读取像素，用有界哈希表在 GPU 去重，异步读回唯一 ID 表与计数，再通过一次 `select_items` 原子提交。原子槽保存源像素索引，竞争线程直接比较只读源纹理，不等待别的线程发布多字 ID；完整四通道身份不被截断。读回结果排序，消除 GPU 插入调度对公开结果顺序的影响。

框选限制扫描像素数、唯一 ID 数、读回字节数和并发请求。二维 dispatch 支持超出单维 workgroup 上限的矩形，并检查填充 lane 的 32 位索引溢出；4096 ID 容量对应 131,080 字节读回，与框选面积无关。新 GPU buffer 的零初始化避免 CPU 另建、映射和复制整张表。超出容量时整个请求报错，不截断选区。2026-10-20 的真实 GPU 回归包含 16 个刻意同桶的四字 ID、跨工作组重复像素、重复运行顺序一致及容量 16/15/0 的成功/拒绝。焦点、框选可视化 overlay 和原生 Gizmo 绘制/变换回写仍需宿主适配。

本次增量验证命令为 `PRISM_REQUIRE_GPU=1 cargo test --offline -p prism_editor_viewport --all-features -- --nocapture`（37 项通过、无跳过，Apple M2/Metal），以及 `cargo clippy --offline -p prism_editor_viewport --all-targets --all-features --no-deps -- -D warnings`、包级 fmt 与定向 diff 检查。沙盒内无法枚举 Metal adapter 的失败未计入 GPU 覆盖；上述成功测试在授权的沙盒外执行。

### 内容目录与增量维护

`ContentCatalog` 的依赖编辑仅更新新增/移除的反向边；只有路径变动才更新树与路径索引。相同记录、重复依赖边与相同 VCS 观察不会推进 generation 或使游标/缓存失效。资产和树分页从前页末项继续查找，子树第一页从前缀开始，避免深分页重复扫描全部前页；树查询借用索引，不复制整个目录。过滤仍需检查未匹配候选及目录后代，不能据此宣称所有查询成本恒定。

路径保留 UTF-8 空格、换行和 Unicode 身份；绝对路径、反斜杠、NUL、空分量及点/父目录分量显式拒绝。文件/目录前缀冲突在修改索引前检查。`replace_vcs_statuses` 校验完整候选后一次性替换，清除过期状态，非法/重复路径不会部分发布。诊断收集过程按 `max_diagnostics` 保留确定性排序前缀，不先分配全量错误列表。

依赖闭包与环检测借用邻接集合/迭代器，不在节点数限制生效前复制高扇出依赖列表。深度以根为零统一计数，到达边界叶子或已见回边成功；发现更深的新节点才拒绝，避免完整边界路径误报超限。

`FilesystemDiscovery` 以标准库工作线程读取真实目录和元数据；默认剪枝仓库/构建产物，不跟随符号链接。并发、输出/枚举数量、路径字节、深度、文件大小及协作式超时均有界；只发布完整排序快照，错误/取消不返回半批数据。标准库无法打断阻塞 OS 调用，drop 会 join，不承诺硬实时截止或恶意目录替换隔离，详见 `prism_editor_content_filesystem_zh.md`。

`replace_from_filesystem` 用宿主 manifest/sidecar 解析稳定资产身份，不以路径哈希冒充重命名稳定 ID。扫描开始前捕获 generation，完成时拒绝覆盖更新的编辑；解析失败、非法/重复记录、前缀冲突或预算失败均保持旧目录。`replace_assets` 同样支持其他源适配器的完整候选，接受路径互换/删除，保留缺失依赖诊断和按路径的 VCS 观察。候选记录、路径长度/深度、边/标签数量和逻辑字节分别受限；只在有效变化时推进一次 generation，无变化不重建派生索引，路径不变保留树，反向引用按边差分更新。

2026-10-20 执行 `cargo test --offline -p prism_editor_content --all-targets`：10 项单元、8 项目录回归、28 项真实文件扫描及 4 项原子发布测试。`cargo clippy --offline -p prism_editor_content --all-targets --all-features --no-deps -- -D warnings` 使用同一独立 target 验证。回归覆盖过滤分页/缓存淘汰、400 次混合修改的反向引用比对、所有扫描/发布预算、真实重命名保留宿主身份、代际过期和失败不部分发布。这里只验证源发现/目录数据，不把任务状态模型计作实际 importer/exporter/Cook 执行。

### 内容校验与操作授权

`validation::ValidationRegistry` 按稳定规则 ID 执行宿主贡献，检查前先限制资产、依赖边及规则调用数；诊断输出分别限制发射次数（含重复）、保留行数、消息和标识字节。回调只能为当前资产输出诊断；忽略 sink 错误也不能获得通过结果。宿主回调是可信代码，回调内部耗时和分配不属于这套预算的隔离保证。

`DiagnosticReport` 与 `ValidationReceipt` 的错误/警告计数独立于显示截断，隐藏错误仍然阻止操作。依赖深度未完成、规则执行失败均不能授权。`authorize` 在生成借用的 `ValidatedCatalog` 前核对精确目录快照、规则集合及 Save/Commit/Cook 操作；有效目录修改和规则变动撤销旧 receipt，相同 generation 的独立目录或分叉 clone 不能互用授权。无变化观察保留 receipt；严格策略另可拒绝警告。注册期间规则配置须保持不变，配置改变后重新注册；外部状态与异步输出发布还须由适配器检查其修订。此模块提供门禁能力，不能据此声称所有保存/提交/构建入口已接入。

2026-10-20 新增 6 项实际校验回归，连同此前 50 项目录/扫描/发布测试全部通过。覆盖显示截断隐藏错误、零显示容量、操作策略、未完成检查、回调错误与被忽略的 sink 错误、重复诊断预算、同代际分叉快照以及规则变更。

### 大世界执行数据层

`CellIoProvider` 返回拥有所有权的异步 Future，`WorldIoDriver` 限制请求数和预留字节数，提供取消/完成事件。`EditorWorldData` 管理 payload 驻留、失败恢复、编辑作用域、分层可见性/可编辑性、租约 token 与 dirty 保护。

保存确认检查请求快照、当前修订、字节、有效锁和 actor dirty 状态，旧完成事件不能清掉新编辑。`FileCellIoProvider` 已在提交 `d70e41247` 落地真实文件持久化：64 字节版本化 envelope、坐标/GUID/长度/校验和验证、有界异步 worker、每 cell OS advisory lock、同目录 stage→fsync→rename→目录 fsync。加载恢复 durable revision，保存按已观察的 base revision 拒绝冲突；取消围栏阻止后续发布，rename 后目录同步失败显式返回 `CommitUncertain`，不能据此清除 dirty。临时目录测试包含实际文件往返、错误 envelope、符号链接拒绝、并发 provider 和子进程锁/进程退出恢复；故障注入不等于物理断电验证。

文件持久化按 cell 原子提交，尚不提供跨 cell/OFPA 事务、备份修复或恶意目录替换隔离；具体格式与恢复步骤见 `pkg/prism_editor_world/README.md`。OFPA 目前仍是稳定 actor 文件计划，HLOD 是 show/hide 代理计划，不能计作 actor codec/写盘执行或实际烘焙。

2026-10-20 在当前 checkout 执行 `cargo test --offline -p prism_editor_world --all-targets`，14 项单元、7 项执行层、12 项真实文件系统、8 项恢复测试全部通过。另重新执行编辑器全 features 的 normal/build/dev Cargo 依赖树检查，未发现 `bevy_*`。

### 应用生命周期与 UI

`EditorApplication` 区分 Headless/Windowed。Windowed 必须显式注入 `WindowBackend` 和 `GpuSurfaceLifecycle`，按窗口→GPU→surface 启动并逆序清理；部分失败同样回滚，清理错误不阻止其他资源释放。窗口数量和每帧事件均有限额，多窗口独立挂起，零尺寸暂停、恢复尺寸后继续，关闭不再多提交一帧。

16 项应用测试使用 recording backend，验证顺序、恢复、预算和退出传播；另有 native CLI smoke run 在 Apple M2/Metal 上实际提交 1 帧并返回 `error=None`。这仍不等于完整输入→action→模型变更→Loom 绘制→场景编辑闭环。`prism_editor_ui` 的 bound builders 可将语义事件解析为 `EditorAction`。

## 验证记录与复现

### 原生文档会话接线

`DesktopDocumentSession` 持有单文档 `DocumentFileAdapter`、schema allowlist 和 controller limits。`run_desktop_with_document` 将它接入 winit user event/waker：Ctrl/Cmd+S 提交异步保存，Ctrl/Cmd+O 提交加载；保存完成后才调用 `acknowledge_ui`，加载完成后先 `prepare`，再调用 renderer 的 `prepare_document` 构造候选场景，候选 UI 安装失败时丢弃候选，成功安装后提交 renderer 候选。native 鼠标在 viewport 内按下/释放时区分点选和拖拽框选，框选结果走 GPU readback 后一次性提交多选。配置了文档会话时，窗口关闭会在 dirty、活动 draft、pending I/O 或 reconciliation-required 状态下阻止静默退出；未配置文档会话的旧 `run_desktop` API 保持原行为并始终返回 UI/model 所有权。

在仓库根目录为本次验证选择独立 target，避免多个任务共用或清理彼此的构建目录：

```sh
export CARGO_TARGET_DIR="$(mktemp -d /private/tmp/prism-editor-verify-XXXXXX)"
export CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0
cargo test -p 'prism_editor_*' --all-targets
PRISM_REQUIRE_GPU=1 cargo test -p 'prism_editor_*' --all-targets --all-features -- --nocapture
cargo clippy -p 'prism_editor_*' --all-targets --all-features --no-deps -- -D warnings
cargo tree -p 'prism_editor_*' --all-features --prefix none --edges normal,build,dev
```

- 本轮 `cargo test -p 'prism_editor_*' --all-targets --all-features` 与同命令前置 `PRISM_REQUIRE_GPU=1` 均通过，0 失败；不把无适配器时的可选跳过计入 GPU 证据。
- GPU 硬件：Apple M2，IntegratedGpu，Metal。强制回归输出真实 adapter 信息及 `EXERCISED`，并覆盖 viewport 与 editor app 的真实 wgpu 路径。
- 严格 Clippy 范围为编辑器包及其全部 targets/features（`--no-deps`）；不能扩写为全仓库/全部依赖零告警。此前完整依赖 lint 曾遇到 `prism_ecs` 既有告警，未越界修改。
- 编辑器 Cargo 依赖树的 normal/build/dev 路径未发现 `bevy_*`；根 workspace 的 `serde`/`serde_json` 声明也已补齐，保证使用 `workspace = true` 的构建 CLI 清单可解析。该结论基于实际依赖树，而不只搜索源码文字。
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
