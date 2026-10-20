# Prefab Loom 面板

本模块将 prism_editor_core::prefab 的真实作者态模型接到 Loom retained Element。面板持有真实 PrefabStore 和 CommandStack；所有作者态写入由 PrefabCommand 调用内核事务，逆历史使用内核 PrefabReceipt 的稀疏记录。没有独立 UI 数据库或模拟命令执行器。

## 接入

UI 根模块声明 prefab、prefab_view，并导出：

- PrefabPanelController、PrefabPanelAction、PrefabPanelError、PrefabPanelLimits、PrefabPanelSnapshot。
- BoundPrefabPanel、PrefabOverrideVisual、PrefabPanelViewStats。

这些 roots 已由主线程接入 UI 根模块，本实现只拥有 src/prefab.rs、src/prefab_view.rs、tests/prefab_panel.rs 和本文档。无需修改 Cargo 依赖。使用现有 prism_editor_core / prism_ui，测试使用现有 prism_ui_render_backend。没有 Bevy 或业务组件类型依赖。

host 将真实 Store 移交 PrefabPanelController::new(store, owner, limits)，用 build() 取得可 mount/update 的 Element。store() 仅提供只读访问；关闭可用 into_store() 取回所有权。面板内核拥有者不能同时与另一个可写 Store 副本保持双向权威。

host 用 configure_targets(expected, base, recipients) 配置应用基底目标和显式传播候选。列表有界、不扫描全仓库后代；面板显示目标稳定 ID，接收者逐项勾选，默认均选中。候选合法存在、无重复在配置时验证，严格继承关系、字段冲突和组合预算由真正执行的内核事务检查。切换 Owner 会清除选择、分页、目标与候选，host 应重新配置。资产面板支持传播；实例面板支持还原和应用基底，传播按钮禁用。

将原生点击转换为 UiEvent，调用 handle_event(&bound, key, &event)。只有真实绘制且启用的 key 能 dispatch；Activate 和带 Replace/Add/Toggle/Extend 的 Select 都支持字段选择。Extend 的明确语义为添加单个地址，不扫描两端之间的未显示字段；跨页多选使用 Add/Toggle。Commit 事件被显式拒绝，本面板不提供任意值输入器，字段赋值可经 host 的真实作者工具调用 execute。

## 作者态、精确路径与分页

列表仅显示当前作者层的显式字段覆盖，包括 Set、Remove 和显式等值 pin；继承字段并不伪造成本地覆盖。PrefabOverrideVisual 持有完整的嵌套 slot/entity/component/field 语义路径及实际 PrefabOverride。UTF-8、换行、点、斜杠、空字段名按原样区分。显示标签会转义并截断，动作绑定和 visual.property 保留精确地址；key 对语义段编码长度及十六进制字节，避免分隔符碰撞。

Set 的值只克隆 Arc 引用。Null 与 Remove 不同，浮点显示精确位模式，字符串预览有界，数组/对象/字节仅显示类型和长度，不递归展开值。结构性 slot/entity/reparent 编辑不在字段覆盖列表内；可通过同一个 execute 事务入口操作，内核仍负责验证。

构建直接在 layer.overrides 的 BTreeMap 上做 keyset range，不调用 compose、不遍历完整 Prefab 树，不复制整层或 Store。每页默认 32、最多 128 行；range 迭代最多 rows+1 个条目，额外条目只作有后续的工作证据。前后端 key 比较及 range seek 是树查找，统计 examined 不计树内部比较。Previous 从当前首键向前逆序取一页，再反转本页；Next 从当前尾键向后取一页。没有随翻页数增长的历史栈或 offset skip。模型命令、undo/redo 后回首页，避免保留失效游标。

选择、候选、复制地址大小、历史均有独立硬上限。默认选择 64、候选 64、单地址 4096 字节、历史 128 项，均可降低。地址上限包含路径容器开销，超限 build 显式失败，不静默漏行。关闭页面会释放页面 Arc 引用；旧 bound 仍被 host 持有时会保留其有界值引用。重复持有无限份页面属于 host 生命周期管理。

## 命令、原子性和 fence

Revert 为所有选定覆盖生成一批 Revert；ApplyBase 为所有选定覆盖生成一批 ApplyToBase；Propagate 对选定资产覆盖生成一批 Propagate，接收者为显式勾选集。每次按钮只创建一个 PrefabCommand，而非循环提交多个命令。任一字段、后代、祖先或内核预算拒绝时，整批不发布、不增历史、不清选择、旧 view 仍有效。传播还预先检查 fields × (recipients + 1) 的内核 edit 展开预算。内核会保护中间变体覆盖和较宽属性覆盖，不强行覆盖兄弟字段。未选实例的显式 pin 和未触及记录的 Arc 身份保持原样。

Undo/Redo 直接调用 CommandStack 和 PrefabCommand 逆历史，不保存每次全 Store 克隆。历史条目总量以 undo+redo 受限，正常新分支会丢弃 redo 后再占用一项；undo 深度已经到上限时明确拒绝新命令，不自动丢弃旧历史。内核接受的 no-op 命令也占一个历史项。每项保留被修改记录的前后 Arc，记录大小仍受 PrefabLimits 约束，并不宣称单项历史是常数字节。

snapshot() 返回带私有 Arc 身份和诊断 revision 的 PrefabPanelSnapshot。execute(expected, edits)、configure_targets 和 replace_store 必须匹配精确模型 capability。控制器独占 Store 且无可写逃逸，任何成功作者命令、undo、redo、替换 Store 都更新模型身份。即便 undo 恢复相同值、从相同编码载入、独立 Store revision 恰好一致，也拒绝旧 capability。读取 encode/compose 不改变身份。replace_store 在验证新 owner 后才交换所有权并清空历史、选择和配置。

每个 BoundPrefabPanel 另外保存控制器 view 身份。分页、选择、接收者、检查对象等视图变化会更新 view 身份，保留模型 token；旧行 key 即便仍存在也不能作用到新视图。build 本身不更新身份，同一未变状态的重复构建可无操作 reconcile。handle_event 在解码动作前同时验证 model/view fence。dispatch 仅用于纯解码；不要绕过 handle_event 将旧 view 的动作传给受信任 host API apply。

对于异步 host 工作：先捕获 snapshot，后台只计算提案，回到唯一 UI/模型线程调用 execute(&snapshot, edits)。失败应显示返回的 PrefabPanelError，刷新 view 后让用户重新操作；不自动重放到新 epoch。原生 app 的 dock 注册、设备事件路由、保存对话框以及诊断投递由 host 接入，本模块没有编辑 app 文件。

有界增量的范围是 UI 查询/物化和稀疏记录逆历史。实际内核事务仍会在 PrefabLimits 的 max_steps/entities/value 等预算内验证和组合受影响依赖，并可能克隆被编辑的一层；本面板没有将此操作标为整引擎常数时间更新。

## 验证

公共 tests/prefab_panel.rs 直接导入 prism_editor_ui，未以 path 重编译内核或 UI。回归通过真正 Store 组合值和 Arc 身份检查命令效果，包括多字段应用/还原、选择接收者传播、多实例独立 pin、后一个字段冲突时原子回滚、传播展开预算、历史预算、分页双向边界、模型及控制器 ABA、嵌套路径、精确值共享。

retained 测试真实 mount/update Element，以 RetainedScene 计算 layout、生成 draw list、CPU rasterize 检查选择/未选行像素；再从绘制控件执行真实 Revert 并验证组合值与 retained 更新。相同状态重建检查 RecordingBackend 零差异。

2026-10-21 公共 API 回归：`cargo test -p prism_editor_ui --test prefab_panel`，11 passed、0 failed、0 ignored。覆盖真实内核事务、组合结果、逆历史、精确 fence、有界分页和 retained layout/raster。

完整严格检查命令：`cargo clippy -p prism_editor_ui --all-targets --all-features --no-deps -- -D warnings`。当前未通过：共享 checkout 中其他任务拥有的 `src/graph_gesture.rs:301/304` 触发 `semicolon_if_nothing_returned`；该次检查没有报告 Prefab 模块错误。不能将此结果视为全包 strict Clippy 已通过，修复共享文件后须由联合验证重新执行。

验证统一设置 `CARGO_TARGET_DIR=/private/tmp/prism-prefab-panel.ym06D6 CARGO_BUILD_JOBS=1 CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 CARGO_INCREMENTAL=0`。该独立 target 只属于本任务，验证结束后用 `cargo clean --target-dir /private/tmp/prism-prefab-panel.ym06D6` 清理；不清理主线程或其他任务 target。
