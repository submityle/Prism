# Prefab 作者面板完整性与冲突审查

本文对照设计 §12 记录真实交付边界。组合、继承、嵌套、事务与保存 codec 使用现有 `prism_editor_core::prefab`；UI 不另建组合器或历史模型。面板接入契约和分页详见 `prism_editor_prefab_panel_zh.md`。

## 要求覆盖

| 设计要求 | 面板实际能力 | 仍需集成或限制 |
| --- | --- | --- |
| 嵌套 Prefab 与变体 | 行保留稳定 slot/entity/component/field 路径，真实命令由内核组合嵌套与继承 | 尚无创建变体、插入 slot、重设父级的专用界面；host 可调用同一 execute 入口 |
| 实例显式覆盖与高亮 | Set/Remove/等值 pin 可分页展示、多选，实际共享值可经 visual 查询 | 尚无完整继承属性浏览或反射类型值输入器；列表不能替代 Inspector |
| 还原、应用到基底、下传播 | 已绘制按钮驱动单笔真实命令；显式候选接收者逐项选择；Undo/Redo 走稀疏逆历史 | 传播候选和应用祖先由 host 有界配置，未全库枚举后代；传播冲突明确返回错误 |
| 分层覆盖冲突 | 应用审查显示目标及中间层重叠 pin，可取消或确认精确 pin 解决事务 | 宽/窄覆盖不会自动拆分；需单独决定字段或选择较近的目标 |
| 模型/视图同步 | 私有 Arc 模型 token 与独立 view token 同时围栏；失败保持旧模型与历史 | 场景到 ECS/反射 schema 映射、生产内容验收由对应 host/域工具负责 |
| 原生 app 与保存 | 已向 Erdos 交接 controller、bound subtree、事件和精确保存 ack 契约 | Erdos 报告已实现挂载、Undo/Redo 和脏数据卸载围栏；本文作者未验收原生窗口或生产文件流程 |

## 真实应用冲突流程

选定实例或变体上的显式字段后，host 配置一个应用祖先。`prefab:review-apply` 创建有界审查，显示全部选中源值（包含其他页选中项），以及从源到祖先路径上相关的作者覆盖，包括目标上将被替换的精确 pin。值使用真实 `PrefabOverride` 和共享值 Arc，显示预览有界；不会将继承值假装成作者记录。

`prefab:cancel-apply` 只关闭审查，不改变 Store、历史或模型 token。确认前的选择、分页、目标配置、接收者或模型变更都会使审查失效，并旋转 view token。旧确认事件不能应用于新的选择或 epoch。后台提案仍应捕获模型 snapshot，再由唯一模型线程走 `execute(expected, edits)`。

不存在宽/窄重叠时，`prefab:confirm-apply` 创建一笔 `PrefabCommand`：先 Revert 审查出的精确中间 pin，再对每个选定源覆盖 ApplyToBase。目标精确 pin 会被替换，源 pin 被移除。继承这些路径的其他后代会受影响，确认控件明确提示这一影响；其他后代自己的显式覆盖继续保留。Undo 在同一历史步骤恢复目标、中间层和源记录，Redo 再执行同一事务。

某个较宽对象 pin 可能同时包含未选择的兄弟字段；较窄 pin 也可能表达需要保留的局部决定。遇到任一种，审查列出真实记录但禁用确认，受信任 `ConfirmApplyReview` 也明确拒绝。没有自动删除宽/窄 pin、猜测值合并或反复强制重试。用户可取消后检查该层、单独还原字段，或由 host 配置较近目标。

审查不是成功预执行的保证。确认仍由真实内核检查依赖、字段存在性、最终组合与预算。例如中间层新增字段上存在源 Remove，先撤销中间 pin 后将 Remove 应用到不存在该字段的基底会失败。内核原子回滚整笔，保留中间 pin、源、模型 token、历史与当前审查；host 应呈现 `PrefabPanelError::Command`，不可默默清除或部分发布。

## 有界性

审查按精确地址树查找相关祖先前缀与连续后代区间，不 compose、不克隆完整 Store、不扫描全库层。祖先深度使用内核 `max_depth`，字段深度和单地址使用面板硬上限；每次审查最多 8192 次前缀 lookup/range 迭代步骤。BTreeMap 内部比较成本不计作一次额外迭代，仍受有界地址和内核集合容量约束。

审查 pin 数和选中源值数分别受 `PrefabPanelLimits::selection` 限制，默认各 64、最大各 128；超过时整体返回 Capacity，保留旧审查/视图。待提交的还原数加应用数另受内核 `max_edits` 约束。这里只复制有界地址和 Arc 引用；真实事务仍执行内核有界依赖检查，并可能克隆被编辑层。没有把全 Prefab 克隆作为 UI 增量或 Undo。

## 公共 API 与接入

不增加依赖或根导出。现有 `PrefabPanelAction` 增加 ReviewApplyBase、CancelApplyReview、ConfirmApplyReview；`BoundPrefabPanel::apply_review()` 返回可选真实 pin 行，Some(empty) 表示没有重叠 pin 的有效审查；`apply_proposals()` 返回全部选中源值；`apply_review_blocked()` 表示宽/窄覆盖需要单独决策。

原生 host 只需继续 mount/reconcile `element()`，保留 BoundPrefabPanel，通过其 bindings 和 controller.handle_event 路由。新增按钮自动参与现有绑定机制。异步保存仍用 snapshot()/matches_snapshot()，view-only 审查/取消不使文件变脏；确认成功才更换模型 token。

## 验证证据（2026-10-21）

`cargo test -p prism_editor_ui --test prefab_panel`：17 passed、0 failed、0 ignored。新增六项回归验证精确中间/目标 pin 解决、单步 Undo/Redo、未选 pin 和实例 Arc 保留、取消及陈旧确认、宽/窄冲突拒绝、pin 物化和 edit 展开预算、确认提交中途失败时整批回滚。跨页审查直接检查实际源值 Arc、真实 retained layout 和确认控件 raster 像素，再通过该控件调用内核事务。

全包 strict Clippy 仍被独占范围外 `src/graph_gesture.rs:301/304` 的两处 `semicolon_if_nothing_returned` 阻挡；未修改该文件。仅允许该 lint 的公共 Prefab 测试 Clippy 诊断通过，不能标为全包 strict 通过。完整命令和自有 target 生命周期见面板文档的验证段。
