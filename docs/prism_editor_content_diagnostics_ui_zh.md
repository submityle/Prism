# E7 内容诊断与依赖浏览 UI

本切片仅修改 `pkg/prism_editor_ui/**` 与本文，基于内容浏览器提交
`18b9af39a` 和内容 validation API 提交 `ee749f273`。实现 §11 的真实
有界依赖列表、校验报告与定位动作；节点画布、原生桌面接入和保存/提交/cook
操作仍由各自所有者负责。未增加 Bevy 依赖。

## 公开接入边界

UI 导出 `ContentDiagnosticsController`、`ContentDiagnosticsAction`、
`ContentDiagnosticsError`、`DependencyDirection`、`BoundContentDiagnostics`
及 `MAX_DEPENDENCY_NAVIGATION_HISTORY`。沿用 `UiEvent` / `UiTrigger`，
不改变 `EditorAction`。Erdos 的 app 接入可使用：

```rust,ignore
// 宿主持有 catalog、registry、controller、displayed_view 和 Loom runtime。
// 仅在应用资产选择变化时同步 Focus。
controller.apply(&catalog, &registry,
    ContentDiagnosticsAction::Focus(Some(asset_id)))?;
let view = controller.build(&catalog, &registry);
// 挂入 dock body；更新时用 Loom reconciliation。
ui.mount(view.element());

// 输入适配器按 view.bindings() 的真实 Element key 路由 Activate。
let action = controller.handle_event(&catalog, &registry, &view, key, &event)?;
// Focus / LocateIssue / Back 后，读取结果并同步应用资产选择。
let selected = controller.selected_asset();
let next_view = controller.build(&catalog, &registry);
ui.update(next_view.element());
// 用 next_view 替换保存的 displayed_view，再接收后续事件。
```

`build` 返回绑定树；两个输入均只读。`dispatch` 仅解析 owned action，
`handle_event` 额外检查原控制器身份、revision、catalog generation 和 receipt
当前状态。错误触发类型、无绑定/禁用目标以及旧树被明确拒绝。宿主必须将一个
控制器持续绑定到其 catalog；替换 catalog 实例后应重新建立控制器。

`Focus`、`Direction`、`Back`、依赖/报告的上一页与下一页、`PageSize`、
`RunValidation(operation)`、`LocateIssue(index)` 均为显式命令，字段没有公开可变
访问入口。定位只改变 UI 当前资产，不修改 catalog，也不直接操作其他面板或文件。
app 需把最终 `selected_asset()` 同步到主选择/内容浏览器，并按需打开资产编辑器。
这里没有原生点击、像素渲染、文件打开或保存流程已经完成的主张。

## 依赖导航

“Dependencies (outgoing)”调用 `ContentCatalog::dependency_closure`；
“Dependents (incoming)”调用 `dependent_closure`。两者均为传递闭包，按稳定
AssetId 排序显示，排除根资产，遵守 catalog 的节点数与深度上限。每个方向的
列表独立于校验 findings 列表；分页大小共用一个配置，默认 64，每个列表最多
`MAX_VISIBLE_ROWS`（512）行。0 被拒绝，超大配置被截到上限。

不是图形占位：每条实际依赖均为真实 keyed Loom Element；存在的资产可导航，
缺失目标显示其真实 ID 并禁用打开。遍历超限明确显示错误，不伪造部分图或空成功
结果。选择、方向或 catalog generation 改变时依赖页回到开头；上一页通过确定
offset 返回，末页下一页无绑定。asset ID 保持行身份，路径只是显示信息。

Back 保留最多 64 个历史选择，包含清空状态，不保存 catalog 副本或无限的路径
栈。资产被删除后仍显示缺失选择和真实遍历错误；可清空或重新选择。

## 校验报告

Save/Commit/Cook 三个按钮只执行对应的 `ValidationRegistry::validate`。UI
保留一份真实 `ValidationReceipt`，重绘、依赖导航和 findings 翻页不会再次运行
贡献规则。结果按 receipt 自身的确定性顺序分页；行包含 severity、rule/code、
真实资产 path 与 message。定位 action 使用当前 receipt 的 index，且执行时
再次核对 freshness、index 和 AssetId 是否仍存在。无资产定位的全局诊断无绑定。

报告显示 operation、所有 finding 的 error/warning 计数、complete、truncated
以及默认 policy 下的 status。截断计数与保留行数分别呈现；没有 retained findings
不能代替完整性判断。规则 callback 失败产生的真实 finding 和不完整报告仍可读。
catalog 或规则集变更后，旧结果保持可读并明确标记 stale；定位禁用，必须显式
重跑。运行前预算/规则错误通过 `ContentDiagnosticsError::Validation` 返回宿主，
不覆盖已有 receipt 或选择；宿主负责呈现这一操作错误。

`report_status` 使用 receipt 的 snapshot/registry 身份围栏和默认 policy，
不调用贡献规则。它只为当前报告展示服务。`receipt()` 允许宿主取得真实 receipt；
实际保存、提交或 cook 必须在操作准备时重新调用其 `authorize` 并应用项目 policy，
不能把 UI 上的旧状态当作授权。宿主拥有规则注册、异步调度与外部数据围栏；本 UI
显式调用同步 registry API，不启动线程。

## 验证与资源所有权

`tests/content_diagnostics.rs` 在真实 `Ui<RecordingBackend>` 上挂载、计算布局、
派发真实 Element key 对应的语义事件，再对更新树执行 reconciliation。测试覆盖
双向分页、返回/缺失引用、实际规则执行与报告定位、catalog/registry 失效、完整性/
截断提示、唯一绑定 key 和可命中布局、512 行与 64 条历史上限、真实循环诊断、
干净报告与预算失败时保留旧 receipt。2026-10-20 验证：原有 20 项单元测试、
8 项内容浏览器集成测试与本次 7 项诊断集成测试全部通过；严格 Clippy
（`--all-targets --no-deps -- -D warnings`）、rustfmt 和 diff 检查通过，
normal/build/dev 依赖树无 Bevy crate。RecordingBackend 测试不代表原生 OS/GPU
呈现验证。

UI 只拥有查询/导航/分页、真实 receipt 和 Element/action bindings。catalog、
源扫描、Git、validation registry、资产持久化、原生输入、Loom runtime 和 GPU
资源均由宿主或其他模块持有。验证使用单个 `/private/tmp/prism-content-diagnostics-ui-*`
隔离 workspace/target，关闭 debug 和 incremental，离线双 job；不更改共享
Cargo.lock，不清理其他 agent 的构建目录，完成后清理该自有 target。
