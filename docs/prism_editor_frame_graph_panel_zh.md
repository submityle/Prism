# FrameGraph 调试与预算面板

`prism_editor_ui` 的 `FrameGraphPanelController` 将不可变的
`Arc<prism_editor_profiler::GraphCaptureReport>` 转为实际 Loom Element。
图中的 pass、资源、访问与依赖均来自已校验的报告；面板不生成演示图。

## 宿主接入

```rust,ignore
let mut panel = FrameGraphPanelController::default();
panel.publish(Some(report_arc))?;
let view = panel.build();
// 将 view.element() mount/reconcile 到 dock body。
// 遍历 view.bindings()，以真实 retained layout rect 注册 hit target。
panel.handle_event(&view, target_key, &UiEvent::Activate)?;
let replacement = panel.build();
```

`BoundFrameGraphPanel` 保留显示时的 report Arc。`handle_event` 同时检查
report 的 `Arc::ptr_eq` 与 controller/view 修订令牌，再检查该 Element
实际发布的 binding。另一 controller、相同 frame ID 的另一 Arc、重复发布
同一 Arc、以及成功导航前的旧 view 都拒绝为 `StaleView`。失败的 publication
或无效目标不会部分更新状态或令牌。宿主必须为新布局同步替换 view 与 hit
targets；不能只保留 frame ID。`dispatch` 仅做纯 binding 查询，宿主事件入口
应使用 controller 的 `handle_event`。

`publish(None)` 清空显示。`apply` 是可信宿主主动导航接口；pass/resource
编号是本次报告中的表下标，不是跨报告稳定 ID。发布报告会清空选择与页码。

根模块由集成方添加 `mod frame_graph; mod frame_graph_view;`，导出：

```rust,ignore
pub use frame_graph::{
    FrameGraphPage, FrameGraphPanelAction, FrameGraphPanelController,
    FrameGraphPanelError, FrameGraphPanelLimits, FrameGraphSelection,
    MAX_FRAME_GRAPH_PAGE_ROWS,
};
pub use frame_graph_view::{
    BoundFrameGraphPanel, FrameGraphAccessVisual, FrameGraphBudgetVisual,
    FrameGraphDependencyVisual, FrameGraphPanelStats, FrameGraphPassVisual,
};
```

运行时复用已有 `prism_ui`、`prism_editor_profiler` 依赖，无 Bevy。生产构图
测试额外需要如下 dev-dependencies；`render-graph` 仅用于测试构图 adapter：

```toml
prism_editor_profiler = { path = "../prism_editor_profiler", features = ["render-graph"] }
prism_render_architecture = { path = "../prism_render_architecture" }
prism_render_driver = { path = "../prism_render_driver" }
prism_render_graph = { path = "../prism_render_graph" }
```

## 显示、导航与边界

| 分页 | 默认行数 | 顺序与内容 |
| --- | ---: | --- |
| Passes | 8 | 编译执行序，graphics/compute/transfer 三条队列轨道 |
| Resources | 6 | 报告原资源表，真实生命周期与 bytes |
| Dependencies | 8 | 报告规范依赖序，显式依赖/资源 hazard/queue transfer |
| Accesses | 8 | 选中 pass 的访问序，或选中资源的所有使用者及 SSA |
| Budgets | 6 | 报告规范预算类别序，consumed/limit |

各页大小必须为 1..=32；五个页码彼此独立。报告准入另外检查
`GraphCaptureLimits::default()` 的 pass/resource/access/dependency/budget
总数与单条文本长度，避免用巨大报告绕过显示分页。准入不重复编码报告。
Element 仅构造当前页面，底层报告 Arc 保留完整数据；不是每次把整张图
铺成隐藏节点。遍历过滤仍受报告准入上限约束。

图画布宽 800 像素，横轴表示当前页的编译顺序；它不是时间轴。节点过窄时
不画溢出的文字，点击后仍能查看真实详情。每条当前依赖的两端都在当前 pass
页时才绘制有界折线；跨页边明确计数，并可在依赖行选择后点击 prerequisite/
dependent pass 跳转到对应页。资源选择显示 owner/slot/frame、buffer usage
或 texture extent/format/dimension/mips/samples、alignment 与生命周期。
访问行显示原 access ordinal、access kind、input/output SSA，并提供 pass
和 resource 双向导航。选择依赖保持最近 pass/resource 的关系过滤上下文。

critical path 直接使用报告分析结果；只有每个 pass 都有采集时长且来源有效时
才展示。缺失、部分 timing 或空图不会把拓扑长度冒充关键耗时。CPU 与 GPU
来源明确显示；面板不会制造 timestamp。它不补推断跨队列并发或墙钟帧耗时。

预算高亮依据报告 `consumed > limit`，等于预算不超限；比值裁剪到 0..=1，
零预算分别处理 0/0 与正数/0。logical bytes、alias heap bytes 和预算账本
是报告记录值，不代表实时驻留 VRAM 或当前 GPU 对象。

## 验证与范围

`tests/frame_graph_panel.rs` 使用真实 `FeatureCatalog`、`ResourceTemplates`、
`prepare_runtime`、`capture_runtime_graph` 与二进制 codec 往返，再驱动
retained layout、实际 raster、选择/分页/身份围栏、SSA 与预算断言。计时用
`Instant` 测量测试 CPU 工作，仅检验 measured-path 展示契约，不声称这是
运行编辑器或 GPU 提交的采集证据。异常编码测试从生产报告编码派生。

本切片已实际运行 workspace 的
`cargo test -p prism_editor_ui --test frame_graph_panel --offline`，7/7 通过。
断言包括真实连线 retained 几何与 raster 颜色、选择后 pass 像素变化、
重复 build 无 retained 更新、32 pass 页内布局、跨页依赖跳转，以及超界
报告拒绝时原 publication/view 仍有效。隔离 manifest 对同一真实 UI 根和
测试也运行 7/7 通过；未使用替代事件模型或替代渲染图类型。

隔离真实共享事件模块和本面板两个源码的严格 Clippy（lib 与上述集成测试，
`--no-deps -- -D warnings`）已通过。workspace 同样的严格命令在本次检查时
仍被范围外 `graph_gesture.rs` 两处 `semicolon_if_nothing_returned` 阻塞；
本切片没有修改或禁用该检查，最终 crate 联合 lint 由根集成方复核。

此切片提供面板与宿主 API；native app 挂载、渲染器持续发布报告和实际 GPU
中间资源预览由对应所有者集成。报告不持有可预览 texture/buffer 句柄，
此面板没有实现中间缓冲预览。
