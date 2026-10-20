# Profiler Loom 面板接入

本切片实现设计 §9 中 CPU/GPU 帧时间轴的 Loom 面板。采集由
`prism_editor_profiler::capture::CpuProducer` 与可选 `gpu::GpuProducer` 负责；
面板消费采集完成后插入的 `FrameStore`，输出标准 `Element`，不依赖 Bevy。
帧图资源预览、内存预算面板和 native app 循环属于其他切片。

## 公共集成增量（main 所有）

本切片只新增 `profiler.rs`、`profiler_view.rs`、两个测试与本文档。
`pkg/prism_editor_ui/Cargo.toml` 需要 main 合入以下增量，保留现有配置：

```toml
[dependencies]
prism_editor_profiler = { path = "../prism_editor_profiler" }

[dev-dependencies]
prism_ui_render_backend = { path = "../prism_ui_render_backend" }

[features]
profiler-gpu-tests = ["prism_editor_profiler/gpu"]

[[test]]
name = "profiler_gpu_panel"
required-features = ["profiler-gpu-tests"]
```

GPU 测试复用已经存在的 `futures-lite = "2"` 和 wgpu 30 dev-dependency
（`default-features = false`，`wgsl`、`metal`）；不需要修改既有 `gpu-tests`。
普通面板仅依赖 profiler 数据模型，不因此开启生产 GPU 采集。

`pkg/prism_editor_ui/src/lib.rs` 精确增量：

```rust
mod profiler;
mod profiler_view;

pub use profiler::{
    ProfilerAction, ProfilerController, ProfilerDomain, ProfilerFrameSelection,
    ProfilerPanelError, ProfilerPanelLimits, ProfilerSources, ProfilerTimeWindow,
    MAX_PROFILER_INDEXED_SPANS, MAX_PROFILER_VISIBLE_FRAMES,
    MAX_PROFILER_VISIBLE_SPANS, MAX_PROFILER_VISIBLE_TRACKS,
};
pub use profiler_view::{BoundProfilerPanel, ProfilerSpanVisual, ProfilerViewStats};
```

## app 侧调用契约

```rust,ignore
let mut panel = ProfilerController::default();
panel.publish(ProfilerSources {
    cpu: Some(Arc::new(completed_cpu_store)),
    gpu: completed_gpu_store.map(Arc::new),
});
let view = panel.build();
// 将 view.element() 作为 profiler dock 的真实内容贡献给现有 Loom 树。
// 保留这一份 view，直到它所对应的树被替换。
panel.handle_event(&view, hit_key, &UiEvent::Activate)?;
let next_view = panel.build();
```

CPU 与 GPU 的 `Arc<FrameStore>` 分别发布。快照不可原地修改；采集方发布新
Arc，旧视图继续持有自己的绑定。CPU 与 GPU 的同值 FrameId 不合并；
GPU query 与 Instant 未校准，切换域会重置 span 选择和时间窗口。
GPU 缺席时显示该域无已发布采集；能力拒绝与 readback 错误继续由采集方报告，
面板不填充示例时间或合成 GPU 帧。

`publish`、每次成功操作都会更换不可复用的 Arc 身份围栏；
`handle_event` 对旧视图及其他控制器视图返回 `StaleView`。错误命令不更换围栏。
直接 `view.dispatch` 只解析绑定；native 适配应走 `handle_event` 完成新鲜度检查。

`view.bindings()` 返回实际交互节点 key 和 `UiTrigger::Activate`。
宿主应在 mount/update、compute_layout 后遍历 **全部** Element（含未绑定节点），
与 `RetainedScene::visit_layout` 的前序结果逐一关联，再按绑定 key 筛选命中区域。
命中使用实际布局框，前景优先；不要由标签长度或轨道行号估算像素位置。
本面板使用标准 box/background 绘制，现有 draw-list renderer 可直接绘制这些条。

稳定 key 包括：

| 节点 | key |
| --- | --- |
| 根 | `profiler:panel` |
| CPU/GPU 切换 | `profiler:cpu` / `profiler:gpu` |
| 帧 | `profiler:frame:{cpu或gpu}:{frame_id}` |
| span | `profiler:span:{cpu或gpu}:{frame_id}:{span_id}` |
| 轨道标题 | `profiler:track:{index}` |
| 轨道绘图区 | `profiler:plot:{index}`（无 Activate 绑定） |

缩放、平移、Fit frame/span、Live 跟随和翻页均有实际可点击控件。
未启用的控件不产生绑定。native 滚轮、键盘、拖拽可以调用
`apply(ProfilerAction::Window(ProfilerTimeWindow { start_ns, duration_ns }))`；
interval 使用帧相对整数纳秒，必须非空、无溢出且位于帧内。
`ZoomIn/Out` 围绕窗口中心，`PanEarlier/Later` 移动半窗口，边界自动夹紧。
指针中心缩放可以由宿主通过真实 plot 宽度计算区间后提交 Window。

选择帧会暂停跟随；新快照保留仍存在的暂停帧、时间窗口和轨道/跨度页。
开启跟随会选择 sequence 排序后的最新帧并 fit；源中帧消失则重新选最新帧。
选择 span 后显示真实名称、lane、parent、start/duration，Fit span 使用该跨度区间。

## 图形与资源边界

帧柱图按当前图表页的最大 duration 归一化；零/极短帧保留最小可见指示高度。
轨道按 lane、父跨度深度分组；同组重叠跨度进行区间分行，不互相遮盖。
span 条的位置和宽度来自真实帧相对时间，并在当前窗口边界裁剪。
零时长 span 画 0.2% 宽的瞬时标记，末端标记向内夹紧；详情仍显示 0 ns。
CPU 条蓝色、GPU 条绿色、选中条金色。时间轴无自造时长，也不混排两个时钟域。
浮点仅用于最终显示比例；整数纳秒仍是操作、选择与裁剪的依据。

| 资源 | 默认 | 硬上限 |
| --- | ---: | ---: |
| 每次物化帧柱 | 16 | 32 |
| 每次物化轨道 | 8 | 16 |
| 每次物化 span 条 | 128 | 256 |
| 单选中帧索引 span | — | 65,536 |

`ProfilerPanelLimits` 在创建控制器时验证，不允许 0 或超过硬上限。
帧、轨道、当前窗口中的 span 都有独立页导航和总数，不静默截掉不可达数据。
测试用 2 帧/2 轨道/3 span 的窗口访问全部 1000 个 span，每棵树小于 130 节点。
超出索引上限时保留帧柱图并显示 `IndexCapacity`，不展示部分时间轴冒充完整帧。

索引最多保留 O(N) span 索引，分组、parent 深度与重叠分行耗时 O(N log N)，
parent 深度计算使用迭代方式。build 扫描可见轨道中的跨度，最坏受单帧索引
上限限制；Element/绑定/可视条数量只随配置窗口增长。不递归克隆 SpanTree。
宿主持有的整个 FrameStore 仍受其 `ProfilerLimits` 约束，面板不另外复制整帧。
宿主按 dock 可用空间选择轨道限额，并负责 dock 的裁剪/滚动布局。

## 验证与接线边界

本次验证在独立临时 Cargo harness 中完成：引用真实新源文件、既有
`action.rs`/`support.rs` 公共事件契约与仓库内真实依赖；不需要修改 main
独占的 lib/Cargo。并行 content 修改期间未将整个 editor_ui 或 native app
的通过状态算作本切片证据。编译使用单 job、debug=0、incremental=false。

- CPU/面板 session `77437`：8/8 通过。实际 Instant 帧 3,016,875 ns；
  嵌套 span、布局比例、选中像素、缩放/裁剪、分页全可达、旧事件拒绝、
  CPU/GPU 域分离、0/u64::MAX 边界及超预算显式拒绝。
- 强制真实 GPU session `67035`：1/1 通过，Apple M2 / Metal；
  query/readback frame/span 均 228,667 ns，256 个计算输出独立核对通过，
  此实际 FrameStore 驱动绿色 span 像素、点击选择与 Fit span。
  GPU 测试要求真实 adapter 与 timestamp 能力，无能力直接失败，不跳过。
- 新面板库与两个测试执行 `cargo clippy ... -- -D warnings`，无警告。
- 最终联合复验 session `48248`：CPU/图形 8/8、真实 GPU 1/1；
  CPU frame 3,075,417 ns，Apple M2 / Metal frame/span 228,584 ns。
- 像素证据来自 Loom RetainedScene 的 CPU 参考栅格；GPU 证据来自真实采集
  query/readback。这里不声明 native 窗口或 GPU UI renderer 挂接已经完成。

main 合入公共导出和依赖后，可在其验证 target 中复跑：

```sh
cargo test -p prism_editor_ui --test profiler_panel
PRISM_REQUIRE_GPU=1 cargo test -p prism_editor_ui --features profiler-gpu-tests --test profiler_gpu_panel -- --nocapture
cargo clippy -p prism_editor_ui --lib --test profiler_panel --features profiler-gpu-tests --test profiler_gpu_panel -- -D warnings
```

剩余集成是 main 的公共导出/依赖与 native app 的 dock 贡献、采集快照发布、
实际布局命中和事件转交；本切片没有改 app、全局设计文档或其他面板。
