# 编辑器文件热迭代 provider

本模块负责文件依赖包的有界异步准备、精确修订围栏与 last-good 发布，提供可执行的 Loom 和 shader provider。生产模块为 `prism_editor_domains::{reload,reload_loom,reload_shader}`，always-on，不依赖 Bevy。原生 app 的文件监听、调度和诊断展示由 host 接线。

## 调用与原子发布

1. host 为一个 live resource 建立 `ReloadPipeline`，显式注册根目录下全部依赖的相对文件路径、schema 原始字节及 `ReloadLimits`，用 `slot(initial)` 绑定初始 live resource。路径不可重复，不接受绝对路径和 `..`，实际解析后的路径必须在根目录内且为普通文件。
2. 原生 watcher 将修改、删除、rename 后相关依赖变化交给 `notify(relative_path, now)`。所有已注册路径共用一个有界 dirty set；同路径的一万次事件仍只有一个条目。每次事件立即废弃旧请求和结果，不必等 debounce。`start(now, false)` 在 quiet period 或 max-delay 到期后启动一个 worker；`true` 显式强制开始。初始 lane 为 dirty。
3. `ReloadRequest::poll` 非阻塞返回 pending、完整候选或明确错误。worker 真实读取所有注册文件，在源文件/总字节预算内生成不可变原始字节快照，调用 provider 的真实 parser/compiler。worker 不接触 live tree、GPU handle、document 或 command stack。失败不会自动无限重试；host 可显示诊断、等待新事件或显式重新 notify。
4. `publish_loom` 或 `publish_shader` 接受候选。发布校验 pipeline 身份、source epoch 身份和 schema 身份，并在 fallible host build 前后重新读取、逐字节比较所有依赖；只有全部成功才调用 infallible commit。schema 比较不是 hash，A→B→A 也拒绝 A 的旧候选。另一 pipeline 的 candidate/slot 被拒绝。未收到事件的磁盘修改同样可以在发布时被发现。

`ReloadLimits` 限制依赖数量、每文件/schema 字节、全部文件字节、worker/未消费结果数、Loom 节点/深度、shader 输出字节、debounce、max-delay 和 timeout。默认依赖64、单文件1MiB、合计4MiB、并发2、节点4096、深度32、输出4MiB。合法预算本身有硬上限。worker、request、已取出的 prepared candidate 共同持有许可：未消费的成功/错误结果都占容量；丢弃/取消 request 后仍需 worker 真正退出才释放。消费取消/超时会关闭结果通道，防止已到达的成功包继续滞留。轮询已消费 handle 返回 pending，不重复交付错误。

`cancel` 与 timeout 是协作式：文件步骤/解析循环/编译调用前后检查。标准文件 I/O、WESL/Naga 单次编译与 host GPU build 不能强行抢占；它们返回后结果仍会被拒绝。此模块不声称进程隔离、编译器硬内存限制或立即停止系统调用。编译器可能在产生受限输出之前分配临时内存；输入与完成结果有明确上界。worker panic 转为 `WorkerFailed`。退出不在 UI 线程 join 阻塞。

发布的两次逐文件读取会产生 I/O，host 应在资源构建/发布调度中计入帧预算。第二次读取是约定的发布观察边界；多文件系统没有事务快照，后续磁盘写入需可靠 watcher 继续送达。根目录检查不是对抗恶意并发 symlink/文件类型替换的安全沙箱。`notify` 接受已注册相对路径；绝对 watcher 路径应由 host 相对化并过滤，队列不得无限累积。模块没有另起无限 watcher 队列。

## 可执行 Loom 与保状态边界

`LoomReloadProvider::new(view, styles)` 的精确 schema 为 `LOOM_LITERAL_SCHEMA`（`prism.loom.literal/1`）。它创建真实 `prism_ui::Element`，候选 `plan(&HotReloader<T>)` 调用原生 `plan_for`，`publish_loom` 调用原生 `reload`。状态 `T` 不需 Clone，滚动位、输入文本、动画进度留在 `HotReloader` 的稳定 NodePath 状态槽。删除的状态按原生 reload plan 清理。候选与 live tree 的节点、深度和同级身份唯一性在 plan 前检查；不通过就不修改 live tree。

现有完整 Loom parser 是私有 Rust proc macro，会执行编译期 Rust 语法。本 provider 显式只支持可在运行期安全解析的数据子集，遇到不支持语法直接报错；不会把字符串占位当作已编译 UI：

```text
box {
  key: "panel";
  class: "card";
  style: { opacity: 0.8; };
  text("新标题\n保留输入状态") { key: "title"; }
  custom("project.widget") { key: 7; }
}
```

支持 box/text/custom、嵌套 child、字符串或 i64 key、显式 `stable_id: "identity";`、class 列表、inline style、`//` 注释和可选根尾分号。字符串采用 JSON 转义并保留 Unicode；没有 lossy 解码。NodePath 优先使用 StableId，然后 key，最后位置及 kind。若初始树来自 `loom!`，应显式保留其 StableId（宏根为 `""`，首层为 `"0"`、`"1"` 等），或从本 provider 初始化 tree；不能去掉初始 StableId 后仅靠相同 key 假称状态仍可匹配。节点插入/重排时应保留既有语义身份，改变身份代表重新创建状态。真实宏树重排回归检查显式 ID 保留两节点的独立输入态。依赖样式文件使用显式 literal schema：

```text
class "card" {
  width: px(320);
  background_color: rgba8(24, 30, 42, 255);
  flex_direction: column;
}
```

支持长宽/min/max、padding/margin/gap、字体大小/权重、border radius/width、flex basis/grow/shrink、定位四边、opacity、颜色、display/flex_direction/position/justify_content/align_items。长度为 px/percent/auto；颜色为 rgba8；数值必须有限且在属性范围内。class 基础属性解析为真实 Element 样式，inline 优先于 class。拒绝重复同级 key、重复属性/样式/class 定义、未知 class 和错误类型。动态 Rust 表达式、signal、event closure、for_each、style token、state selector 和 breakpoint 不在此 schema；它们继续走现有编译/运行时接口，不能误报为本 provider 已热更新。

host 将 `slot.get().current()` 交给实际 `Ui::mount/update`。provider 不复制 document，也不接收 `CommandStack`，更不清空历史；真实命令栈回归覆盖成功/失败 reload 后既有 undo 和 redo 均可继续。

## WESL / WGSL 与真实 GPU 替换

`ShaderReloadProvider::wgsl(path)` 使用 `WGSL_SCHEMA`；`wesl(root_module, modules)` 使用 `WESL_SCHEMA`。WESL module→文件映射必须显式完整注册，通过 WESL 0.4.2 的真实 import linker/mangler 编译。仅允许 captured bundle 的 VirtualResolver，缺失依赖明确错误，不从未注册磁盘路径偷偷取文件。输出交给 Naga30 parser/validator，entry points、stage、workgroup size 来自实际 Naga module。产物含真实 RHI `ShaderModuleDescriptor`。

`publish_shader(..., build)` 在 host build closure 中创建 GPU candidate，最后 swap live handle 并返回旧资源供 host 在正确提交边界回收。host 必须使用真实 backend fallible validation/error scope，并只在成功时返回资源。Naga 校验成功不足以证明设备特性、bind layout、pipeline 格式匹配。物理 GPU 回归使用 wgpu error scope、compute dispatch 与 readback，预期输出1→2；有效 WGSL 却不匹配设备 layout 时拒绝发布，后续 dispatch 仍输出2。未接 host GPU callback 的 CPU provider 不能声称完成视口热替换。

本模块复用 WESL/Naga 与既有 RHI 描述，没有重建 renderer、pipeline cache 或 asset server。资产导入/依赖传播应复用 `prism_asset` 的既有 last-good loader/`replace_runtime`，关卡变更需经过 domain 事务与冲突策略；PIE 组件编辑和 dylib 系统替换仍由对应 runtime/构建层负责。本交付不将这些未接入的流程记为已完成。

## 集成与验证

根模块由主线程接线。生产依赖：prism_ui、prism_ui_hotreload、prism_ui_style、prism_render_driver、serde_json1、naga30（wgsl-in）、wesl0.4.2（default-features=false）。回归 dev 依赖：prism_editor_core、wgpu30（std/wgsl/metal，其他平台按后端启用）、futures-lite2。生产模块 always-on，物理 GPU 用例为显式 ignored，不能在没有 GPU 时静默通过。

验证使用本 checkout 当时可用的 Cargo target；不要清理由其他任务共享的 target。执行期间磁盘剩余空间不足，后续 lint 必须先确认空间并使用协商好的共享 target。

```sh
cargo test --offline -p prism_editor_domains --test reload --test reload_shader
cargo test --offline -p prism_editor_domains --test reload_shader -- --ignored --nocapture
cargo clippy --offline -p prism_editor_domains --all-targets --all-features --no-deps -- -D warnings
```

2026-10-21 在真实 checkout 执行 `cargo test --offline -p prism_editor_domains --all-targets --all-features`：库测试 13 项、Loom/provider reload 集成 13 项、shader 集成 3 项通过；真实 GPU 用例 1 项按 `ignored` 标记未执行。`cargo fmt --check -p prism_editor_domains` 和相关 `git diff --check` 通过。严格 Clippy 因磁盘写满、无法写入依赖 metadata 而中止，尚无通过结论。物理 GPU dispatch、严格 Clippy 和 host 原生 watcher/诊断接线仍待验证/集成。
