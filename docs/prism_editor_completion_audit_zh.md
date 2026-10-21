# Prism 编辑器范围完成度审计

审计日期：2026-10-20
审计分支：`codex/prism-editor-completion-audit`
审计范围：当前 checkout 的全部 `pkg/prism_editor_*` 源码、16 个编辑器 workspace 包的 Cargo normal/build/dev 依赖，以及编辑器相关设计/实现状态文档。本文只新增审计记录，不修改业务源码，不把契约测试当作产品闭环证据。

## 结论

- 未发现编辑器包的 `bevy_*` normal/build/dev 直接或传递依赖。
- 未发现 `TODO`、`FIXME`、`todo!` 或 `unimplemented!` 未完成标记。
- 发现 3 个 `placeholder`，均为搜索框的用户提示属性，不是实现占位。
- 当前主框架设计/实现状态文档明确区分了契约切片和完整产品，未发现把契约测试直接宣称为完整编辑器交付的表述。
- 发现 2 项文档状态漂移，见下方 P1/P2；它们不改变运行时代码，但会误导实现计划或验证基线。

## P1：远程设计文档把已存在的 crate 写成“尚未创建”

证据：

- `docs/prism_editor_remote_design_zh.md:461` 写明 `prism_editor_remote`“本 crate 尚未创建”，并将 protocol、transport、codec、methods 等全部列为 PLANNED。
- 当前 crate 已被 workspace 纳入 `Cargo.toml:230`，清单位于 `pkg/prism_editor_remote/Cargo.toml:1`，源码模块在 `pkg/prism_editor_remote/src/lib.rs:26`，并公开 binary codec、JSON-RPC、endpoint 和 PIE 等实现；README 也在 `pkg/prism_editor_remote/README.md:3` 明确描述这些已存在能力。
- `docs/prism_editor_framework_implementation_zh.md:25` 将该包列为已验证的 JSON-RPC、binary codec、PIE、TCP 和受控进程桥，并明确保留“Prism 引擎专属 adapter”缺口。

影响：维护者可能按过时设计重新创建已经存在的协议层，或错误判断当前实现状态；同时该段与框架实现状态文档互相矛盾。

建议：将 `docs/prism_editor_remote_design_zh.md:461` 改为“已实现的通用协议/transport/codec/PIE/进程桥”和“仍未实现的安全、产品认证、Prism 专属 runtime adapter、生产部署”等边界；保留设计代码块为目标或示例时必须逐项标注，不要再写“本 crate 尚未创建”。

## P2：测试数量基线与当前源码不一致，且表内总数自相矛盾

证据：

- `docs/prism_editor_framework_design_zh.md:17` 声明 16 个包共 225 项测试。
- `docs/prism_editor_framework_implementation_zh.md:13` 的包级测试数表相加为 212；该表中 `prism_editor_app` 为 16、`prism_editor_viewport` 为 24（默认 22）。
- 对当前 checkout 的全部编辑器 Rust 文件执行 `rg -n '#\[(test|tokio::test|async_std::test)\]' pkg/prism_editor_*`，得到 304 个 test-family 属性；其中包括当前主任务未提交改动新增的测试属性，因此这是源码属性计数，不等同于某一次已编译 target 的最终 `cargo test -- --list` 数量，但足以证明 225/212 不能作为当前源码的无解释基线。

影响：文档中的回归规模无法复现，读者不能判断“225”是旧 checkout、可执行测试数、去重数还是其他统计口径；这削弱了编辑器完成度审计的证据链。

建议：在实现状态文档中删除固定总数，或记录精确命令、commit/工作树状态和 `cargo test -- --list` 的生成结果；包级表与总数必须使用同一统计口径。主任务提交后应重新生成一次，不要把 dirty 工作区的测试数写成已提交基线。

## P3：`placeholder` 命中是合法 UI 属性，不是未完成实现

命中位置：

- `pkg/prism_editor_ui/src/outliner.rs:47`：`SearchField` 的 `Filter scene` 提示。
- `pkg/prism_editor_ui/src/console.rs:33`：`SearchField` 的 `Filter console` 提示。
- `pkg/prism_editor_ui/src/command.rs:70`：`SearchField` 的 `Search commands` 提示。

这些调用位于完整的搜索字段构建和列表渲染路径中，不是 `todo!`、空函数或错误占位。`docs/prism_editor_framework_implementation_zh.md:79` 对此已有正确说明，不需要删除或替换。

## 依赖审计：无 Bevy 依赖

### 包清单

当前 workspace 的编辑器成员为 16 个，精确列在 `Cargo.toml:224-239`；对应清单文件为：

`pkg/prism_editor_app/Cargo.toml`、`pkg/prism_editor_collab/Cargo.toml`、`pkg/prism_editor_content/Cargo.toml`、`pkg/prism_editor_core/Cargo.toml`、`pkg/prism_editor_dock/Cargo.toml`、`pkg/prism_editor_domains/Cargo.toml`、`pkg/prism_editor_graph/Cargo.toml`、`pkg/prism_editor_panels/Cargo.toml`、`pkg/prism_editor_profiler/Cargo.toml`、`pkg/prism_editor_reflect/Cargo.toml`、`pkg/prism_editor_remote/Cargo.toml`、`pkg/prism_editor_shell/Cargo.toml`、`pkg/prism_editor_timeline/Cargo.toml`、`pkg/prism_editor_ui/Cargo.toml`、`pkg/prism_editor_viewport/Cargo.toml`、`pkg/prism_editor_world/Cargo.toml`。

### 检查结果

- `rg -n -i 'bevy' pkg/prism_editor_*/Cargo.toml` 无输出。
- 对上述 16 个包逐一运行 `cargo tree -p <package> --all-features --edges normal,build,dev`，合并输出后再执行 `rg -i '\bbevy(_|-|$)'`，结果为 0 行。
- `pkg/prism_editor_*/Cargo.toml` 中没有 `[build-dependencies]` 段；唯一的 `[dev-dependencies]` 位于 `pkg/prism_editor_viewport/Cargo.toml:13`，为 `futures-lite`。应用的 `futures-lite` 是 `pkg/prism_editor_app/Cargo.toml:27` 的可选 normal dependency，不是 build/dev 依赖。
- Rust 源码中的 `bevy_*` 仅出现在历史边界说明：`pkg/prism_editor_core/src/lib.rs:5` 与 `pkg/prism_editor_shell/src/lib.rs:6`；没有 `use bevy...`、`extern crate bevy...` 或 `bevy::` 代码引用。

因此，“编辑器 normal/build/dev 依赖树无 `bevy_*`”这一项有当前 checkout 的直接依赖树证据，不只是文档或源码文字搜索结论。

## 未完成标记审计

对 `pkg/prism_editor_*` 全部源码执行大小写不敏感扫描：

```text
rg -n -i --glob '*.rs' '\b(TODO|FIXME|todo!|unimplemented!|placeholder)\b' pkg/prism_editor_*
```

结果仅为上面列出的 3 个 `.placeholder(...)`；没有 `TODO`、`FIXME`、`todo!` 或 `unimplemented!`。这只证明没有这些显式标记，不等价于证明完整产品已经交付。

## 文档是否把契约测试误报为产品完成

当前没有发现这种直接误报：

- `docs/prism_editor_framework_design_zh.md:16-18` 明确称其为设计规格，列出仍待接入的完整场景绘制、原生桌面渲染/输入、Gizmo、生产存储和具体域 UI，并要求区分验证切片与产品流程。
- `docs/prism_editor_framework_design_zh.md:138` 明确写出不能用契约测试代替原生适配、runtime adapter、geometry/depth、设备输入和 Gizmo 证据。
- `docs/prism_editor_framework_design_zh.md:458-460` 和 `docs/prism_editor_framework_implementation_zh.md:3,7,9,81-89` 均说明当前不是完整桌面产品，下一条验收主线仍是原生窗口、Loom、真实场景、拾取、Inspector、Undo/Redo、保存/重载的端到端闭环。
- `pkg/prism_editor_viewport/README.md:233-237`、`pkg/prism_editor_world/README.md:67` 和 `pkg/prism_editor_remote/README.md:45-65` 也明确写出宿主适配、生产存储、runtime adapter 等边界。

需要修复的是上面的文档漂移和统计口径，而不是把当前契约测试扩写成产品完成声明。

## 主任务未提交改动

审计时以下文件处于 dirty 状态，按用户指示只读检查，未编辑、暂存或回退：

- `pkg/prism_editor_app/src/desktop.rs`
- `pkg/prism_editor_app/src/desktop_document.rs`
- `pkg/prism_editor_app/src/desktop_scene.rs`
- `pkg/prism_editor_app/src/lib.rs`

当前 diff 为 168 行新增、6 行删除，新增了 host-owned `SceneDocumentAdapter`、候选 scene prepare/discard/commit 流程及 2 个相关测试；例如实现位于 `pkg/prism_editor_app/src/desktop_scene.rs:60-82,305-336`，renderer 接口位于 `pkg/prism_editor_app/src/desktop.rs:115-130`。这些是“主任务未提交改动”，本审计不把它们计入已提交产品能力，也不以它们证明完整原生端到端闭环已经完成。当前文档仍将引擎专属 adapter、完整输入/渲染/编辑回写闭环列为未交付边界，口径是安全的。

## 审计限制

- 本次没有执行全仓库构建、测试或 GPU 验证；按请求只运行了限定范围的 `rg`、`cargo metadata`/`cargo tree` 和只读 diff/行号检查。
- 因此本文不重新确认文档中历史记录的 GPU、native smoke 或全包测试通过事实；这些只能作为文档声称的历史证据，不能由本次审计重新背书。
- 未触碰任何业务代码，也未清理或使用其他任务的 target 目录。

## 2026-10-21 后续实现闭环：原生交互取消失败时宿主 fail-closed

当前实现位于 `main`。本次只闭环一个独立的 Editor 宿主一致性缺口，不据此宣称编辑器产品已经全部完成。

此前 suspend、设备丢失恢复、窗口 resize/scale-factor change、失焦和 Escape 路径在原生交互取消失败后仍会继续后继生命周期转换。由于 Gizmo 取消需要把编辑前矩阵重新应用到场景，而该恢复过程可能失败，继续 resize 或初始化后继设备会让宿主在部分回滚状态上运行。

现在这些生命周期入口统一使用 fail-closed 语义：

- 取消成功时仍通过既有输入结果路径更新 redraw/prepared 状态。
- 取消失败时记录终止错误并请求退出；resize、失焦和 Escape 不再继续处理后继尺寸或输入状态。
- suspend 与设备丢失路径即使取消失败，仍会 suspend renderer、释放 native device 状态并清除 renderer pending/retry 状态。
- 设备丢失时若取消失败，不再初始化后继设备，避免把部分回滚场景迁移到新设备。

本次验证：

- `rustfmt --edition 2024 pkg/prism_editor_app/src/desktop.rs`：通过。
- `git diff --check -- pkg/prism_editor_app/src/desktop.rs`：通过。
- `cargo fmt --manifest-path pkg/prism_editor_app/Cargo.toml -- --check`：通过。
- `cargo test --manifest-path pkg/prism_editor_app/Cargo.toml --features desktop --lib native_interaction_cancel`：未完成；共享 `target` 被清理后触发依赖重建，编译 `prism_render_architecture` 时因磁盘空间耗尽中止，尚未进入本次 Editor 测试断言。该结果不计为测试通过，也不是本次代码的编译错误证据。
