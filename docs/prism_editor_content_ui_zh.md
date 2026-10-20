# 内容浏览器 Loom UI 与控制器

本切片实现设计 §11 的有界内容目录/列表及控制器。改动仅在
`pkg/prism_editor_ui/**` 和本文；不修改 `prism_editor_content`、
`prism_editor_app` 或既有 `EditorAction`。目录数据来自真实 `ContentCatalog`
公开 API，无新增 Bevy 依赖。

## 公开接入 API

```rust
use prism_editor_content::ContentCatalog;
use prism_editor_ui::{ContentBrowserController, UiEvent};

let mut catalog = ContentCatalog::new(); // 应用持有并接收 source 原子发布
let mut controller = ContentBrowserController::default();
let mut view = controller.build(&mut catalog)?;
// ui.update(view.element());
// 原生后端依实际布局对 view.bindings() 的 (key, UiTrigger) 建立命中表。
controller.handle_event(&mut catalog, &view, "content:home", &UiEvent::Activate)?;
view = controller.build(&mut catalog)?;
// ui.update(view.element());
Ok::<(), prism_editor_ui::ContentBrowserError>(())
```

`BoundContentBrowser` 拥有真实 `Element` 与确定性有序的绑定，不是单独的 UI
metadata 模型。`element()` 可挂入现有 dock body；`bindings()` 返回实际树上
的交互键与事件类型；`dispatch()` 只生成自有 `ContentBrowserAction`。
推荐由 `handle_event` 完成绑定解析、视图新鲜度校验和 `apply` 执行，随后重建
并交给 Loom reconciliation。原生 app 不需要把内容动作加入 `EditorAction`。

| Element 键 | 事件 | 控制器行为 |
| --- | --- | --- |
| `content:search` | `Commit(Text)` | 路径、kind、tag 搜索 |
| `content:kind` / `content:tag` | `Commit(Text)` | 精确 kind / 规范化 tag 过滤；空值清除 |
| `content:vcs` | `Commit(Text)` | VCS 过滤；空值或 `all` 清除 |
| `content:favorites-only` / `content:pinned-only` | `Activate` | 切换过滤条件 |
| `content:folder:<canonical path>` | `Activate` | 打开该文件夹 |
| `content:home` / `content:up` | `Activate` | 根目录 / 上级目录 |
| `content:previous` / `content:next` | `Activate` | 按 catalog cursor 翻页 |
| `content:asset:<AssetId.raw()>` | `Select { mode }` | 替换、添加、切换或当前页范围多选 |
| `content:favorite:<id>` / `content:pin:<id>` | `Activate` | 显式动作读取最新记录、修改标记并调用 `catalog.update` |
| `content:clear-selection` | `Activate` | 清除跨页选择 |

禁用按钮仍展示，但没有绑定，不能分发动作。VCS 输入接受 `unknown`, `clean`,
`untracked`, `added`, `modified`, `deleted`, `renamed`, `ignored`, `conflicted`；
未提供状态按 catalog 的 `Unknown` 展示，不推断成 Clean。行中显示实际 staged
标记。kind、tag、搜索使用 catalog 的过滤语义，大小写归一化为 ASCII 规则，
不宣称实现 Unicode 全量 case folding。文件夹在过滤后仍有匹配后代才显示。

## 分页、身份和更新语义

默认请求 64 行，最终不超过 UI 上限 512 与 catalog `max_page_size`。
仅物化当前文件夹的当前页，空目录/无匹配有明确空态。历史最多保存 64 个
上一页 cursor，超过后丢弃最老项；Home 可回到根目录第一页。选择最多 4096 个
稳定 AssetId，跨页/过滤/目录导航保留；`Extend` 选取当前页锚点到目标的闭区间，
锚点不在当前页时仅选目标。`ClearSelection` 显式清除。

资产行 key 来自 ID，重命名不改变身份；文件夹 key 使用精确规范路径。收藏和
固定的更新保留路径、kind、tags、dependencies 等其他字段，通过 catalog 的
更新接口维护索引与 generation。过滤条件或目录改变重启分页；catalog 发布、
Git 状态更新或收藏修改造成 generation 变化时清除旧 cursor，裁剪已删除的选择，
打开的目录已删除则回到根目录。

旧视图事件返回 `StaleView`，宿主应重建 UI 后接受下一次输入；不重放旧点击。
文本输入限制 4096 UTF-8 bytes，非法路径、错误值类型、未知 VCS 文本、零页大小
和不可见资产选择均返回明确错误。收藏/固定采用目标状态动作，供宿主使用
`apply` 时重试同一状态而不是盲目翻转。

## 所有权和原生 app 剩余边界

UI 控制器只持有 query、cursor 历史、当前页身份及选择；绑定树是拥有值。
ContentCatalog、source 扫描/原子发布、Git worker、偏好持久化由主 agent/应用
拥有；Loom runtime、命中测试、焦点、IME 草稿、修饰键和窗口由 app agent Erdos
拥有。此切片不启动线程、不创建窗口、不分配 GPU/纹理资源。

Erdos 需在原生布局里挂载 `view.element()`，按其实际布局与 `bindings()` 建立
命中表；按钮提交 Activate，行提交 Select，四个输入提交 Text Commit。路由到
独立内容控制器后重建并 reconcile；source/catalog 发布时同样刷新，且不能仅
依赖既有 editor model revision 判断是否重绘。选择的只读 `selection()` 可供
资产 Inspector/拖放的上层接入，但这些行为不属于本切片。

当前实现列表，未添加缩略图或缓存就绪标记。后续若接缩略图，需要应用提供真实
缓存条目及资源生命周期；本切片无伪缩略图、TODO、空实现。也不宣称完成设计 §11
中的集合、依赖图、导入/导出、Cook 或整个原生内容浏览器集成。

## 验证

`pkg/prism_editor_ui/tests/content_browser.rs` 使用真实 catalog 和
`Ui<RecordingBackend>`：实际 Element key → UiEvent → 控制器 → catalog.update
→ 新树 → Loom update，检查 SetText、Remove、布局尺寸及稳定节点复用，
而非仅比较控制器 metadata。包含过滤组合、Unicode/空格目录、翻页、跨页多选、
过期事件、删除刷新、空态及页/历史边界。

2026-10-20 在独立临时 Cargo workspace 中直接依赖主 checkout 的 UI crate，
使用同一 target 验证，避免更新共享 Cargo.lock：

```text
cargo test --offline --manifest-path <临时workspace>/Cargo.toml -p prism_editor_ui --all-targets --jobs 2
cargo clippy --offline --manifest-path <临时workspace>/Cargo.toml -p prism_editor_ui --all-targets --no-deps --jobs 2 -- -D warnings
```

结果：既有 20 项单元测试、新增 8 项集成测试全部通过，严格 Clippy 通过；
`rustfmt --check`、`git diff --check` 通过。该独立依赖图未发现 `bevy` 或
`bevy_*`。测试证明无变化重建零 backend ops、收藏/固定产生实际文本更新、
过滤移除行、重命名复用既有节点，以及所有绑定目标有非零真实布局尺寸。
这些是 headless Loom 验证，未运行或宣称完成原生窗口的输入/绘制验证。
