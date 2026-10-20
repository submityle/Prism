# 内容网格、集合与真实缩略图 UI

本切片只修改 `pkg/prism_editor_ui/**` 与本文。沿用既有
`ContentBrowserController`、独立 `ContentBrowserAction` 和真实 Loom
`Element`/`RecordingBackend` 链路，不扩展 `EditorAction`，不修改 app、content、
viewport 或 Loom。缩略图生产来自 content 的公开 `thumbnail` API；GPU 绘制使用
主线程提供的 `prism_editor_viewport::image` compositor。没有引入 Bevy 依赖。

## 网格与集合

`ContentBrowserAction::Layout(ContentLayout::Grid { columns })` 选择 1 至 8 列；
卡片宽 160px，按真实行节点布局，List/Grid 按钮发出显式 action。页面仍受
`MAX_VISIBLE_ROWS` 与 catalog page limit 约束；布局切换保留选择与查询。

`CreateCollection`、`RenameCollection`、`DeleteCollection`、`OpenCollection`、
`AddSelectionToCollection`、`RemoveSelectionFromCollection` 是唯一集合写入入口。
树中的输入通过 Commit、按钮通过 Activate、资产通过 Select 路由到 controller。
集合只读投影由 `collections()` / `active_collection()` 提供；至多 64 个集合，
每个至多 4096 个 AssetId。名字非空且受共享文本字节上限约束，成员添加先检查
并集容量，再原子写入。集合 ID 在单个 controller 生命周期内单调分配。

集合跨文件夹应用 search/kind/tag/VCS/favorite/pinned 过滤，只查询有界成员集，
按 path、AssetId 排序并分页；不扫描整个 catalog。重命名后仍保留成员身份，
已删除资产的 ID 留在集合但不生成行。catalog 发布重置分页并剪除已删除选择；
旧 view 的事件继续被拒绝。当前没有集合序列化/恢复 API，跨会话持久化未完成。

## Erdos 原生接入协议

1. 保留 controller 与当前显示的 `BoundContentBrowser`；先 `build(&mut catalog)`
   得到 `visible_assets()`。宿主为有真实缩略图来源的 ID 构造完整
   `ThumbnailKey { asset, revision, source, width, height }`；source 为绝对路径，
   revision 来自宿主实际源码版本，UI 不以 catalog generation 或猜测 mtime 替代。
2. `ContentThumbnails::new(shared_service)` 共享已有服务；调用
   `sync_visible(&controller, keys)`，事件循环在 `needs_poll()` 时 tick `poll()`，
   状态改变后重建。请求与 poll 不做 caller 文件 I/O。缺少来源的资产不提交 key。
3. `build_with_thumbnails(&mut catalog, &thumbnails)` 返回真实布局树。
   `view.images(): &[ContentImageVisual]` 中每项含 `visual_key: String`、
   `request: ThumbnailKey`、`image: Arc<Thumbnail>`；只在 Ready 时存在。
   `image.rgba8()/width()/height()` 是实际服务数据。
4. 对应树节点为 `Element::custom("prism-editor-content-image")`，key 精确为
   `content:image:{asset.raw()}`；宽高保持图像比例，最长边布局为 128px。
   Erdos 按 key 找到本帧真实 Loom 布局，累加父变换并应用滚动、DPI、祖先裁剪和
   opacity，得到 physical rect/clip/opacity。不要把后端 ID 当稳定 asset ID。
5. 按完整 request 身份维护 opaque `ImageHandle`，实际 RGBA8-sRGB 上传后，
   使用 `ImageQuad { image, rect, clip, opacity }` 调用 `prepare`，再
   `encode(&mut encoder, &target_texture)`。未 Ready 不上传；没有布局、被裁出或
   已换页的 visual 不提交。Custom 节点仅作布局锚点，本身不构成图片绘制完成。

宿主继续将实际 semantic key/event 交给
`handle_event(&mut catalog, &displayed_view, key, &event)`；accepted action 或
catalog 发布后重建/同步 keys。catalog generation 不符时整个缩略图投影被隐藏，
直到宿主同步当前 key，避免发布后误用旧页面。宿主应在更新树的同一轮更新图像
投影，避免继续使用旧 `BoundContentBrowser` 中的 Arc 与布局。

## 资源所有权与边界

| 资源 | 所有者与释放条件 |
| --- | --- |
| catalog、来源版本、真实路径、扫描/Git | content / app；UI 只借用公开 API |
| 集合、选择、页游标、Element/bindings | UI controller / displayed view；宿主替换旧 view |
| 当前页请求与 Ready Arc | `ContentThumbnails`；sync 移除离页/旧 key，clear 释放全部 |
| 共享缩略图缓存、worker 数量/源像素限制 | `ThumbnailService`；UI 共享同一实例，Busy 请求排队 |
| texture/opaque handle、布局投影、提交 | app / viewport image renderer；离页/换 revision/关项目释放 |

UI 最多保留当前页 512 个 key、32MiB 解码像素，可用 `with_budget` 缩小。
超出像素预算的结果变为真实 Limit 失败；失败可显式 `retry(asset)`，Busy 保留
Queued。非法/重复/非可见 key 在修改之前拒绝。`retained_bytes()` 不包括服务缓存、
旧 displayed view 或宿主克隆的 Arc，也不包括 GPU 纹理。宿主关闭项目需同时
清除页面、服务与纹理；`service.clear()` 不能撤销其他地方已持有的 Arc。

本切片不宣称完成原生窗口集成。Erdos 仍拥有实际桌面输入、布局投影、图片提交、
project lifecycle 与窗口验收；本文给出可直接集成的公开 API。

## 实际验证

复用唯一自有小 target，隔离 manifest 直接指向此 crate 的真实 lib/tests，
关闭 debug/incremental，offline、jobs=2；不构建整个共享 workspace。

- session 51944：20 unit + 8 browser + 7 diagnostics + 7 grid，共 42 项通过。
  包括真实 Loom mount/reconcile、网格几何、同树重复 reconcile 无 backend 写入、
  集合事件增删改查/分页/过滤/选择/favorite，以及真实 PNG 缓存命中、revision
  变更失败与重试、catalog 发布过期、key 原子校验、字节预算及离页请求释放。
- session 80128：显式 `gpu-tests` 测试
  `real_png_to_loom_layout_to_gpu_pixels_clip_opacity_and_texture_release` 通过。
  Apple M2 / Metal 物理 GPU，无 adapter 时失败而非跳过。真实 2×1 红/绿 PNG
  经服务解码、`images()` key 对应实际 Loom 布局 `[0, 324.8, 128, 64]`，
  上传 compositor、绘制到 sRGB attachment 并回读：左红/右绿，外部黑；
  右半裁剪后左侧黑，opacity=0.5 的绿色 sRGB 值约 188；同几何 prepare 不重复
  上传，remove 后 texture usage 从 `(1,8)` 变为 `(0,0)`。

普通测试：`cargo test --offline --manifest-path <isolated>/Cargo.toml -p
prism_editor_ui --all-targets --jobs 2`。GPU 测试另加 `--features gpu-tests
--test content_grid real_png_to_loom_layout_to_gpu_pixels_clip_opacity_and_texture_release
-- --nocapture`。测试使用 std 临时目录写真实 PNG，Drop 精确删除夹具。

最终 session 9580 将 `--all-targets --features gpu-tests` 合并执行，43 项全部
通过（20 unit、8 browser、7 diagnostics、8 grid/GPU）。严格 Clippy
`--all-targets --features gpu-tests --no-deps -- -D warnings` 通过，normal
依赖树无 Bevy。验证后清理本切片唯一自有 isolated target，不清其他人的 target。
