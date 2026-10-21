# Editor Scene Session：作者文档到 ECS 与视口

## 范围

`EditorSceneSession` 实现 opt-in `EditorSceneMesh` profile 的作者文档、真实 Prism ECS 和 `DesktopScene` 投影。CLI 创建/打开 `SceneProject` 时使用同一 catalog 和 session；空工程保持为空，不注入演示对象。它不是任意组件到几何的隐式转换器，也不代表完整 Editor 已交付。

`EditorController` 是唯一作者数据权威。场景与 ECS 只读投影，不另建 Undo/Redo。`world()`、`bridge()`、`scene()` 只提供只读访问；模型变更须经 controller 命令。明确摄像机通过 `with_camera` / `set_camera` 安装；默认使用单位 clip-space 矩阵，不推断项目相机或物理规则。

## 构造与宿主预算

构造 API 现为 `EditorSceneSession::new(&mut EditorController) -> Result<Self, DesktopError>` 与 `EditorSceneSession::with_camera(&mut EditorController, SceneIdMatrix, SceneIdDepth) -> Result<Self, DesktopError>`。可变借用用于安装宿主策略，而不是把 ECS 投影写回作者模型。先执行完整投影验证/装配并 stage scene，再以最后一个可失败步骤调用 `register_document_budgets`，成功后发布 scene。此前任何失败均不留下部分策略或 scene。

session 与 `install_scene_authoring_commands` 使用共享不可变策略 `scene.geometry_bytes`：所有 mesh（包括隐藏对象）的 position/index payload 合计最多 64 MiB（`64 * 1024 * 1024` bytes）。借用式计量只读长度，不读几何元素、不 clone；其他 schema 贡献零，但完整投影仍拒绝不支持的 schema。应在首次编辑前安装 profile/预算；已有 undo/redo 且未预装该策略的 controller 不能事后 attach，已预装同一共享策略则可带历史 attach。

通用 `ControllerDocumentBudget::new(id, limit, contribution)` 的计量回调接收 `&SelectionItem` 和 `&dyn Reflect`，返回 `Result<u64, String>`；策略是宿主配置，不是文档可执行内容。`register_authoring_profile(commands, budgets)` 原子安装命令及策略：命令容量、策略冲突、候选计量或预算失败均不发布其中任何部分。`document_budget_usage(id)` 只返回缓存的 used/limit。同一共享回调 `Arc` 和上限的策略 clone 可幂等重装；新增策略仅允许在 undo/redo 均为空时安装，并使安装前的 prepared（包括空操作）失效。完整 API 与错误契约见 [Editor 文档聚合预算](prism_editor_document_budget_zh.md)。

Reload 的 `PreparedDocumentLoad::install/install_ui` 在发布前继承宿主命令和策略，并对加载候选的新根重新计量，绝不沿用旧文档 used 缓存。失败保留原 controller/UI、历史及 durable base；成功清空文档历史但保留宿主配置。直接替换 controller 的宿主也须显式转交配置。

## 事务与差分同步

- `scene.geometry_bytes` 在 controller 的 bootstrap、属性候选及结构变更发布前检查全文档总量，而不是提交成功后才由 scene 投影报错。属性批次仅计量最终 changed roots；先减全部旧贡献再加全部新贡献，任一策略超限、checked arithmetic 溢出或贡献错误均保留模型、selection、history、dirty 和 revisions。标量 Undo/Redo 使用稀疏缓存逆补丁，从当前总量推导，保留其后 bootstrap 的贡献；结构恢复可重新计量涉及根。
- controller 用实例身份、shape revision 与 model revision 区分外来文档、对象/层级变化和属性修改；选择等 UI-only 修改不触发几何重建。
- 有界 changed-fields 日志按对象合并顶层字段，Undo/Redo 也记录同样路径。日志缺口保守回退完整重建；idle 路径只核对身份/revision，O(1)。
- 完整构建先检查对象数、entity 身份和借用长度的累计几何预算，再执行 `FromReflect`、描述符克隆和 ECS 装配。上限为 4096 objects / 64 MiB vertex-index payload；这不是总进程内存上限。
- matrix/visible-only 更新只借用读取相应属性，保留 ECS entity 和 CPU geometry allocations。几何更新仅解码被修改对象，先检查最终累计预算；同批 grow/shrink 不按中间总量误拒绝。
- 全部 CPU identity、schema、matrix、geometry 与预算检查在发布前完成。候选 prepare 不改变 live ECS；discard 丢候选，commit 同时切换 ECS 与场景。prepare 期间禁止 synchronize 和摄像机变更，避免候选被覆盖。
- 活跃 controller 编辑或 Gizmo preview 期间暂停已提交投影同步。Gizmo 仍使用 drag-start 快照和 controller 稀疏命令，释放后归入同一文档历史。

controller 的 `ControllerObjectValidator` TypeData 在注册/解码时验证完整根，在隔离编辑候选完成后验证最终状态；不会因耦合字段的临时无效中间状态拒绝合法批次。readonly、metadata、标量转换和资源限制仍生效。准入失败不发布模型、selection、history 或 revision。

对象 validator 负责单根 schema/finite/indices/matrix 等合法性，文档策略负责聚合贡献；两者互补。session 的 `document_geometry_bytes` 及增量累计预检仍保留为 `FromReflect`、descriptor/ECS allocation 之前的防御性边界，不能替代 controller 的预发布准入。64 MiB 仅指当前几何 payload，不是总进程、历史根或 GPU 内存上限。

## GPU 成本与边界

`DesktopScene` 保留 geometry renderer，区分 geometry 和 visibility dirty 集合。retained upload 只遍历两个集合的并集，以 `SceneIdPass::apply_patch` 提交这些对象的 metadata upsert，再将 commit 差分应用到 ID draw entries，不再对全场景 `submit` 或清空全场景 identity 绑定。visibility-only 不上传几何或 transform；geometry-only 只上传被修改 mesh，保留其 transform binding，且不重建 renderer。token 未变的对象保留原 identity uniform buffer/bind group。fresh attach/recovery 仍全量上传 geometry/transform、`submit` 完整 metadata，并以 `synchronize` 填充新的 renderer，不能因 pass entries 未变而遗漏绑定。

多次 session synchronize 后才 encode 时，dirty 并集按对象去重，从当前 descriptor 读取最终 visibility（包含反复隐藏/恢复及几何替换），不提交中间快照。矩阵更新通过已有、相等值跳过的 transform 路径处理；携带 geometry+matrix 的 replacement 同样走此路径，后续 encode 不重复上传矩阵。无 GPU 或等待 fresh upload 时，修改先保留在 CPU 状态，初始化/恢复必须完整同步最终状态，不能只处理 dirty 子集。设备恢复、完整文档替换和结构命令触发的 shape revision 变化仍可重建 GPU 资源；当前 ID 差分复用保证针对同一 shape 的属性/几何更新，不表示对象增删也只分配增量资源。

`DesktopSceneStatus` 记录实际 geometry uploads/bytes、transform uploads 和 renderer creations，并新增跨 renderer 替换累计的 `identity_allocations`。底层 `SceneIdGeometryRenderer::identity_allocations()` 在实际创建每对 identity uniform buffer/bind group 时递增，一对记为一次；相同 token 的 metadata 更新、单 mesh 几何更新不递增，完整重建会递增。它不是存活资源数，也不包含 mesh、transform、camera 或 preview 资源。`EditorSceneSyncWork` 记录 ECS/描述符同步工作。计数不等价于帧时或 GPU profiling。

`DesktopSceneStatus::scene_metadata_upserts` 累计成功提交的 metadata item 数：fresh snapshot 计入全部对象，retained patch 仅计入 dirty 并集，同一对象同时 geometry/visibility dirty 只计一次。矩阵-only 或 idle encode 不增加该计数；它不是实际改变的可见对象数，也不是所有内部遍历次数或 GPU 分配数。

结构删除在 controller 确定 alias/子树闭包后单遍裁剪 selection，保留存活顺序、仍存活的显式 primary，并清理失效 anchor，避免逐个删除对象重复扫描 selection。这不改变 shape-change 的完整投影/GPU 同步边界；详见 `prism_editor_object_authoring_zh.md`。

成本限制：controller 当前每个 touched object 的候选事务仍克隆一次完整根；不能把投影的借用路径称为整条编辑链零拷贝。完整构建会同时存在作者数据、typed ECS mesh 和 scene descriptor，GPU 另有 buffers；完整 GPU upload 会扫描 pass metadata。没有大型项目 P50/P95/P99 或跨后端性能结论。

## 运行入口

desktop 默认持续运行，只有显式 `--frames` / `--time-limit-seconds` 才有限时/帧约束；headless 默认一帧。显式帧数未达到时返回错误，包括没有呈现任何帧的情况。frames smoke 默认最多等待 30 秒，用户指定时间上限为 1..86400 秒。尺寸、重复/冲突选项和非 desktop 项目参数均严格检查。

## 验证

以下区分最新预算切片统一验证与较早已完成记录，不将不同版本的测试数量相加。

### 最新预算切片：统一回归已通过

session `59915` 的强制 GPU app + viewport、all features/targets 统一回归已 exit 0：353 passed、0 failed/ignored。覆盖全部 16 项聚合预算测试，包括独立多预算原子拒绝、scalar 编辑后 bootstrap 的 Undo/Redo 基线、空 prepared 在新 policy 安装后的失效，以及 Reload 重计量和 `new/with_camera` 可变 controller 构造准入。session `75350` 的同范围 strict Clippy（`--no-deps -- -D warnings`）exit 0；依赖树无 Bevy。精确命令和日志见 [Editor 文档聚合预算](prism_editor_document_budget_zh.md)。这些结果不代替原生输入 E2E 或完整 Editor 验收。

### 上一切片统一验证（早于本次预算接线）

- session `36236`：`PRISM_REQUIRE_GPU=1 cargo test --offline -j1 -p prism_editor_app -p prism_editor_viewport --all-features --all-targets`，exit 0；各结果为 `146/17/2/15/14/6/23/8/4/11/9/18/14/0/42`，共 329 项通过，0 failed/ignored。日志：`/private/tmp/prism-editor-authoring-final-gpu.log`。
- session `40272`：同两包 `--all-features --all-targets` 的 `cargo clippy --no-deps -- -D warnings`，exit 0。日志：`/private/tmp/prism-editor-authoring-strict.log`。
- `cargo tree` 检查同两包所有 features 的 normal/build/dev 依赖，无 `bevy` 匹配。

该轮统一验证覆盖 retained dirty-union、transform binding 保留、metadata upsert 计数、异构 geometry survivor 重排、初始化/恢复及 selection pruning/range anchor 回归。native 仍仅验证呈现 3 帧，没有 input E2E；不能以 offscreen GPU 测试替代原生输入验收。

### 较早历史验证

2026-10-22 的较早工作树快照曾运行（非当前 latest patch）：

```sh
CARGO_TARGET_DIR=/private/tmp/prism-editor-world-wake-target \
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 \
PRISM_REQUIRE_GPU=1 cargo test --offline -j 1 -p prism_editor_app \
  --features desktop --lib --bin prism_editor_app \
  --test controller --test controller_validation --test scene_session
```

exit 0：142 library + 17 CLI + 15 controller + 6 controller validation + 13 scene session，共 193 项通过，0 failed/ignored。日志：`/private/tmp/prism-editor-scene-session-regression.log`。强制 GPU 测试没有 adapter 时必须失败，覆盖真实 offscreen 可见性/拾取、Undo/Redo、renderer 复用和 dirty-only 几何上传。另有实际临时目录 SceneProject 保存/独立打开、反射 codec 到 ECS/hierarchy、候选围栏、非法几何与合法 sibling 的原子拒绝测试。

全文档预算哨兵直接调用生产 `document_geometry_bytes`，无需分配大型网格：测试空文档、精确累计、累计超限、迭代前对象数拒绝，以及读取 root 前非法身份拒绝。

### 后续历史验收（早于 latest GPU patch）

主线此前报告 app + viewport 强制 GPU 回归共 325 项通过，以及结构定向回归 12 项通过。这些均早于最新 retained dirty-union / transform binding patch，保留作版本边界；不与当前 329 项结果相加，也不将两个历史计数简单相加。native smoke 已实际呈现 3 帧，但没有原生输入 E2E，不构成桌面输入闭环验收。

### 已记录的 GPU/结构回归范围

下列回归及新增计数、异构 geometry 和恢复断言已纳入上述 session `36236` 的统一测试通过记录。

- 真实 offscreen GPU 单对象既有回归增加 identity 分配断言：初始为 1，visibility Undo/Redo、matrix、geometry 与 idle encode 均保持 1。
- 三对象真实 GPU 回归 `gpu_multiple_syncs_before_encode_retain_unrelated_identity_resources_and_final_visibility`：首次分配为 3；同一 mesh 多次改动、其他 mesh 矩阵修改及多个对象反复 visibility 同步后才 encode，只上传一个脏 mesh 且 identity 分配仍为 3。最新增量增加 geometry+matrix replacement 的矩阵只上传一次、纯几何不新增 transform，以及 dirty-union metadata 计数断言：初始 3，第一批并集两个对象后为 5，第二批后为 8，单网格修改后为 9，完整文档重建后为 12。实际点选/框选检查最终可见性和未修改对象仍可拾取。完整重建作为正向对照，累计 identity 分配增为 6（包含隐藏对象）。该测试沿用 `PRISM_REQUIRE_GPU=1` 的强制 GPU 约定。
- 新增 `aggregate_budget_rejection_precedes_geometry_access_and_preserves_all_live_objects`：用可切换反射长度的小型哨兵，在合法 controller 批次提交后模拟单 mesh 合法、累计超 64 MiB。生产增量预检必须在读取元素/clone 前拒绝，且不发布同批 sibling matrix/visibility；重复失败保留 ECS identity、几何地址、描述符、generation 和工作计数。恢复长度后整个批次可重试并 Undo。哨兵模拟防御性边界，不引入生产模型绕过 controller 的写入口，也无需大型分配。
- 新增 `gpu_authoring_create_delete_undo_redo_fences_point_and_box_picks_across_empty_shapes`：通过正式安装的 Add triangle / Delete selection 命令走空场景、创建、Undo/Redo、删除、Undo/Redo。每次 shape 变化前交替提交真实异步点选或框选，断言发布取消旧请求、不覆盖命令后的 selection，且新 encode 前两类拾取均被围栏；encode 后验证实际点选/框选、ECS/描述符及真实 GPU 上传/identity 分配计数。空 shape 不分配 identity，idle encode 不新增上传。
- `gpu_structural_batch_create_and_middle_delete_undo_redo_remap_survivor_draw_keys`：原子创建三个分离 mesh，通过正式删除命令移除中间对象，覆盖删除与批量创建的 Undo/Redo、全空与恢复。最新增量改用不同 silhouette 和 payload 长度的异构 geometry（三角形/四边形），通过正反像素与全视口框选检查右侧 survivor 从 draw/instance key 2 重排为 1 后仍匹配自身几何与身份，并按实际 geometry bytes 与 metadata upserts 断言成本；资源计数明确允许结构全量重建。
- 新增 `gpu_deferred_dirty_meshes_attach_and_resume_with_final_transforms_and_visibility`：在无 GPU 阶段先产生 geometry/matrix/visibility dirty，初始化时检查完整上传；suspend 期间多次同步再恢复，验证最终可见性、矩阵、点选/框选以及隐藏对象绑定也已装配。恢复后的混合 replacement 断言矩阵在同步时仅上传一次，encode 不重复上传，并检查 metadata/identity 分配和 idle 不增加上传。

指定 Rust 文件已完成定向 rustfmt（`skip_children=true`）及 diff whitespace 检查；最新统一回归覆盖上述 GPU patch、selection pruning、结构删除/Undo/Redo 后 range anchor 恢复和本次预算接线。格式检查不替代运行验证，测试通过也不扩大本切片的功能边界。

上述不是原生输入/窗口恢复验收，也不是完整 World 或完整 Editor 交付。对象创建/删除及最新 GPU 性能增量已有上述统一验证记录；native 仍仅有 3 帧呈现、无 input E2E，真实桌面端到端输入、World runtime/OFPA 正式接线、具体域工具、项目恢复/迁移、多窗口及规模性能验收仍是独立边界。
