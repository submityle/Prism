# Prism Asset Build：顶级次世代 AAA 资产构建系统设计

> **产品名：Prism Asset Build**。名称中不使用 `Framework`；但产品本质是可扩展的资产构建基础设施，而不是一条写死的线性流水线。
>
> 本文设计从外部源资产进入项目开始，覆盖导入、规范化、派生数据烘焙、平台 Cook、质量验证、内容寻址缓存、打包、签名、Patch/DLC 和运行时交付。`Prism Asset Build` 生成制品；`prism_asset` 运行时负责解析、挂载、异步加载、流送、热重载和回收。

- 文档状态：开发期目标规格（Draft，允许破坏性调整；不是已交付能力声明）
- 变更记录：本轮进一步冻结“BuildGraph 可并行规划 ≠ BuildExecutor 已并行发布”的边界，并将 raw v2 物理布局、跨平台锁、Source/Recipe 主链和发行安全能力列为独立验收项。
- 当前基线：本地确定性 build/CAS/DDC/revision 垂直切片、PNG artifact importer、虚拟几何 canonical processor 和 raw package/runtime bridge 已有代码与测试；BuildGraph 另提供按 DAG wave 并行执行的真实线程安全路径，但 BuildExecutor 的 DDC/CAS/事务发布仍是串行闭环；CLI 已补齐受限的 `package`/`validate` 发行命令，但它们只覆盖当前单 PNG 垂直切片，不是通用 Cook；完整 importer/recipe、远程服务、稳定发行协议和发布安全链仍未完成。
- 兼容策略：开发期不承诺旧 API/格式兼容；发行前冻结 Artifact Key、Schema、Container、Manifest 与签名策略，并建立兼容矩阵及迁移/拒绝规则。
- 目标：顶级次世代 AAA 内容规模，同时兼顾中小项目的渐进采用
- 核心取向：效果不被工具链限制、离线换运行时性能、增量优先、确定性优先、可诊断优先
- 参考形态：Unreal Asset Registry / Cook / DDC / IoStore、Unity Importer / Addressables、Bazel/Nix 内容寻址、DirectStorage/GDeflate、现代虚拟纹理与虚拟几何流水线
- 关联文档：`docs/prism_asset_design_zh.md`、`docs/prism_material_pipeline_design_zh.md`、`docs/prism_rendering_architecture_zh.md`、`docs/prism_platform_design_zh.md`、`docs/prism_derived_data_contract_zh.md`（派生数据契约登记表）

## 实施状态（2026-10-13）

当前仓库已交付一个可验证的 M0 垂直切片，但距离本文定义的 AAA 构建系统仍有明确缺口：

- **已完成（核心切片）**：`pkg/prism_asset_build_core` 已实现显式输入的确定性 `ArtifactKey`、内容哈希、BLAKE3 本地 CAS、确定性串行 `BuildGraph`/`Processor` 契约（节点依赖、拓扑、环/缺失依赖诊断、失败传播）；`BuildGraph::execute_parallel` 还提供按稳定 JobId 分波、线程安全 Processor、默认受主机并行度限制且可由 `execute_parallel_with_limit` 显式限流、确定性失败选择的真实独立 DAG wave 执行路径，以及本地 revision 事务。`BuildExecutor` 将这些组件接成单机串行闭环：按显式 Processor/设置/Profile/工具链/环境/输出契约生成 key，校验 DDC descriptor 与 CAS 字节，缓存损坏时失效并重建，全部节点成功后才发布新 Revision。发布前现验证 Artifact dependency closure（缺失、自依赖、重复边、环）且拒绝重复 AssetId；Unix 上使用安全封装的 advisory file lock 串行化独立 store handle，并在替换 `CURRENT` 前再次核对期望 Revision。虚拟几何 v1 资产已有 `CanonicalVirtualGeometryProcessor`，可验证、确定性重编码并生成 CAS/Manifest descriptor。
- **部分完成**：本地 DDC 已提供持久化 `ArtifactKey -> ArtifactDescriptor` 索引、确定性编码/校验和原子文件替换，与 CAS 分离；Unix DDC 索引已用 advisory `flock` 保护跨进程 read-modify-write，避免独立句柄互相覆盖更新。仍无 DDC 分层/驱逐策略、真实跨进程测试、非 Unix 跨进程协调、BuildExecutor 级并行 DDC/CAS 调度与完整失效分析；底层 BuildGraph 已提供独立 DAG wave 并行执行，但尚未接入发布闭环。发布事务已接入 Unix 跨进程锁、带持久化 `TransactionId` 的 append-only journal（兼容读取旧 payload frame）、Prepare/Commit/Abort marker、可选 RuntimePackage 的 descriptor/hash 绑定、启动恢复、孤儿 manifest/runtime 输出清理、部分 journal 尾帧截断恢复、`UnknownCommit` 状态识别、事务状态查询、按 transaction id 的显式 reconcile 和提交 Revision 重放；RevisionStore 已提供基于已提交 Manifest roots、带宽限期并在 pending transaction 时延期的本地 CAS 可达性 sweep，但完整磁盘故障注入、非 Unix 跨进程协调、远程幂等重试协议和事务级持久化 registry 仍未完成。
- **部分完成**：`pkg/prism_asset_container` 已实现 v1 raw 容器 writer/reader、严格 ToC/范围/哈希校验、文件 `RangeSource`（positional I/O、短读/截断门禁、并发 range read）、overlay、确定性 Runtime Manifest v2（独立 Container Table 绑定长度、BLAKE3 内容 hash 和稳定 generation token）、`ManifestAssetReader`，以及独立确定性 `ChunkPlanner`（硬依赖闭包、环/缺失依赖、对齐、容量、优先级和跨 Chunk 依赖先后顺序）。`prism_asset_build_core::package_runtime` 已将 Build Manifest/CAS 字节桥接为 Runtime Manifest 与 raw 容器，并将 `ChunkPlanner` 接入实际容器布局（支持 v2 计划布局的写入与严格回读）；本轮已修复并测试 chunk 起始对齐、manifest/chunk priority 传递、同名容器 remount 替换旧 mount、同名同 GUID 换包的身份拒绝，以及 publication 对 `AssetId -> StableGuid` 完整覆盖的严格拒绝。当前 v2 仍是内部 raw 物理布局切片，不是已冻结的完整 wire contract：ChunkPlan 元数据、依赖/流送表、平台/build flags、压缩块表、加密和签名元数据尚未持久化或实现。容器层已有文件按范围读取，但还没有 CDN/DirectStorage 按 Chunk 流送调度。压缩/加密/签名、Provenance、Patch/DLC 和发布回滚仍未实现。
- **部分完成的构建主链**：SourceSnapshot/ImportRecipe 的显式身份、`ImportRecipe::plan` 对处理器身份的绑定、source-root → recipe processor → BuildExecutor 的真实串行垂直链、基于成熟 `png` crate 的 PNG→规范 RGBA8 artifact importer，以及调用同一核心的 `prism_asset_build_cli`（`plan/import/build/inspect`）已落地并有测试；CLI 还提供受限的 `package`/`validate`：从当前 Revision 运行 `ArtifactQualityGate`，调用 `package_runtime` 生成 canonical Runtime Manifest/raw container/publication descriptor，并对已有发行目录做逐字节复核。该入口需要显式 `--asset-id`/`--logical-path`，当前只支持“source AssetId = image AssetId - 1”的单 PNG 垂直切片，不代表通用 Cook。完整 Source/Importer 插件体系、IR、字段级失效、BuildExecutor 的并行 DDC/CAS 调度、DDC L0–L3、远程 CAS/worker、双跑确定性门禁、完整崩溃恢复与质量报告仍未完成。当前 executor 仍不是完整生产 Cook 系统。
- **未完成的发行链**：构建侧 Runtime Manifest 已有 v2 容器内容身份绑定，但尚未完整表达硬/可选依赖、初始驻留 Chunk、流送层级、codec/schema 和安装能力条件；压缩/加密/签名、Provenance、patch/DLC、平台 profile/能力矩阵及发布回滚仍未完成。当前 ChunkPlanner 到 raw 容器写入的垂直切片已接通，但还不是完整 Cook/发行链。
- **未完成的 AAA 规模能力**：VT/VG/HLOD/World Partition/长音频分块、遥测回灌、Build Farm、百万资产 Registry、插件隔离/准入和多项目 Artifact 治理。

本文的 M0/M1…路线图是验收目标，不代表这些能力已经存在；只有有代码、测试和运行时对接的切片才可标为完成。

### 可执行验收矩阵（2026-10-13）

矩阵中的“已实现”仅表示表列出的本地切片有实现路径及回归测试，不代表 AAA 生产成熟度、跨平台保证或性能达标。`部分实现` 表示存在可验证子集；`未实现` 表示当前没有可作为产品能力验收的端到端实现。代码/测试路径是复核入口，不把本节设计要求当成证据。

| 能力边界 | 状态 | 当前实现与可复核证据 | 尚未满足的验收条件 |
| --- | --- | --- | --- |
| Artifact 身份、确定性 DAG 规划与处理器执行 | 已实现（本地核心切片） | `prism_asset_build_core/src/hash.rs`、`graph.rs`；测试 `recipe_key_is_deterministic_and_all_declared_inputs_matter`、`topological_order_is_stable_across_insertion_orders`、`reports_missing_dependencies`、`reports_cycles_with_stable_job_ids`、`executes_real_transforms_and_preserves_declared_input_order` | 完整导入插件协议、字段级失效分析、超大图/百万资产规模证据未完成 |
| DAG wave 并行 | 已实现（BuildGraph 层） | `graph.rs`；`parallel_execution_matches_serial_outputs`、`parallel_execution_reports_smallest_failed_job_in_wave`、`parallel_execution_rejects_zero_concurrency`、`parallel_execution_with_limit_matches_serial_outputs` | 该 API 只执行 DAG processor，不负责 DDC/CAS/Revision 发布 |
| BuildExecutor 并行执行 | 部分实现（代码已有，专属验收不足） | `executor.rs` 提供 `execute_parallel` / `execute_parallel_with_limit`，对 ready wave 并行准备 DDC/CAS 处理，再按稳定顺序更新 DDC 与发布；当前 core 测试尚无针对这些 API 的专项并发、缓存损坏/命中及失败排序回归 | 需补串并行制品/Manifest 等价、限流、并行失败确定性、缓存损坏恢复和发布原子性测试；不代表跨进程或远程调度 |
| 本地 CAS、DDC 与 BuildExecutor 缓存校验 | 部分实现 | `cas.rs`、`ddc.rs`、`executor.rs`；`detects_corruption_on_read`、`persists_and_restores_across_multiple_opens`、`ddc_hit_skips_processor_and_publishes_next_revision`、`corrupt_cas_blob_is_rebuilt_instead_of_returned_as_a_hit` | 分层/驱逐、远端 CAS、真实跨进程与跨平台协调测试未完成 |
| SourceSnapshot / Recipe / PNG 导入垂直链 | 部分实现 | `source.rs`、`recipe.rs`、`import/image.rs`、`executor.rs`；`plan_binds_recipe_identity_and_declares_source_dependency`、`plan_rejects_a_recipe_for_a_different_processor`、`source_snapshot_and_recipe_drive_a_real_build_executor_chain`、`png_importer_publishes_a_canonical_image_artifact` | 当前 PNG canonical RGBA8 切片不代表通用 importer、纹理 mip/压缩/平台 Cook 或 runtime texture loader 已完成 |
| 本地 Revision 事务与恢复 | 部分实现 | `transaction.rs`、`journal.rs`；`publishes_and_failed_transaction_keeps_previous_revision`、`reopening_after_pointer_publish_recovers_missing_commit_marker`、`reopening_truncates_a_partial_journal_tail`、`reopening_fails_closed_on_a_corrupt_journal_frame`、`separate_store_handles_serialize_concurrent_revision_publication` | 完整磁盘故障注入、非 Unix 等价锁语义、远程幂等协议和完整事务级恢复仍未完成 |
| Runtime Manifest / raw 容器打包桥接 | 部分实现 | `package.rs`、`prism_asset_container`；`packages_manifest_cas_and_runtime_manifest_end_to_end`、`packaging_is_deterministic_across_mapping_order`、`package_runtime_consumes_chunk_capacity_alignment_and_manifest_routes` | 现有 raw v1/v2 切片不是冻结发行协议；压缩、加密、签名、Provenance、Patch/DLC 与发布回滚未实现 |
| CLI | 部分实现（本地 PNG 工作流） | `prism_asset_build_cli/src/pipeline.rs`、`tests/package_validate.rs`；`plan_is_stable_for_same_source`、`build_publishes_then_reuses_verified_ddc_artifacts`、`invalid_png_does_not_publish_a_revision`、`cli_build_package_validate_round_trip`、`cli_validate_rejects_tampered_runtime_container`；`package` 调用 `ArtifactQualityGate` + `package_runtime`，`validate` 重建并逐字节校验 Runtime Manifest、PublicationDescriptor 和容器 | 仅支持当前单 PNG source/image 两资产映射；不代表通用 Build/Cook CLI；JSON 事件协议、目标平台 Cook、压缩/加密/签名和完整发行闭包仍未完成 |
| GPU 直通/解压、远程 Build Farm、完整 Cook/发行链 | 未实现 | 当前资产构建代码中无可验收的端到端能力；目标规格见本文 §12–§18 | 需分别提供实现、失败路径、运行时接线及可重复验收证据后才能升级状态 |

失败原子性采用精确定义：Processor 失败后不得发布新 Revision；`processor_failure_does_not_publish_revision` 与集成测试 `tests/publication_atomicity.rs::failed_later_job_does_not_publish_a_new_revision` 覆盖串行执行时失败不推进发布指针。缓存写入不是事务回滚，也不得描述成所有磁盘字节均回滚；BuildExecutor 并行路径的等价及失败原子性仍需单独验收。

运行时资产内核的加载、热重载、流送和容器消费状态以 [`prism_asset_design_zh.md`](prism_asset_design_zh.md) 的“实施状态”为准；本矩阵只判定 Asset Build 与其运行时交付边界，不将两侧目标规格合并成已交付能力。

---

## 1. 定位与核心结论

> **读者门槛**：本文件把“验收规格”与“仓库当前垂直切片”并列描述。看到“应/必须/目标”时，必须同时查阅 §实施状态与对应测试；没有实现路径、失败测试和可重放产物的条目，不得进入“已完成”清单。

### 1.1 一句话定位

**Prism Asset Build 是 Prism 的内容编译器与制品生产系统**：它把不可控、面向创作的源数据，转换为确定、可缓存、可验证、可流送、面向目标平台的运行时数据。

```text
Source Assets
    ↓ Import / Normalize
Editable Assets + Intermediate Representation
    ↓ Build Graph / Bake
Immutable Artifacts
    ↓ Cook / Validate / Chunk / Package
Runtime Manifest + Containers + Patch Metadata
    ↓
prism_asset Runtime
```

### 1.2 为什么不能只做一条 Pipeline

AAA 资产不存在统一的线性处理过程。一个角色源文件可能同时产生渲染网格、虚拟几何页、骨架、动画、布料代理、碰撞体、材质、缩略图和编辑器预览；其中每项又受平台、质量档位和项目策略影响。因此系统的真实模型必须是 **Artifact DAG（制品有向无环图）**：

```text
                                     ┌─ Skeleton ─ Animation Segments
                                     ├─ Skin Binding ─ GPU Skin Data
FBX/USD ─ Scene IR ─ Mesh IR ────────┼─ Render Mesh ─ Meshlets ─ VG Pages
                 │                   ├─ Collision Mesh
                 │                   └─ Preview Mesh
                 └─ Material IR ─ Material Bake ─ Shader/Pipeline Requests
                                      └─ Texture IR ─ Mips ─ Platform Texture
```

导入、烘焙、Cook、验证、打包和发布只是 DAG 上的标准 Recipe。系统核心负责“怎样规划、执行、缓存、追踪和提交”；插件负责“怎样理解和转换某种内容”。

### 1.3 硬目标

1. **效果**：不因构建系统能力不足而限制材质、几何、动画、毛发、布料、地形、体积、音频和电影级内容。
2. **性能**：将能离线完成的工作全部离线化；运行时只做必要的解析、解压、上传和驻留决策。
3. **易用性**：拖入即用、默认正确、错误可修复；高级选项按需展开，而不是把底层格式细节暴露给普通美术。
4. **易扩展**：增加资产类型、处理器、平台和发布目标时，不修改核心调度与制品模型。
5. **易维护**：稳定协议、小型 crate、显式版本、结构化诊断、可重放构建和端到端追溯。
6. **规模化**：支持百万级资产记录、十万级构建节点、多人共享 DDC、构建农场和多平台并发 Cook。
7. **确定性**：相同输入、工具链、Profile 和依赖得到逐字节一致的制品。
8. **可靠性**：构建失败不污染有效制品；编辑器和运行时始终可以继续使用最后一次成功版本。

### 1.4 非目标

- 运行时不解析 FBX、USD、PSD、EXR 等创作格式。
- 不把目录结构等同于加载边界、打包分组或资产身份。
- 不允许处理器绕过制品接口直接改写 Registry 或发布目录。
- 不以“支持所有格式”为第一目标；优先保证核心格式链路达到生产质量。
- 初期不自研所有编解码器；第三方工具通过受控适配器接入，并锁定版本与许可证信息。
- 不让远程缓存成为正确性依赖；远程服务不可用时可退回本地构建。

---

## 2. 设计原则

### 2.1 Asset 与 Artifact 必须分离

- **Asset**：用户认知中的逻辑资产，有稳定 `AssetId`，允许移动、重命名和修改。
- **Artifact**：某次确定性构建产生的不可变制品，以内容寻址，可缓存、去重、打包和分发。
- **Blob**：Artifact 的二进制载荷，可被多个 Artifact 引用。
- **Manifest**：某次成功构建中，从 Asset/逻辑路径到 Artifact/Chunk 的不可变映射快照。

同一个 `AssetId` 可以在不同 Platform Profile 下映射到不同 Artifact；相同 Artifact 内容也可以被多个 Asset 复用。

### 2.2 路径不是身份

路径只用于组织和发现，不能作为引用稳定性的根。资产身份采用内嵌的稳定 `AssetId`；源数据、导入设置与处理器版本分别参与内容键计算。

### 2.3 默认增量，完整重建只是校验手段

所有节点必须声明输入、版本和配置。系统根据键判断复用、失效与重建，不依赖修改时间。完整重建用于发行验证、缓存审计和确定性校验，而不是日常工作流。

### 2.4 结果事务化

处理器输出先进入临时区；只有全部验证成功后才原子提交 Artifact、Registry Revision 和 Manifest。失败、取消或崩溃不能留下“半个新版本”。

### 2.5 编辑体验与发行正确性使用同一内核

编辑器、CLI、CI 和 Build Farm 共用图规划、处理器、缓存键和验证逻辑。编辑器可以使用更快的 Preview Recipe，但不能另写一条行为不一致的隐式导入链。

### 2.6 策略与机制分离

- 机制：DAG、缓存、调度、事务、诊断、内容存储。
- 策略：纹理质量、LOD 误差、平台格式、包分组、验证严重度。

策略由 Profile 和 Recipe 声明，不散落在处理器的 `if platform == ...` 分支中。

### 2.7 离线制品与进程内派生：一条内核，两种调度

“烘焙”不是一种资产类型，而是把可离线化的计算换成运行时成本。一份派生数据是否进入本系统，由 `DerivationClass`（§4.6）判定，而不是由“它是不是烘焙”判定：

- **离线制品（`OfflineShipped` / `OfflineEditorOnly`）**：输出只由 `源字节 + ImportSettings + Profile/平台 + Schema + 工具链指纹` 决定，可字节复现、可共享 DDC、可进发行包——必须作为本系统的 Recipe/Processor，其 DAG、缓存键、CAS、DDC、Revision、事务与 Provenance 全部由本系统持有。
- **进程内派生（`RuntimeDerived`）**：输出依赖运行时/会话状态（视点、破坏、编辑脏区），或必须在交互时延内重算——不进 DDC、不进 Manifest，由编辑器/运行时的进程内调度器执行，但必须复用同一 Bake Kernel（§6.6）。
- **驱动本地缓存（`DriverLocal`）**：键含驱动/GPU/设备信息，非内容寻址，永不进 DDC/Manifest（如 PSO/Pipeline Cache）。

边界两侧共享同一内核与同一确定性纪律（§18.1）：编辑器预览烘焙与发行烘焙必须是同一份算法代码的两种调度，不允许“编辑器专用近似分支”。这是 §2.5 在烘焙场景下的具体化，也是 `prism_asset` 只做消费、`prism_bake_*` 只做算法内核的职责依据。

每一种派生数据都必须在本系统与 `docs/prism_derived_data_contract_zh.md` 登记五元组：输入集合 / 键材料 / 失效触发 / `DerivationClass` / 确定性等级。

---

## 3. 总体架构

```text
┌──────────────────────────────────────────────────────────────────┐
│ Frontends                                                        │
│ Prism Editor · CLI · CI · Build Farm · Project Automation        │
├──────────────────────────────────────────────────────────────────┤
│ Workflow                                                         │
│ Build Request · Recipe · Platform Profile · Quality Policy       │
├──────────────────────────────────────────────────────────────────┤
│ Planner / Orchestrator                                           │
│ Asset Graph · Artifact DAG · Invalidation · Scheduler · Retry    │
├──────────────────────────────────────────────────────────────────┤
│ Processor SDK                                                    │
│ Importer · Transformer · Baker · Compiler · Validator · Packager │
├──────────────────────────────────────────────────────────────────┤
│ Artifact Services                                                │
│ CAS · DDC · Registry · Manifest · Provenance · Transaction       │
├──────────────────────────────────────────────────────────────────┤
│ Platform Services                                                │
│ prism_tasks · prism_platform · VFS · Process · Crypto · Network  │
└──────────────────────────────────────────────────────────────────┘
                                  │ published contract
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│ prism_asset Runtime                                              │
│ Mount · Resolve · Async Load · Stream · Residency · Evict        │
└──────────────────────────────────────────────────────────────────┘
```

### 3.1 建议 crate 划分

```text
pkg/prism_asset_build_core/       # ID、键、Artifact、协议、事件、错误
pkg/prism_asset_registry/         # Asset Registry、Revision、依赖/反向依赖
pkg/prism_asset_graph/            # DAG 规划、失效传播、循环诊断
pkg/prism_asset_processor/        # Processor SDK、注册表、Schema 与上下文
pkg/prism_asset_build/            # Orchestrator、调度、事务、重试、取消
pkg/prism_asset_cas/              # Blob CAS、去重、完整性校验、GC
pkg/prism_asset_ddc/              # 本地/共享/远程派生数据缓存
pkg/prism_asset_profile/          # Platform/Quality/Profile 合并与验证
pkg/prism_asset_validation/       # 规则引擎、预算、基线和报告
pkg/prism_asset_package/          # Chunk、容器、Manifest、Patch/DLC
pkg/prism_asset_build_cli/        # import/build/cook/package/inspect/diff
pkg/prism_asset_build_editor/     # 编辑器适配，不持有业务真相

pkg/prism_import_image/           # 具体插件；以下同类可独立演进
pkg/prism_import_gltf/
pkg/prism_import_usd/
pkg/prism_bake_texture/
pkg/prism_bake_mesh/
pkg/prism_bake_animation/
pkg/prism_bake_audio/
pkg/prism_bake_shader/
```

`prism_asset_build_core` 不依赖具体格式、GUI 或网络；构建服务依赖 `std`，而 `prism_asset` 的运行时身份/句柄核心仍可保持 `no_std + alloc` 边界。

### 3.2 与 `prism_asset` 的职责边界

| 能力 | Prism Asset Build | `prism_asset` Runtime |
|---|---:|---:|
| 源格式解析、导入设置 | 是 | 否 |
| 资产构建图与 DDC | 是 | 否 |
| Mesh/纹理/动画离线烘焙 | 是 | 否 |
| Cook、打包、签名、Patch | 是 | 否 |
| 运行时容器挂载 | 产出协议 | 执行 |
| `AssetId`/依赖契约 | 共同协议 | 共同协议 |
| 异步加载、流送、驻留 | 生成流送数据 | 执行 |
| 编辑期热重载 | 触发增量重建 | 原子接收新制品 |

构建侧和运行时侧共享**版本化协议 crate**，但不能直接共享包含编辑器依赖的实现 crate。

---

## 4. 领域模型与稳定身份

### 4.1 核心类型

```rust
pub struct AssetId(pub u128);             // 用户逻辑资产身份
pub struct SubAssetId(pub u64);           // 一个源包内的稳定子资产身份
pub struct ProcessorId(pub u128);         // 处理器身份，不以类型名隐式推导
pub struct ArtifactKey(pub [u8; 32]);      // 构建语义键
pub struct ContentHash(pub [u8; 32]);      // 实际字节内容哈希
pub struct BuildRevision(pub u64);         // Registry 的单调修订号
pub struct SchemaVersion(pub u32);
pub struct ProfileHash(pub [u8; 32]);
```

### 4.2 三种不同的哈希

不可混淆：

1. **Source Hash**：规范化源输入的内容哈希。
2. **Artifact Key**：预测某个节点语义输出的缓存键。
3. **Content Hash**：节点实际输出字节的哈希，用于完整性验证和 Blob 去重。

```text
ArtifactKey = H(
    protocol_version,
    processor_id,
    processor_version,
    ordered_input_artifact_keys,
    canonical_settings,
    effective_profile_hash,
    toolchain_fingerprint,
    declared_environment_fingerprint
)
```

不允许把机器绝对路径、时间戳、线程数、随机种子或本地用户名隐式加入键。确实影响输出的环境必须显式声明并规范化。

### 4.3 `AssetRecord`

```rust
pub struct AssetRecord {
    pub id: AssetId,
    pub kind: AssetKindId,
    pub display_name: String,
    pub logical_path: LogicalAssetPath,
    pub source: SourceDescriptor,
    pub import_settings: SettingsBlob,
    pub labels: Vec<LabelId>,
    pub hard_dependencies: Vec<AssetRef>,
    pub soft_dependencies: Vec<AssetRef>,
    pub editor_only: bool,
    pub revision: BuildRevision,
}
```

硬依赖进入就绪闭包；软依赖只进入预取、可选内容或运行时按需解析。构建规划必须区分二者，避免“引用了一个皮肤列表就把所有皮肤打进启动包”。

### 4.4 `ArtifactDescriptor`

```rust
pub struct ArtifactDescriptor {
    pub key: ArtifactKey,
    pub kind: ArtifactKindId,
    pub schema: SchemaVersion,
    pub content_hash: ContentHash,
    pub blob: BlobRef,
    pub byte_len: u64,
    pub alignment: u32,
    pub platform: Option<PlatformId>,
    pub dependencies: Vec<ArtifactDependency>,
    pub chunks: Vec<ChunkDescriptor>,
    pub provenance: Provenance,
}
```

Artifact 一经提交不可原地修改。任何修改产生新键/新内容并通过新 Manifest 指向它。

### 4.5 子资产稳定性

FBX/USD/glTF 内部的 Mesh、Skeleton、Animation 等子资产不能只按数组下标寻址。Importer 需要构造稳定 `SubAssetId`：

- 优先使用源格式的持久对象 ID。
- 其次使用规范化层级路径 + 类型 + 作者标识。
- 同名冲突必须报诊断，不允许依赖枚举顺序。
- 重导入时输出子资产匹配报告：保留、增加、删除、疑似重命名。
- 删除仍被引用的子资产时阻断提交，或要求显式重定向。

### 4.6 `DerivationClass`

每一份派生数据（烘焙产物、缓存、代理、预计算场）都必须在 Recipe/Processor 描述符与 `docs/prism_derived_data_contract_zh.md` 中声明其调度与持久化等级：

```rust
pub enum DerivationClass {
    /// 离线构建、进发行包、可共享 DDC；必须字节可复现。
    OfflineShipped,
    /// 离线构建、仅编辑期消费；可缓存，不随包发行。
    OfflineEditorOnly,
    /// 进程内重算（脏区/dirty-set 驱动）；可持久化为本地 cache，不进 DDC/Manifest。
    RuntimeDerived,
    /// 键含驱动/GPU/设备，非内容寻址；永不进 DDC/Manifest。
    DriverLocal,
}
```

判定顺序即 §2.7 的三条：纯函数性 → 是否值得发行/共享 → 时延容忍。任一不满足就不得标记为 `OfflineShipped`；误标会让非确定或会话相关的产物污染共享 DDC，是 §1.3 确定性与 §26 可靠性风险的直接来源。

---

## 5. Source、Editable Asset 与中间表示

### 5.1 四层内容形态

| 层次 | 用途 | 可变性 | 是否进入发行包 |
|---|---|---|---:|
| Source Asset | DCC/外部工具创作真相 | 外部可变 | 否 |
| Editable Asset | Prism 编辑语义、身份与设置 | 可事务修改 | 否 |
| Intermediate Artifact | 跨处理器、高保真 IR | 不可变 | 通常否 |
| Runtime Artifact | 平台与质量档位优化数据 | 不可变 | 是 |

### 5.2 `.prism` 编辑态资产包

一个 `.prism` 文件是一个自描述编辑资产，不依赖旁车 `.meta`：

```text
Header
  magic / schema / AssetId / kind / feature flags
Import Settings
  typed canonical settings blob
Source Record
  source URI / source format / source hash / optional embedded source
Sub-asset Directory
  stable SubAssetId / name / kind
Dependency Records
  hard / soft / editor-only dependencies
Editable Payload
  lossless or sufficiently high-fidelity authoring representation
Optional Preview
  thumbnail / bounds / waveform / summary
```

它不是运行时格式，也不是 DDC。为了版本控制友好，元数据支持确定性文本视图或专用语义 Diff；大型载荷可通过外部 CAS 引用，避免频繁改动大二进制。

### 5.3 IR 设计原则

- Importer 只负责将外部格式转换为规范 IR，不直接生成所有平台制品。
- IR 保留效果相关的高精度信息，量化和有损压缩由后续 Baker 决定。
- IR Schema 独立版本化，并提供显式迁移器。
- IR 不泄露第三方 SDK 内存布局或生命周期。
- 常见资产使用共享 IR；特殊资产允许注册专用 IR，但必须可序列化、可哈希、可验证。

---

## 6. Processor SDK

### 6.1 统一处理器，而非五套相互割裂接口

Importer、Baker、Compiler、Validator、Packager 在执行层统一为 `Processor`，通过类别与权限区分能力：

```rust
pub trait Processor: Send + Sync {
    fn descriptor(&self) -> &ProcessorDescriptor;

    fn analyze(
        &self,
        ctx: &AnalyzeContext,
        request: &ProcessRequest,
    ) -> Result<ProcessPlan, ProcessError>;

    fn execute(
        &self,
        ctx: &ExecuteContext,
        plan: &ProcessPlan,
        output: &mut ArtifactWriter,
    ) -> Result<ProcessReport, ProcessError>;
}
```

`analyze` 必须无副作用，用于：

- 动态发现显式依赖。
- 选择输出 Artifact 类型。
- 计算规范设置和缓存键材料。
- 声明 CPU、内存、I/O、GPU、外部进程需求。
- 声明是否允许远程执行、是否可缓存、是否可重试。

`execute` 只能读取计划声明的输入和能力；未声明访问被视为可重复性违规。

### 6.2 描述符与能力声明

```rust
pub struct ProcessorDescriptor {
    pub id: ProcessorId,
    pub name: &'static str,
    pub semantic_version: Version,
    pub category: ProcessorCategory,
    pub input_kinds: Vec<ArtifactKindId>,
    pub output_kinds: Vec<ArtifactKindId>,
    pub settings_schema: SchemaId,
    pub deterministic: bool,
    pub remote_capable: bool,
    pub supports_cancellation: bool,
    pub resource_class: ResourceClass,
}
```

任何会影响输出语义的处理器变更都必须提升版本。只修改日志或性能且逐字节输出不变时，可不使缓存失效，但需通过双跑证明。

### 6.3 类型安全的设置和输出

- 设置经 `prism_reflect` 注册，自动生成 Inspector、CLI Schema、默认值、范围和文档。
- 序列化前做 canonicalization：字段排序、浮点规范、路径规范、默认值展开。
- Processor 声明输入/输出 kind，规划期检查连接合法性。
- 禁止以字符串约定隐藏输出；所有 Artifact 都有注册 Kind 和 Schema。

### 6.4 原子 Artifact Writer

```text
begin_output
  → write metadata
  → stream payload/chunks
  → finalize hashes
  → run output validators
  → prepare commit
  → commit transaction
```

- 临时输出与 CAS 位于同文件系统时优先原子 rename。
- 大制品必须支持流式写入，禁止要求整块驻留内存。
- 写入时同步计算哈希，不进行额外全文件扫描。
- 取消和失败自动回滚临时文件。
- 只有完整验证通过的输出才能对其他构建可见。

### 6.5 插件兼容与隔离

插件分三级：

1. **In-process trusted**：Prism 官方、经过审核；最低 IPC 成本。
2. **Worker process**：第三方 SDK、可能崩溃或泄漏；通过版本化 IPC 隔离。
3. **Remote worker**：高成本烘焙或 Build Farm；只接触声明的输入 Blob。

第三方格式解析默认建议 Worker Process，避免恶意/损坏资产破坏编辑器主进程。插件必须声明许可证、工具链指纹、目标架构与可用平台。

### 6.6 Bake Kernel 与两种调度

算法与调度分离，是“源资产烘焙”与“场景派生烘焙”收敛为一条线的关键：

- **Bake Kernel**：`prism_bake_*` crate，纯算法核心——无 I/O、无调度、无全局状态；输入为 IR/结构化参数，输出为字节或结构化产物。必须存在 CPU 金标准，GPU 实现（若有）与之逐元素对拍（范式见 `pkg/prism_micromap_gpu/tests/gpu_parity.rs`）。
- **离线调度**：本系统的 `Processor` 包一层 Kernel，负责 `analyze` 声明输入与键材料、选择 `ArtifactKind`、经 `ArtifactWriter` 流式写入并提交事务。
- **进程内调度**：编辑器/运行时的 `BakeService`（跑在 `prism_tasks`）以 dirty-set 驱动同一个 Kernel，产物进本地 runtime cache，不写 CAS/DDC（见 `prism_asset_design_zh.md` §13.2）。

禁止在 Kernel 之外复制算法：任何“编辑器专用”或“运行时专用”的第二实现，都属于 §2.5 所禁止的行为不一致链，也是 `docs/prism_derived_data_contract_zh.md` 中 `OfflineShipped` 与 `RuntimeDerived` 必须指出同一内核的原因。

---

## 7. Recipe、Profile 与策略组合

### 7.1 Recipe 是“要什么”

Recipe 声明构建目标和所需输出，不硬编码执行顺序：

```text
EditorPreview   快、可降质、强调反馈速度
Development     接近运行时，保留诊断和调试元数据
Test            固定输入、严格验证、便于自动化
Shipping        最高质量门禁、裁剪编辑数据、签名
DedicatedServer 无渲染/音频内容，仅保留逻辑、碰撞、导航
Patch           相对于指定 Base Manifest 生成增量
DLC             独立内容闭包与挂载边界
```

项目可组合标准 Recipe，而不是复制整份配置。

### 7.2 Profile 是“按什么约束做”

```text
EffectiveProfile =
    Engine Baseline
  + Project Defaults
  + Platform Capabilities
  + Device/Quality Tier
  + Build Configuration
  + Asset Group Override
  + Asset Override
```

合并必须具有固定优先级和字段语义：标量覆盖、集合并/替换必须在 Schema 中声明；最终 Profile 规范化后计算 `ProfileHash`。

Profile 至少包含：

- GPU/CPU/内存/存储能力。
- 纹理格式与质量策略。
- Mesh/LOD/虚拟几何策略。
- Shader Feature 与 Variant 策略。
- 动画和音频压缩误差。
- 流送页尺寸、首驻留层级和预取策略。
- 包分组、压缩、加密、签名策略。
- 验证规则与严重级别。

### 7.3 能力选择，不散落平台名字

处理器根据能力选择实现，例如 `supports_bc7`、`supports_astc_hdr`、`supports_gpu_gdeflate`，而非直接判断 `Windows`/`iOS`。平台适配层负责产生能力集合。这使新设备档位和兼容层无需修改所有处理器。

---

## 8. Artifact DAG 与失效模型

### 8.1 两张图

系统维护两张相关但不同的图：

- **Asset Graph**：逻辑资产之间的硬/软/编辑器依赖。
- **Build Graph**：具体构建请求展开后的 Processor Node 与 Artifact 边。

Asset Graph 长期存储在 Registry；Build Graph 按 Recipe/Profile/目标 Manifest 规划，可以持久化为调试快照。

### 8.2 节点状态机

```text
Unknown → Planned → WaitingInputs → Ready → Running
                                ├→ CacheHit ───────────┐
                                ├→ Succeeded ──────────┤
                                ├→ Failed              ├→ Committed
                                └→ Cancelled            └→ Superseded
```

`Succeeded` 表示节点得到有效临时输出；只有所属事务提交后才成为 `Committed`。编辑器产生新修改时，旧构建可以进入 `Superseded`，其结果不得覆盖更新版本。

### 8.3 精确失效

失效由键变化驱动：

- 源内容变化：失效读取该源的后继节点。
- 设置字段变化：只失效声明依赖该字段的节点；初版可退化为整个设置 Blob。
- 处理器版本变化：失效该处理器输出及后继。
- Profile 变化：只失效读取变化能力/策略的节点。
- 依赖 Artifact 变化：沿 Build Graph 后继传播。

通过记录节点读取的配置字段，实现字段级依赖可显著减少大型项目重建范围；该能力应作为 M3 优化，而不是 M0 前置复杂度。

### 8.4 循环与动态依赖

- Build Graph 必须无环，规划期输出最短可解释环路径。
- 逻辑软引用可成环，但不能进入强构建依赖闭包。
- 动态依赖只能在 `analyze` 阶段发现；`execute` 不得新增未声明输入。
- 分析结果本身可缓存，其键由源目录/元数据输入决定。

### 8.5 构建去重

相同 `ArtifactKey` 的并发请求合并为一个 Shared Job：

- 多个调用者独立持有订阅和取消令牌。
- 单个调用者取消不终止仍被其他请求需要的作业。
- 最后一个订阅者取消后，调度器才尝试取消 Processor。
- 完成结果通过 CAS 广播复用，避免重复压缩、Shader 编译和网格处理。

---

## 9. 调度、并发与资源治理

### 9.1 调度目标

构建系统追求的是总吞吐、交互延迟和稳定资源占用之间的平衡，不是简单“线程全部跑满”。调度基于 `prism_tasks`，增加内容构建所需的资源令牌：

```text
CPU threads
Peak memory
Disk read/write bandwidth
Network bandwidth
External process slots
GPU compute/VRAM budget
License-limited tool seats
```

### 9.2 优先级

```text
Interactive   编辑器当前选中资产和视口预览
HotReload     当前关卡受影响资产
Foreground    用户显式 Build/Cook
Background    索引、缩略图和预热
CI            批量吞吐优先
Maintenance   CAS GC、校验和压缩整理
```

采用优先级老化避免后台任务永久饥饿；已运行的大内存节点一般不抢占，只在安全取消点让出。

### 9.3 内存感知调度

Processor 在分析期提供 `estimated_peak_memory`。调度器只有在预算允许时启动；实际超过声明时记录违规并动态限流。大型贴图、几何和视频必须使用流式/分块算法，避免多个节点同时形成内存峰值。

建议保证：

- 构建服务默认峰值不超过配置内存的 70%。
- 交互编辑时保留系统和编辑器安全余量。
- OOM 风险前主动暂停可暂停节点，而不是让 OS 杀死进程。
- 节点峰值估算和实际值进入历史模型，后续规划使用滑动分位数修正。

### 9.4 外部进程与远程 Worker

- Worker 使用租约；心跳超时后节点回到可重试状态。
- 输出在 Worker 侧完成哈希，服务端复验元数据和随机抽样/完整哈希。
- 只传输 CAS 中缺失的 Blob，避免重复上传源数据。
- Worker 能力包含 OS、架构、GPU、工具版本和插件集合。
- 非幂等或不可重试处理器禁止远程执行。
- 构建日志按 Node ID 汇聚，保持全局时间线和因果关系。

---

## 10. CAS 与 DDC

### 10.1 职责分离

- **CAS**：按 `ContentHash` 保存不可变 Blob，负责完整性、去重、租约与 GC。
- **DDC**：从 `ArtifactKey` 查询 `ArtifactDescriptor`，决定语义构建结果是否可复用。

```text
ArtifactKey ──DDC──▶ ArtifactDescriptor ──▶ ContentHash ──CAS──▶ bytes
```

仅有 CAS 不能判断处理器语义复用，仅有 DDC 不能保证 Blob 完整性。

### 10.2 分层缓存

```text
L0 Memory metadata cache
L1 Local NVMe CAS/DDC
L2 Team shared cache
L3 CI/Release immutable cache
```

查询优先近端，回填异步进行。远端超时应快速退回本地执行，不阻塞交互。Shipping 制品可提升到不可变发布命名空间，普通开发缓存不得覆盖它。

### 10.3 负缓存与失败缓存

- 可对确定性、输入相关的失败短期缓存，避免损坏资产被每帧热重载重复处理。
- 工具崩溃、网络失败、磁盘满等环境失败不能作为长期负缓存。
- 失败条目包含 Processor 版本、输入键和诊断摘要，任一变化立即失效。

### 10.4 CAS GC

采用 Manifest/Revision 可达性 + 租约。当前仓库已提供本地 `RevisionStore::sweep_unreachable_cas(grace)`，以已提交 journal revision/CURRENT 的 Manifest content hash 为 roots；存在未完成 Prepare 时会延期整次 sweep，并在删除后同步 shard 与 CAS 根目录。完整的活跃编辑器 pin、远程租约、后台预算调度仍属于后续工作：

1. 活跃编辑器 Revision、已发布 Manifest、正在执行事务和用户 Pin 构成根集合。
2. 标记可达 Artifact 与 Blob。
3. 未标记对象经过宽限期后清理。
4. 清理按 I/O 预算后台进行，支持容量上限和 LRU 快速回收。

严禁仅按文件最后访问时间删除，因为已发布但近期未读取的制品仍必须保留。

### 10.5 性能设计

- 小 Blob 聚合存储，减少百万小文件对文件系统的压力。
- 大 Blob 分块并支持范围获取、断点续传和并行校验。
- 元数据索引与载荷分离，常规查询不读取大 Blob。
- 本地索引使用崩溃一致的事务数据库或 append-only log + checkpoint。
- 对压缩后数据再压缩前先估算收益，低收益数据直存以节省 CPU。

---

## 11. 导入体验与源数据治理

### 11.1 标准导入流程

```text
Discover → Detect → Inspect → Propose Settings → Import → Validate
         → Preview → Commit Asset Revision → Schedule Derived Builds
```

导入器先执行轻量 `inspect`，在解析完整载荷前返回尺寸、色彩空间、节点、材质槽、动画片段等摘要，从而让用户确认设置并快速看到风险。

### 11.2 默认正确与渐进复杂度

- 初次导入基于资产语义、项目规则和目标平台给出安全默认值。
- UI 首屏只显示语义选项，如“法线贴图”“角色网格”“电影级”；高级页再显示误差阈值、编码器和页大小。
- 自动推断必须可解释：展示“为何选择 sRGB/BC7/ASTC/流送”。
- 文件名规则只能作为建议，不能静默覆盖明确设置。
- 批量编辑显示将触发的重建范围和预计成本。

### 11.3 重导入

- 源未变：零工作。
- 设置变：复用 Source/IR，只重建相关后继。
- 源变：保留 `AssetId`，执行子资产匹配。
- 导入失败：保留最后成功 Revision，错误版本仅作为诊断草稿。
- 用户手工编辑过的 Prism 属性通过明确 Merge Policy 保留；冲突进入可视化解决，不做隐式覆盖。

### 11.4 Source Provider

统一支持工作区文件、Perforce/Git LFS、对象存储和受控远程源。`SourceUri` 必须可重定位，不将本机绝对路径写入 Asset。Source Provider 提供：读取、内容哈希、变更通知、权限/锁定信息和可选版本标识。

---

## 12. 专项资产 Recipe

### 12.1 纹理

```text
Image Source → Decode → Color/Channel IR → Semantic Validation
             → Mip Generation → Channel Packing → Platform Encode
             → Optional VT Tiling → Runtime Texture Header + Pages
```

效果要求：

- 显式语义：Color、Normal、Mask、HDR、UI、Lightmap、Cube、Volume、Virtual Texture。
- 颜色管理记录源 Profile、工作空间与输出变换；禁止靠扩展名猜 Gamma。
- Mip 使用语义感知滤波：法线重归一化、粗糙度能量修正、Alpha Coverage 保持、HDR 防萤火。
- 压缩前生成可视化误差、PSNR/SSIM 或语义指标；法线使用角误差。
- 支持每 Mip/Page 独立校验、哈希和流送优先级。

性能要求：

- 平台格式离线确定：BCn、ASTC、ETC2 或平台专用格式。
- Mip Tail 合并，减少小块 I/O。
- VT 页按局部性重新排列，边界和 gutter 在 Bake 阶段生成。
- 编码器支持 CPU/GPU 后端，但相同后端版本必须满足确定性契约。

### 12.2 静态/蒙皮网格与虚拟几何

```text
Scene Source → Mesh IR → Repair/Validate → Vertex/Index Optimize
             → LOD/Simplify → Meshlet/Cluster → Hierarchy/BVH
             → Quantize → Page Layout → Runtime Mesh/VG Artifacts
```

效果要求：

- 保留硬边、UV seam、材质边界、蒙皮和 Morph 约束。
- LOD 以屏幕空间误差和语义权重驱动，不只按三角形比例。
- 支持手工 LOD、自动 LOD 和混合策略；报告几何/轮廓/蒙皮误差。
- 碰撞、导航、阴影和光追几何是独立输出，可使用不同精度。

性能要求：

- Vertex Cache、Vertex Fetch、Overdraw 优化按渲染路径选择。
- 顶点格式按属性和误差预算量化，避免所有网格固定宽格式。
- Meshlet 约束、BVH 分支度、页大小与 `prism_virtual_geometry_gpu` ABI 对齐。
- 产生低成本 Fallback Mesh，供不支持虚拟几何的平台和加载早期使用。

### 12.3 材质与 Shader

材质 Artifact 保存稳定的材质域、着色模型、参数布局、资源绑定和静态特性，不直接等同于某次临时 GPU Pipeline。

```text
Material Graph → Semantic IR → Optimization → Permutation Reduction
               → WESL Modules → Backend Compilation/Validation
               → Material Record + Shader Artifacts + Pipeline Hints
```

- 静态开关只用于影响资源布局或主要控制流的特性；普通差异走动态参数。
- 构建时统计 Variant 来源、组合数、重复率、编译耗时和运行时命中率。
- Shipping Recipe 根据关卡/内容 Manifest 裁剪不可达 Variant。
- Shader 编译键包含编译器、WESL/naga 版本、能力集、优化级别和 ABI。
- 为 PBR、NPR、Hair、Skin、Eye、Cloth、Water、Terrain、Volume 提供域级验证。

### 12.4 骨骼与动画

```text
Animation Source → Skeleton Mapping → Resample/Cleanup → Root Motion
                 → Segment → Error-bounded Compression → Runtime Clips
                 → Optional Motion Matching Features
```

- Root Motion、面部、手指和远景骨骼使用不同误差策略。
- 压缩器以最终蒙皮空间误差评估，不只比较局部曲线误差。
- 长动画按时间段分块流送，关键同步点不得跨不可用块。
- 骨架重映射和 Retarget 配置是显式 Artifact，便于缓存和审计。
- Motion Matching 特征生成独立成节点，避免修改数据库配置重压缩所有动画。

### 12.5 音频

- 短、延迟敏感音效生成低启动成本 Artifact。
- 长音乐/环境音生成分块流送 Artifact，含 seek table、loop 和预取头。
- 编码前执行响度、真峰值、静音、爆音和声道布局验证。
- 语言与地区资源作为发布维度，不迫使基础包携带全部本地化音频。
- 波形缩略图和编辑器分析数据为 Editor-only Artifact。

### 12.6 Scene、Prefab 与 World Partition

大型世界不得输出一个巨型场景 Blob。Recipe 至少拆分：

- Scene Header 和全局依赖。
- World Partition Cell。
- ECS Archetype/Component Chunk。
- Render Instance/HLOD。
- Navigation、Physics、Audio Zone、Lighting 和 Reflection 数据。
- Runtime Spawn Bundle 与 Editor-only 数据。

Cell 边界构建需要依赖提升规则：跨 Cell 强依赖提升到共享组或转为软引用；防止一个小对象把整张世界图拉入启动闭包。

### 12.7 毛发、布料、地形、体积与电影内容

这些高级资产通过专用 Processor/Artifact Kind 接入，但复用统一 DAG、DDC、Profile、验证和发布机制：

- 毛发：Guide/Strand/Card/LOD、插值、Cluster、流送块。
- 布料：高模到模拟网格映射、约束、碰撞代理、平台 Solver Profile。
- 地形：高度/材质/植被页、Clipmap、HLOD、物理与导航数据。
- 体积：稀疏体素砖、压缩、mip、光照缓存和流送目录。
- 电影：高码率序列、代理、镜头依赖、预缓存和可选独立内容包。

### 12.8 派生场景数据（GI / 导航 / SDF / HLOD / PCG / 地形）

场景与世界派生的数据不是“资产导入”的附属，而是同一 Artifact DAG 上的一等 Recipe。它们此前分散在 `prism_world_system_design_zh.md`、`prism_scene_design_zh.md`、`prism_terrain_world_design_zh.md`、`prism_gi_lumen_design_zh.md` 中各自定义为“烘焙期”，本节给出统一归属：

| 派生数据 | 键材料（除源内容外） | DerivationClass | 说明 |
|---|---|---|---|
| 烘焙 GI / 光照缓存 | scene 内容 + 种子 + 光照规则 + schema | `OfflineShipped` | GI 主路径为实时无烘焙；此项为可选离线缓存 |
| 导航网格 | scene 内容 + 代理半径/高度 + 区域代价 + schema | `OfflineShipped`（按流送分块） | 运行时破坏/编辑触发的局部重烘为 `RuntimeDerived` |
| 全局/局部 SDF | scene 内容 + 体素尺寸 + 带宽 + schema | `OfflineShipped` | 运行时动态物体的场更新为 `RuntimeDerived` |
| HLOD 代理 | scene 内容 + 成簇规则 + 种子 + 屏幕误差 | `OfflineShipped` | 编辑器预览代理为 `OfflineEditorOnly` |
| PCG 散布 proxy 列 | scene 内容 + 节点图 + 种子 + 规则版本 | `OfflineShipped` | 运行时局部重算为 `RuntimeDerived` |
| 地形侵蚀 / 远景底图 / RVT 烘底 | 高度/掩码 + 迭代参数 + 种子 + schema | `OfflineShipped` | 编辑器局部编辑与破坏触发的重烘为 `RuntimeDerived` |

硬约束：

- 每一项必须在 `docs/prism_derived_data_contract_zh.md` 填写完整五元组，并声明确定性等级（Byte Reproducible / Deterministic-not-byte / Best-effort）。
- 同一内核两种调度（§2.7、§6.6）：`OfflineShipped` 项与其对应的 `RuntimeDerived` 重烘必须调用同一 Bake Kernel，且在同一输入上产出一致；声明为 Deterministic-not-byte 者，差异必须显式列出并落在声明的容差内。
- 成簇/散布/侵蚀等所用“种子”必须是内容派生种子（§18.1），禁止时间戳、未排序并行序或机器状态进入键材料。
- `OfflineShipped` 的场景派生节点与资产节点共用同一失效传播（§8.3）、同一 DDC（§10）与同一事务提交（§17.1）；不得为“场景烘焙”另建一条绕过事务的写入路径。

---

## 13. 验证、预算与效果回归

### 13.1 结构化诊断

```rust
pub struct Diagnostic {
    pub code: DiagnosticCode,
    pub severity: Severity,
    pub message: LocalizedMessage,
    pub asset: Option<AssetId>,
    pub sub_asset: Option<SubAssetId>,
    pub source_span: Option<SourceSpan>,
    pub processor: Option<ProcessorId>,
    pub help: Option<FixSuggestion>,
    pub attachments: Vec<DiagnosticAttachment>,
}
```

稳定错误码用于 CI 规则、文档链接、忽略清单和趋势统计；用户可从诊断直接定位源节点、材质槽、纹理通道或动画区间。

### 13.2 三类门禁

1. **正确性**：损坏、缺依赖、Schema 不兼容、循环、NaN/Inf、越界等。
2. **效果质量**：LOD/压缩误差、颜色偏差、法线角误差、动画蒙皮误差、音频失真等。
3. **性能预算**：CPU/GPU 内存、I/O、解压、Draw/Dispatch、Variant、页数、启动闭包大小等。

严重度随 Recipe 改变：Preview 可允许部分 Warning；Shipping 对关键错误和预算回归直接失败。

### 13.3 预算采用层级聚合

预算不仅针对单资产，还针对：

```text
Asset → Prefab → Streaming Cell → Level → Game Mode → Platform Build
```

单个纹理不过限不代表整个关卡不过限。Build 报告必须支持按类型、标签、Owner、目录、关卡和包分组聚合，并展示最大的增量贡献者。

### 13.4 Golden 与感知回归

- 关键资产保存 Golden 渲染/几何/动画/音频指标。
- 纹理和渲染图执行容差感知 Diff，不只做二进制比较。
- Mesh 比较包围盒、轮廓、Hausdorff/屏幕误差、材质边界和蒙皮结果。
- Shader/材质用固定场景与视角做跨后端图像回归。
- 性能基线采用有方差带的统计门槛，避免机器噪声导致随机红灯。

### 13.5 自动修复的边界

安全、无歧义的修复可一键执行，如法线重归一化、非法名称替换建议；会改变视觉或拓扑的修复必须预览并由用户确认。所有修复作为设置变化或独立 Processor，保证可重放，不允许直接隐式改写源文件。

---

## 14. Cook、Chunk、Package 与 Publish

### 14.1 Cook

Cook 从明确入口集合计算可达闭包，生成目标 Profile 的 Runtime Artifact：

```text
Entry Assets + Recipe + Effective Profile + Build Version
    → reachable asset closure
    → platform artifacts
    → validation
    → runtime manifest
```

编辑器数据、未引用内容和不可用 Feature 在此裁剪。动态加载内容必须通过 Label、Addressable Group 或显式目录列入，不能依赖“碰巧在某目录”。

### 14.2 Chunk 规划

Chunk 不是任意定长切片；它同时服务 I/O、压缩、流送和 Patch：

- 首帧必需数据放入高优先级启动 Chunk。
- 经常共同加载的制品按访问相关性共置。
- 高频变化内容与稳定内容分离，降低 Patch 放大。
- 已压缩纹理/音频避免与高可压缩数据混装。
- VT/VG/音频按天然页/段作为可独立读取单元。
- Chunk 边界满足对齐、Direct I/O、GPU 解压和加密块要求。

规划器输入静态依赖、运行时遥测和项目约束；输出规划理由及预计读放大，不做不可解释的黑盒布局。

### 14.3 容器布局

> **当前实现边界**：以下布局是目标协议。当前实现仅产出 raw v1 容器及 v2 raw 物理布局切片；v2 尚未持久化完整 Chunk Table/Dependency/Streaming 元数据，也未实现压缩、加密、签名、Merkle root 或防回滚。因此当前产物只能作为内部垂直切片验证，不能作为已冻结的发行 wire contract。

```text
Container Header
  magic / format version / platform / build id / flags
Signed Manifest Reference
Asset Directory
  AssetId + SubAssetId → Artifact/Chunk ranges
Chunk Table
  id / offset / compressed size / raw size / codec / hash / alignment
Dependency/Streaming Tables
Payload Chunks
Signature / Merkle root
```

要求：

- 元数据可先读，载荷支持 range read。
- Chunk 独立校验、解压和重试。
- ToC 防篡改，偏移/长度严格界限校验。
- 支持 CPU 解压与 `prism_gdeflate_gpu` 等 GPU 路径。
- 格式支持向后兼容窗口；不兼容时明确拒绝挂载。

### 14.4 分组

标准分组可包含 Boot、Core、Startup、Level、Character、Cinematic、Audio、Localization、Optional、DLC。分组由运行时可达性、安装优先级、更新频率和产品策略驱动，目录和标签仅是输入之一。

### 14.5 Patch 与 DLC

```text
Patch = New Manifest
      + Added Chunks
      + Changed Chunks
      + Tombstones
      + Redirects
```

- 以指定 Base Manifest 为基准，不以开发目录当前状态猜测。
- 相同 ContentHash 的 Chunk 不重发。
- 输出下载大小、安装后增长、重压缩放大和回滚信息。
- DLC 有独立闭包、授权元数据和挂载点，不允许隐式依赖未声明 DLC。
- Base → DLC → Patch → Hotfix 按优先级叠挂；冲突在构建期报告。

### 14.6 发布与供应链

发布阶段执行：Manifest 签名、容器签名/哈希、SBOM/第三方许可证清单、Build Provenance、符号和调试内容分离、CDN 上传、可用性探测和分阶段发布。私钥不进入普通 Processor；签名通过隔离服务或平台安全存储完成。

---

## 15. 运行时交付契约

### 15.1 Runtime Manifest

> **当前实现边界**：当前 Runtime Manifest v2 已是稳定身份到容器名、优先级、容器内 GUID 和容器内容身份（长度、BLAKE3 hash、稳定 generation token）的路由索引，并在运行时拒绝同名换包。硬/可选依赖、初始驻留与流送层级、codec/schema、安装组/能力条件仍未形成完整运行时契约；当前 BLAKE3 摘要也不等同于认证签名。

Manifest 至少回答：

- `AssetId + SubAssetId` 映射到哪个 Artifact。
- 需要哪些硬依赖和可选依赖。
- 初始驻留 Chunk 与可流送层级。
- Artifact Schema、压缩、对齐和完整性信息。
- 所属容器、安装组、语言、DLC 和能力条件。

Manifest 不包含编辑器导入设置和源路径。

### 15.2 首帧与渐进质量

Build 为每类资产生成最小可用表示和增量增强层：

```text
Texture: mip tail → higher mips
Mesh: fallback/low LOD → high LOD/VG pages
Scene: critical cell → nearby/background cells
Audio: prefetch head → stream segments
Animation: header + first segment → later segments
Shader: boot set → deferred variants
```

`prism_asset` 只等待 `Immediate` 闭包即可进入可玩状态，其余由预算驱动流入。构建期验证最小闭包是否自洽，避免运行时才发现缺少低 LOD 或基础材质。

### 15.3 热重载握手

编辑器修改触发增量 Build，成功后发布新开发 Manifest Revision：

1. Build 在后台生成并验证全部受影响 Artifact。
2. `prism_asset` 预取新 Artifact 的最小闭包。
3. 在安全帧边界原子切换 Revision。
4. GPU/音频资源完成 fence 后回收旧版本。
5. 失败则保留旧 Revision，编辑器展示诊断。

不允许运行时观察到新材质配旧参数布局、或新场景配缺失依赖的混合状态。

---

## 16. 编辑器与 CLI 易用性

### 16.1 资产浏览器

支持按类型、标签、Owner、状态、平台、质量风险、内存、包归属、DDC 状态和最近修改筛选；提供依赖/反向依赖、构建制品、发布去向和历史 Revision 查询。

### 16.2 Import Inspector

- 基础/高级两级设置。
- PC/主机/移动等平台效果和成本并排预览。
- 设置变更前显示失效节点、预计耗时、缓存命中和包体变化。
- 提供 Before/After、Mip/LOD/压缩误差、骨骼权重和通道检查视图。
- 多选批改通过规则层覆盖，避免把重复配置写入每个资产。

### 16.3 Build Monitor

展示 DAG 总进度和关键路径，而非简单节点计数：

- 当前阶段、关键路径剩余估计和各资源池占用。
- Cache Hit/Miss 及 Miss 原因。
- 正在运行/排队/阻塞节点。
- 单节点 CPU、内存、I/O、输出大小与日志。
- 取消、重试、降低优先级、定位资产。
- “为什么重建”因果链。

### 16.4 CLI

```text
prism asset scan
prism asset import [--changed] [--dry-run]
prism asset build --recipe editor-preview
prism asset cook --platform <id> --profile <name>
prism asset validate --baseline <manifest>
prism asset package --manifest <id>
prism asset patch --base <id> --target <id>
prism asset inspect <asset-id|path|artifact-key>
prism asset graph <asset-id> [--why] [--reverse]
prism asset diff <revision-a> <revision-b>
prism asset cache status|verify|prune
prism asset doctor
```

当前可复核的本地 PNG 发行命令是：

```text
prism_asset_build_cli build --source <png> --logical-path <project-relative-path> --asset-id <u128> --store <store>
prism_asset_build_cli package --store <store> --output <runtime-package-dir> --asset-id <u128> --logical-path <project-relative-path>
prism_asset_build_cli validate --store <store> --package <runtime-package-dir> --asset-id <u128> --logical-path <project-relative-path>
```

`package` 从当前已提交 Revision 读取 Manifest/CAS，先运行 `ArtifactQualityGate`，再调用 `package_runtime` 写出 canonical `manifest.bin`、`publication.bin` 和 raw container；`validate` 重新计算同一 package 并逐字节校验发行目录。该命令组只覆盖当前 PNG 纵切片的 source/image 映射（`source AssetId = image AssetId - 1`），不代表通用 Cook、平台 Profile、压缩/加密或签名发行链。

所有命令支持人类文本和稳定 JSON/Event Stream；CI 依赖错误码与结构化数据，不解析日志字符串。`--dry-run` 必须给出预计重建范围、DDC 命中、输出大小和门禁风险。

### 16.5 可解释性

每个 Artifact 可追溯到：源版本、Asset Revision、设置、Profile、Processor/工具链版本、输入 Artifact 和 Build Request。用户应能回答：

- 为什么它被重建？
- 为什么它这么大？
- 为什么它在这个包中？
- 为什么选择这个压缩格式？
- 哪个资产把它带入依赖闭包？
- 这个结果由哪台 Worker、哪个工具版本产生？

---

## 17. 事务、一致性与崩溃恢复

### 17.1 两阶段提交

> **当前实现边界**：这里描述的是目标两阶段提交与恢复协议。当前仓库已实现本地带 `TransactionId` 的 append-only journal、Prepare/Commit/Abort marker、Build-only 与 RuntimePackage 发布边界、publication descriptor/hash 绑定、启动恢复扫描、孤儿 manifest/runtime 输出清理、提交状态查询、按 transaction id 的显式 reconcile 和 Revision 通知重放，并覆盖了“CURRENT 已替换但 commit marker 丢失”和 journal 尾帧截断恢复；另有带 grace period 的本地 CAS 可达性 GC。CAS blob 仍不随事务即时回滚；完整磁盘故障注入、跨平台发布协调、远程幂等重试协议和事务级持久化 registry 仍未完成。`CURRENT` rename 成功但后续目录同步失败时，恢复依赖下一次打开或显式状态查询。

```text
Prepare
  build artifacts → validate → prepare registry delta → prepare manifest
Commit
  commit CAS blobs → commit descriptors → append registry revision
  → atomically publish manifest pointer → emit BuildCommitted
```

事务日志为 append-only，带校验和与提交标记。启动恢复时：

- 无提交标记的 Prepare 事务回滚。
- Artifact Blob 可留为不可达对象，交由 GC。
- 已提交但前端未收到事件的事务可重放通知。
- Registry Revision 与 Manifest 指针不能指向缺失 Blob。

### 17.2 乐观并发

构建请求捕获输入 Revision。提交前若资产被再次修改，则旧构建标记 `Superseded`，不得覆盖新 Revision；其 Artifact 仍可进入 CAS，若新构建键相同可立即复用。

### 17.3 幂等与重试

同一个 Node Plan 重跑必须得到同一输出或明确的 Non-deterministic 错误。外部副作用（上传、签名、商店发布）放在 Publish 阶段，通过幂等请求键和审计日志管理，不作为普通可重试 Processor。

---

## 18. 确定性、可复现与审计

### 18.1 确定性纪律

- 使用稳定哈希和稳定排序。
- Map/Set 序列化前按规范键排序。
- 规范化 NaN、`-0`、浮点精度和舍入模式。
- 随机算法使用由输入键派生的显式种子。
- 时间、机器名、绝对路径、线程调度不得进入输出。
- 并行归约必须采用确定顺序或可证明等价算法。
- 外部工具版本、参数和运行环境进入工具链指纹。

### 18.2 Provenance

每次发布保存：源码/内容修订、Recipe、ProfileHash、引擎版本、插件版本、工具链、输入 Manifest、Worker 能力、Artifact/Chunk 哈希、验证报告和签名信息。Provenance 与大型调试数据分离，但发布索引必须能定位它。

### 18.3 复现级别

1. **Semantic reproducible**：运行时语义一致，字节可不同；仅用于明确标注的非发布工具。
2. **Byte reproducible**：Artifact 逐字节一致；普通缓存和 Shipping 的默认要求。
3. **Cross-host reproducible**：不同合规主机一致；官方发布门槛。

不满足 Byte Reproducible 的 Processor 不得进入共享发布 DDC，除非它的输出在后续确定性规范化节点被完全消除差异。

---

## 19. 安全与不可信输入

- 源文件、插件输出、远程缓存、容器和网络响应全部视为不可信。
- 解析前检查长度、偏移、数量、递归深度和整数溢出。
- 解压前检查声明尺寸与全局/节点内存上限，防止解压炸弹。
- Worker 默认最小权限、隔离临时目录、禁止访问未声明工作区内容。
- 远程 CAS 使用传输认证、内容哈希复验和命名空间授权。
- Container/Manifest 采用签名和防回滚版本策略。
- Secret 不进入日志、Artifact、缓存键或 Worker 环境快照。
- Processor 插件启用前验证来源、签名、ABI、许可证和允许能力。
- Fuzz 常驻覆盖源解析器、IR/Artifact 反序列化、ToC、Patch 和解压入口。

---

## 20. 可观测性与成本模型

### 20.1 Trace

统一关联 ID：`BuildRequestId → GraphId → NodeId → ArtifactKey → BlobHash`。每个节点产生规划、排队、缓存查询、输入下载、执行、输出上传、验证和提交 Span，可跨本地与远程 Worker 拼接。

### 20.2 指标

- 图规划时间、关键路径、并行效率。
- 各 Processor p50/p95/p99 时延、失败率、重试率。
- CPU 时间、峰值内存、磁盘/网络字节、GPU 时间。
- DDC 各层命中率、命中延迟、回填量、错误率。
- 每种 Artifact 大小、压缩比、去重率、Patch 放大。
- 每资产“变更到可预览”和“变更到运行时生效”延迟。
- 发布闭包大小、首帧闭包大小、运行时读放大和页局部性预测。

### 20.3 成本归因

Artifact 与包体成本可归因到 Asset、关卡、团队 Owner、功能和变更列表。共享 Artifact 按“唯一成本”和“摊销成本”分别展示，避免多个团队重复背锅或无人承担共享内容成本。

---

## 21. 性能目标（尚非已验证门槛）

以下是第一阶段待测工程目标，不是当前性能承诺或 CI 阻断条件。只有提交可复现 benchmark、固定数据集、硬件/OS/编译配置、预热与采样方法、统计分位和方差带后，才可将对应切片升级为门槛；最终按工作站、构建机和平台基线分别校准：

| 维度 | 指标 | 初始目标 |
|---|---|---:|
| Registry | 百万资产按 ID 查询 p99 | < 100 µs（含本地索引访问） |
| 规划 | 10 万节点增量图规划 | < 2 s |
| 交互导入 | 普通纹理变更到 Preview 可见 | p95 < 500 ms（缓存热） |
| 热重载 | 小资产变更到运行时 Revision 切换 | p95 < 1 s |
| DDC | 团队日常构建命中率 | > 90% |
| DDC | 本地命中元数据查询 p95 | < 2 ms |
| 调度 | 可运行节点调度开销 | < 20 µs/节点均值 |
| 内存 | 构建服务峰值 | 配置预算内，默认物理内存 70% 以下 |
| CAS | 大 Blob 本地读 | 接近顺序磁盘带宽的 85% 以上 |
| Cook | 单资产修改 | 仅重建精确后继闭包 |
| Package | 运行时有效读放大 | < 1.1× 目标内容字节 |
| Patch | 未改大型内容重发 | 0 字节 |
| 运行时 | Build 产物引入主线程流送开销 | < 0.5 ms/frame |
| 确定性 | Shipping 双跑字节一致率 | 100% |

目标数据集候选包括：十万小资产、8K/16K 纹理、电影级角色、开放世界 Cell、百万 Shader 请求、长音频和恶意损坏语料。数据集、运行脚本及固定硬件基线尚需落地；建立重复运行方差与趋势告警后，再决定哪些指标可阻断合入。

**当前验收门槛（可执行，而非性能承诺）**：任何新 Processor 必须通过输入声明完整性、确定性双跑、输出 descriptor/CAS hash 复验、失败不发布和重启恢复测试；任何并行 BuildGraph 实现必须证明同一 DAG 在至少两种 worker 执行顺序下产生相同 Manifest/Artifact bytes，并验证并发上限为正且不会因调度器创建无限线程；任何 RuntimePackage 发布必须验证来源 Build Manifest 身份、Container 长度和 BLAKE3 内容绑定。跨平台锁语义未实现的平台必须明确返回“不支持协调”，不得静默降级为无锁发布。

### 21.1 首次全量与增量分别优化

- 全量构建重视吞吐、并行效率和远程执行。
- 增量构建重视规划延迟、精确失效和交互优先级。
- 不允许用全量吞吐优化破坏编辑器交互，例如一次扫描整个 Registry 才处理单张纹理。
- 不允许只优化热缓存；冷缓存和灾难恢复构建必须有明确容量规划。

---

## 22. 易维护性与演进规则

### 22.1 稳定边界

必须明确版本化：

- Processor Protocol/IPC。
- Asset、IR、Artifact Schema。
- Artifact Key 规范。
- Runtime Manifest 与 Container 格式。
- Shader/Material/Geometry Runtime ABI。
- Profile Schema 和 Recipe Schema。

格式版本与引擎版本解耦，支持有限窗口内的旧格式读取和显式迁移；禁止“尝试猜测”未知版本。

### 22.2 Schema 迁移

- 编辑态资产允许链式迁移，并保留备份/预览 Diff。
- DDC Artifact 通常不迁移，旧键自然失效后重建。
- 已发布 Runtime Artifact 在兼容窗口内由运行时读取，或通过重新 Cook 升级。
- 迁移器本身版本化、可测试、可重复运行。

### 22.3 代码组织纪律

- 核心 crate 不依赖具体资产插件。
- Processor 不直接依赖编辑器 UI。
- GUI 不复制构建规则，只消费 Schema、事件和查询 API。
- 运行时协议类型不引用构建服务类型。
- 每个插件拥有最小明确写域、测试夹具和基准。
- 外部工具适配器集中管理，不把命令行拼接散落在节点代码中。

### 22.4 弃用策略

Processor、Schema 或 Profile 字段弃用时，先提供诊断和自动迁移，再经过至少一个兼容窗口移除。构建报告列出仍使用旧版本的资产和插件，避免升级时一次性爆炸。

---

## 23. 测试策略

### 23.1 测试金字塔

- **单元/属性测试**：ID、键规范化、Profile 合并、图算法、Schema。
- **Processor Golden**：固定输入和逐字节/感知输出。
- **契约测试**：第三方 Processor SDK、IPC、取消、资源声明。
- **集成测试**：导入→Build→Cook→Package→Runtime Load 全链。
- **并发模型测试**：作业合并、取消/完成竞态、事务提交与 Revision 冲突。
- **崩溃恢复**：在每个事务步骤注入进程退出和磁盘错误。
- **Fuzz**：解析器、序列化、容器、Patch 和缓存元数据。
- **Soak**：持续编辑、重导、GC、远程 Worker 上下线和网络抖动。
- **跨平台确定性**：不同 OS/架构/Worker 对拍。
- **运行时联测**：最小驻留闭包、热切换、流送和旧资源回收。

### 23.2 故障注入矩阵

至少覆盖：源文件半写、磁盘满、权限变化、远程 404/超时/损坏、Worker 崩溃、插件死锁、缓存元数据存在但 Blob 缺失、签名失败、Manifest 提交前崩溃、提交后通知前崩溃。目标是返回结构化错误、保持最后有效 Revision，且不泄漏租约或临时文件。

### 23.3 Processor 准入门槛

新增 Processor 必须具备：

1. 固定身份、语义版本和 Schema。
2. 最小成功/失败测试资产。
3. 确定性双跑。
4. 不可信输入或上游验证边界。
5. 取消与临时输出清理测试。
6. 峰值内存和吞吐基准。
7. 缓存键变化/不变化契约测试。
8. 诊断错误码和用户修复建议。

---

## 24. API 草案

### 24.1 构建请求

```rust
pub struct BuildRequest {
    pub roots: Vec<AssetSelector>,
    pub recipe: RecipeId,
    pub profile: ProfileId,
    pub expected_revision: Option<BuildRevision>,
    pub priority: BuildPriority,
    pub mode: BuildMode,
}

pub enum BuildMode {
    PlanOnly,
    Build,
    BuildAndPublish,
    VerifyReproducibility,
}
```

### 24.2 事件流

```rust
pub enum BuildEvent {
    GraphPlanned(GraphSummary),
    NodeQueued(NodeId),
    NodeStarted(NodeId),
    CacheHit(NodeId, CacheTier),
    NodeProgress(NodeId, Progress),
    Diagnostic(Diagnostic),
    ArtifactPrepared(NodeId, ArtifactDescriptor),
    NodeCompleted(NodeId, NodeStats),
    TransactionCommitted(BuildRevision),
    BuildFailed(BuildFailure),
    BuildCompleted(BuildSummary),
}
```

事件具有序号，支持断线续读；高频进度允许合并，诊断、状态迁移和提交事件不可丢失。

### 24.3 查询 API

```rust
trait AssetBuildQuery {
    fn explain_rebuild(&self, asset: AssetId, target: BuildTarget) -> RebuildExplanation;
    fn reverse_dependencies(&self, asset: AssetId) -> DependencyPage;
    fn artifacts(&self, asset: AssetId, profile: ProfileId) -> ArtifactSet;
    fn estimate(&self, request: &BuildRequest) -> BuildEstimate;
    fn diff(&self, a: BuildRevision, b: BuildRevision) -> RevisionDiff;
}
```

`estimate` 返回节点数、关键路径估算、Cache Hit、CPU/内存/I/O、输出大小和风险；这是编辑器易用性与 CI 容量规划的共同基础。

---

## 25. 实施路线

### M0：协议与垂直切片

- 建立 `AssetId`、Artifact、Processor、Profile、Build Event 契约。
- 本地 Registry、CAS/DDC、单机 DAG 调度和两阶段提交。
- 以“PNG/EXR → Texture IR → Mip → 平台纹理 → Runtime Load”贯通全链。
- CLI 支持 plan/import/build/inspect；完成确定性双跑与崩溃恢复测试。

**验收**：单张纹理变更只重建自身闭包；失败保留旧 Revision；第二次 Build 命中 DDC；运行时可加载发布 Artifact。

### M1：核心内容生产

- Mesh、材质、Shader、动画、音频 Processor。
- `.prism` 编辑资产、重导入和稳定子资产匹配。
- Editor Import Inspector、Build Monitor、依赖查询和可视化 Diff。
- Profile 合并、预算验证和结构化报告。

### M2：Cook 与发布

- Runtime Manifest、Chunk Planner、Container Writer。
- Shipping/Server/Localization Recipe。
- 签名、Provenance、Patch/DLC 和运行时叠挂。
- 首帧最小闭包验证和发布包成本归因。

### M3：AAA 流送内容

- VT/VG 页、World Partition、HLOD、长动画/音频分块。
- I/O 局部性规划、运行时遥测回灌和 GPU 解压制品。
- 热重载 Revision 原子切换与 GPU fence 回收闭环。

### M4：团队规模化

- 团队/CI 远程 DDC、Remote Worker、租约与能力匹配。
- Build Farm、分布式调度、跨主机确定性。
- 百万资产 Registry、十万节点规划和容量治理。

### M5：生产成熟度

- 自动质量/性能回归、构建成本趋势、异常检测。
- 多项目共享 Artifact、插件生态与版本治理。
- CDN 分阶段发布、回滚、在线内容目录与商店适配。

### 路线纪律

每个里程碑必须完成一条真实运行时垂直链，禁止先堆大量抽象再等待整合。远程服务、Build Farm 和智能布局都不能早于本地确定性、事务和可诊断性。

---

## 26. 关键风险与取舍

| 风险 | 后果 | 设计应对 |
|---|---|---|
| 过度抽象 Processor | 简单导入开发成本过高 | 提供类型化模板、默认实现和脚手架 |
| Artifact 粒度过细 | 图/元数据/调度开销过大 | 小输出可聚合；以独立缓存和流送价值决定粒度 |
| Artifact 粒度过粗 | 小修改导致大范围重建 | 拆分高频变化与稳定数据，按依赖语义切分 |
| Profile 组合爆炸 | 多平台缓存和测试失控 | 能力驱动、继承组合、只对实际目标实例化 |
| Shader Variant 爆炸 | 编译时间、包体和运行时缓存恶化 | 域约束、静动态分离、可达性裁剪和预算门禁 |
| DDC 污染 | 全团队复用错误输出 | 强键、内容复验、命名空间、确定性审计和隔离发布缓存 |
| 插件崩溃/不安全 | 编辑器不稳定或供应链风险 | Worker 隔离、能力最小化、签名和准入测试 |
| 构建与运行时 ABI 漂移 | 发布后无法加载 | 共享协议、兼容矩阵、端到端测试和 Manifest 门禁 |
| 只优化热缓存 | 冷构建不可接受 | 冷/热基准分开，规划构建农场和灾难恢复 |
| 自动优化不可解释 | 美术无法控制效果 | 输出理由、误差预览、预算归因和可覆盖策略 |

---

## 27. 最终决策摘要

Prism 应将本产品正式命名为 **Prism Asset Build**，并采用以下术语：

```text
Prism Asset Build  产品与系统总称
Recipe             对构建目标的声明
Processor          可插拔转换/验证节点
Artifact DAG       实际执行模型
Artifact           不可变构建制品
CAS                 按内容保存 Blob
DDC                 ArtifactKey 到制品的派生数据缓存
Profile             平台、能力、质量和预算策略
Manifest            一次成功构建的运行时映射快照
prism_asset         运行时消费者
```

最重要的架构约束是：

1. Asset 与 Artifact 分离，路径与身份分离。
2. 系统执行 Artifact DAG，而不是固定线性流水线。
3. 所有输出不可变、内容寻址、确定性且事务提交。
4. Recipe/Profile 描述策略，Processor 实现机制，核心负责调度与治理。
5. 编辑器、CLI、CI 和 Build Farm 使用同一个构建内核。
6. Source、Editable、Intermediate、Runtime 四层数据严格分离。
7. 运行时不承担可离线完成的高成本处理。
8. 效果、性能和包体全部量化，并进入验证、基准与发布门禁。
9. 第三方扩展通过稳定 SDK 接入，但不能改变身份、缓存和事务语义。
10. 本地闭环和真实垂直切片优先于远程服务与宏大抽象。

最终目标不是“可以导入资产”，而是让数百人的内容团队在多个平台上持续生产高质量内容时，仍然做到：**修改反馈快、构建结果稳、运行时成本低、问题可解释、发布可复现、系统可演进。**
