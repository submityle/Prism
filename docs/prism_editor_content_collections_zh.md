# 项目内容集合持久化

集合以项目稳定资产 ID 保存于规范化项目根目录的 `.prism/content-collections.bin`。持久化实现通过 `prism_editor_ui` 根级导出 `CollectionStore` 等类型，使用标准库，不增加 Bevy 或其他依赖。集合名称、ID、成员与下一个集合 ID 均跨会话恢复；最高 ID 对应的集合即使删除，重启也不会重用该 ID。文件不保存搜索条件、页面、选择或缩略图。

## 原生宿主接点

`CollectionStore::for_project(absolute_existing_root)` 建立项目存储；`path()` 返回准确目标路径。根目录必须已存在、为绝对路径且不含 `..`。根目录经过 canonicalize；调用期间宿主负责维持可信、稳定的项目目录。已存在的元数据目录、锁、数据目标若为符号链接或非预期类型会被拒绝。这不是针对恶意并发改目录的沙箱。

宿主打开项目时，对该项目的新 controller 调用 `load_into(&mut controller, &catalog)`；完整验证通过后才替换集合，清空打开的集合并使旧视图过期。不存在数据文件时返回 `missing_file = true` 并将集合重置为空，避免沿用另一个项目的集合。加载不改 catalog 或资产源文件。

集合创建、重命名、删除、添加/移除成员仍经 `ContentBrowserAction` 和 controller 执行；成功后宿主显式调用 `save(&controller)`。文件 I/O 不发生在 Loom 构建或绘制过程中。宿主串行安排项目加载、集合命令及保存，切换项目时先结束旧项目操作，再切换 store/controller。文件锁只串行化文件操作，不合并两个陈旧 controller 的编辑；最后一次成功保存生效。

`CollectionLoadReport` 给出集合数、成员总数及按集合 ID/资产 ID 排序的 `missing_members`。尚未扫描到、已删除或当前 catalog 中缺失的资产保留成员 ID；不生成虚假资产行。后续真实 catalog 发布恢复该 ID 时，集合查询会重新显示资产。有效资产 ID 的完整 u128 范围均受支持，0 和最大值不被当成无效哨兵。格式非法、重复或乱序 ID 则拒绝整个文件，不能悄悄丢成员。

宿主应展示 `CollectionStorageError` 的结构化错误：字段限额、损坏偏移/原因、未知版本、忙锁、路径类型或具体 OS 操作/类别。加载失败不改当前内存或文件；保存也拒绝覆盖损坏/未知版本的已有文件。修复需要宿主先显式备份/移走坏文件，再由用户选择恢复数据，不会自动重建覆盖。当前切片不修改 editor_app；上述项目生命周期和原生错误呈现仍由 app 所有者接线。

## 有界格式和原子发布

硬上限为 64 个集合，每个 4096 个成员，名称 4096 UTF-8 字节，文件 5 MiB。读取前检查文件大小，读取流限制为上限加一个字节；每个长度字段先校验再分配。缺失成员报告最多 262144 条。名称须非空且已 trim，集合 ID 须正数、唯一且升序，成员须唯一且升序；下一个集合 ID 必须大于所有已保存 ID。耗尽的 u64 高水位允许恢复，controller 会拒绝继续创建。

v1 二进制格式为 `PRISMCOL`、LE u32 版本、LE u64 下一个集合 ID、LE u32 集合数量；每个集合含 LE u64 ID、LE u32 名称字节数、UTF-8 名称、LE u32 成员数量、升序 LE u128 成员。尾部 LE u64 FNV-1a 校验检测意外损坏，不提供认证。未知版本、截断、UTF-8 错误、校验错误及多余尾随内容均有失败报告。

项目的永久旁路锁 `.prism/content-collections.lock` 使用非阻塞 OS 文件锁；遇占用返回 `Busy`。写入在同目录使用 create_new 唯一临时文件、write_all、文件 sync_all 后 rename 替换，随后同步目录。rename 之前失败保留旧文件并清理本次临时文件；rename 后目录同步失败返回成功报告中的 `durability_warning`，明确文件已发布但持久化确认不足。进程被强制终止可能留下未发布临时文件，加载不会将其当成最终文件。没有递归目录扫描或清理他人文件。

同目录原子替换已在本机 macOS 文件系统验证。底层系统若拒绝替换已有目标，返回 publish I/O 错误并保留旧目标；不使用先删除旧文件的退化方案。保存/加载均显式使用锁，原子 rename 也使未加锁读者只能看到完整旧文件或新文件。

## 验证

真实文件回归位于 `pkg/prism_editor_ui/tests/content_collection_storage.rs`：独立子进程加载/保存后父进程重新加载、跨 controller 重建、Unicode、删除最高 ID 后不重用、Loom Element 事件和 RecordingBackend 实际协调、无变化零变更、缺失引用保留/恢复/显式移除、缺文件重置、损坏加载与保存不覆盖、全容量往返、恶意长度与超限稀疏文件、锁竞争、并发读者观察原子发布，以及符号链接/非普通文件拒绝。

2026-10-21 使用复用的隔离 manifest `/private/tmp/prism-content-diagnostics-ui-hkRI7n/Cargo.toml`，直接编译仓库实际 UI lib 与上述测试源码，执行 `cargo test --offline --manifest-path <manifest> -p prism_editor_ui --test content_collection_storage --test content_browser --test content_grid --test content_diagnostics --jobs 2`：持久化 9 项、浏览器 8 项、诊断 7 项、网格 7 项全部通过。该 manifest 的依赖按当前 UI Cargo 同步，debug=0、incremental=false；`cargo check --offline --manifest-path <manifest> -p prism_editor_ui --lib --jobs 2` 亦通过。没有将此前缩略图 GPU 测试计作本次持久化文件测试，也没有修改 app/content/全局文档。
