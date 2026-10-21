# Editor Scene Session：作者文档到 ECS 与视口

## 范围

`EditorSceneSession` 实现 opt-in `EditorSceneMesh` profile 的作者文档、真实 Prism ECS 和 `DesktopScene` 投影。CLI 创建/打开 `SceneProject` 时使用同一 catalog 和 session；空工程保持为空，不注入演示对象。它不是任意组件到几何的隐式转换器，也不代表完整 Editor 已交付。

`EditorController` 是唯一作者数据权威。场景与 ECS 只读投影，不另建 Undo/Redo。`world()`、`bridge()`、`scene()` 只提供只读访问；模型变更须经 controller 命令。明确摄像机通过 `with_camera` / `set_camera` 安装；默认使用单位 clip-space 矩阵，不推断项目相机或物理规则。

## 事务与差分同步

- controller 用实例身份、shape revision 与 model revision 区分外来文档、对象/层级变化和属性修改；选择等 UI-only 修改不触发几何重建。
- 有界 changed-fields 日志按对象合并顶层字段，Undo/Redo 也记录同样路径。日志缺口保守回退完整重建；idle 路径只核对身份/revision，O(1)。
- 完整构建先检查对象数、entity 身份和借用长度的累计几何预算，再执行 `FromReflect`、描述符克隆和 ECS 装配。上限为 4096 objects / 64 MiB vertex-index payload；这不是总进程内存上限。
- matrix/visible-only 更新只借用读取相应属性，保留 ECS entity 和 CPU geometry allocations。几何更新仅解码被修改对象，先检查最终累计预算；同批 grow/shrink 不按中间总量误拒绝。
- 全部 CPU identity、schema、matrix、geometry 与预算检查在发布前完成。候选 prepare 不改变 live ECS；discard 丢候选，commit 同时切换 ECS 与场景。prepare 期间禁止 synchronize 和摄像机变更，避免候选被覆盖。
- 活跃 controller 编辑或 Gizmo preview 期间暂停已提交投影同步。Gizmo 仍使用 drag-start 快照和 controller 稀疏命令，释放后归入同一文档历史。

controller 的 `ControllerObjectValidator` TypeData 在注册/解码时验证完整根，在隔离编辑候选完成后验证最终状态；不会因耦合字段的临时无效中间状态拒绝合法批次。readonly、metadata、标量转换和资源限制仍生效。准入失败不发布模型、selection、history 或 revision。

## GPU 成本与边界

`DesktopScene` 保留 geometry renderer，区分 geometry 和 visibility dirty 集合。visibility-only 只刷新 pass metadata，不上传几何或 transform；geometry-only 只上传被修改 mesh，不重建 renderer。矩阵更新通过现有 transform buffer 路径处理。设备恢复和完整文档替换仍可重建 GPU 资源。

`DesktopSceneStatus` 记录实际 geometry uploads/bytes、transform uploads 和 renderer creations；`EditorSceneSyncWork` 记录 ECS/描述符同步工作。计数不等价于帧时或 GPU profiling。

成本限制：controller 当前每个 touched object 的候选事务仍克隆一次完整根；不能把投影的借用路径称为整条编辑链零拷贝。完整构建会同时存在作者数据、typed ECS mesh 和 scene descriptor，GPU 另有 buffers；完整 GPU upload 会扫描 pass metadata。没有大型项目 P50/P95/P99 或跨后端性能结论。

## 运行入口

desktop 默认持续运行，只有显式 `--frames` / `--time-limit-seconds` 才有限时/帧约束；headless 默认一帧。显式帧数未达到时返回错误，包括没有呈现任何帧的情况。frames smoke 默认最多等待 30 秒，用户指定时间上限为 1..86400 秒。尺寸、重复/冲突选项和非 desktop 项目参数均严格检查。

## 验证

2026-10-22，在当前工作树运行：

```sh
CARGO_TARGET_DIR=/private/tmp/prism-editor-world-wake-target \
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 \
PRISM_REQUIRE_GPU=1 cargo test --offline -j 1 -p prism_editor_app \
  --features desktop --lib --bin prism_editor_app \
  --test controller --test controller_validation --test scene_session
```

exit 0：142 library + 17 CLI + 15 controller + 6 controller validation + 13 scene session，共 193 项通过，0 failed/ignored。日志：`/private/tmp/prism-editor-scene-session-regression.log`。强制 GPU 测试没有 adapter 时必须失败，覆盖真实 offscreen 可见性/拾取、Undo/Redo、renderer 复用和 dirty-only 几何上传。另有实际临时目录 SceneProject 保存/独立打开、反射 codec 到 ECS/hierarchy、候选围栏、非法几何与合法 sibling 的原子拒绝测试。

全文档预算哨兵直接调用生产 `document_geometry_bytes`，无需分配大型网格：测试空文档、精确累计、累计超限、迭代前对象数拒绝，以及读取 root 前非法身份拒绝。

上述不是原生输入/窗口恢复验收，也不是完整 World 或完整 Editor 交付。仍需真实桌面端到端输入、面向用户的对象创建/删除作者命令、World runtime/OFPA 正式接线、具体域工具、项目恢复/迁移、多窗口及规模性能验收。
