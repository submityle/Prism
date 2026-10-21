# Editor 可选 Mesh 对象创作

本切片为 `desktop` feature 下的 `EditorSceneMesh` profile 提供显式对象创建和删除，不是通用游戏对象模板，也不代表 Editor 全系统或 World runtime 接入已完成。

## 安装与入口

宿主对自己的 controller 显式调用：

```rust,ignore
install_scene_authoring_commands(&mut controller)?;
// 安装成功后再构造 UI / EditorSceneSession；已有 UI 安装后需 refresh。
let scene_session = EditorSceneSession::new(&mut controller)?;
```

这会原子注册命令工厂和 `scene.geometry_bytes` 文档聚合预算，不创建对象、不改变选择、不写模型历史、不标 dirty；成功安装新配置可以推进 UI revision。CLI 的 desktop 路径在 create/open 得到 controller 后、构造 `EditorSceneSession` 前安装。无项目启动仍是空 scene，没有默认示例数据。

已注册对应命令时，toolbar 显示：

| 按钮 | 命令 ID | 行为 |
| --- | --- | --- |
| Add triangle | `scene.add_triangle` | 创建一个可编辑 mesh 和根 Outliner row，并选中新对象 |
| Delete selection | `scene.delete_selection` | 将当前 selection 交给通用原子结构删除命令 |

按钮使用普通 `ExecuteCommand`，不是直接改 ECS 或 renderer。未提交 field draft、IME composition 或活动 gesture/pointer capture 会阻止这两条操作；应先提交、取消输入或结束交互，拒绝时保留 draft。直接使用 controller 的宿主仍须自行管理 UI draft，不能绕过 UI 后声称已处理输入围栏。

安装使用 `register_authoring_profile(commands, budgets)` 的有界候选，一次验证全部命令 ID、重复/既有冲突、剩余命令/策略容量、已有对象的预算贡献及 revision，再原子发布。命令槽位只剩一个时，两条命令和新预算均不注册；预算安装失败时，也不留下任何新命令。失败不推进 UI revision，不创建对象，也不改变选择、模型或历史。工厂只在用户执行命令时求值，注册不会调用它们。CLI 在安装错误时退出。仅需命令的 `register_authoring_commands` 仍通过同一入口传入空预算集合。

## 文档预算与预发布准入

`ControllerDocumentBudget::new(id, limit, contribution)` 定义宿主拥有的通用聚合策略；回调类型为 `Fn(&SelectionItem, &dyn Reflect) -> Result<u64, String> + Send + Sync + 'static`。它按反射字段读取 typed/dynamic root，须只读、确定且不依赖可变外部状态。`register_document_budgets` 可单独安装策略，`document_budget_usage(id)` 返回缓存的 `Option<(used, limit)>`，不重新扫描根。策略不序列化，完整契约见 [Editor 文档聚合预算](prism_editor_document_budget_zh.md)。

mesh profile 的 `scene.geometry_bytes` 上限为 64 MiB（`64 * 1024 * 1024` bytes），累计所有 mesh 的 position/index payload，隐藏 mesh 也计入；其他 schema 贡献零不等于场景支持渲染这些类型。计量只借用几何长度、不克隆或读取元素。bootstrap、结构创建/删除和属性批次都在 controller 发布前准入，不能等已提交模型投影失败后才报告超限。同批替换先减全部旧贡献，再加最终新贡献，允许合法的 grow/shrink 平衡；checked arithmetic、贡献回调错误或任一预算超限均原子拒绝。

属性历史保存稀疏贡献逆补丁，Undo/Redo 不重新调用计量回调，总量从当前基线推导并保留后来 bootstrap 对象的贡献；结构恢复允许重新计量涉及根。新策略只能在 undo/redo 均为空时加入；同一共享回调 `Arc` 和上限的策略 clone 可幂等重装，不受既有历史限制，但重复安装整个创作 profile 仍会因命令 ID 冲突失败。新增策略会使此前 prepared（包括空操作）失效。

`EditorSceneSession::new` 和 `with_camera` 现接收 `&mut EditorController`。构造先完整验证并 stage 投影，再以最后一个可失败步骤安装共享的几何策略，随后发布 scene；投影或安装失败不会留下部分新配置。已有历史但未安装策略的 controller 不能事后 attach，应在首次编辑前安装 profile/预算；已预装同一策略则允许带历史 attach。64 MiB 是当前几何 payload 准入，不涵盖历史、候选根、GPU 或总进程内存。

## 对象与历史语义

三角形使用三个顶点 `(-0.5,-0.5,0.5)`、`(0.5,-0.5,0.5)`、`(0,0.5,0.5)`，索引 `[0,1,2]`、identity matrix、`visible=true`。这是当前 identity viewport 下的最小可见创作配方，不引入相机或 gameplay policy。连续 Add 的对象初始重叠，用户可用已有 transform/Inspector 操作调整。

身份通过 `next_entity_identity()` 建议 entity/row pair。它使用 controller 生命周期高水位，涵盖 bootstrap、live、history 和已丢弃 redo 身份；建议本身不占用身份，create 原子成功才推进高水位。失败不会消耗身份，数值耗尽报错而非回绕。重开文档从已加载身份重建边界，旧会话由 controller/pick identity fence 隔离；这不是跨项目全局 ID 分配服务。

原子性、预算、schema 准入、选择快照与历史由通用 controller 提供：

- Create 为一个 history entry，默认选中新对象；预算/准入失败不改变模型、选择或历史。
- Delete 删除对象所关联 rows 的后代对象，以及受影响对象的所有别名/子树。它可能比一个可见 row 范围更大，不是仅隐藏当前 row。
- 删除闭包确定后，selection 只遍历一遍，按受影响身份集合裁剪，不再为每个删除对象重复扫描整个 selection；保留存活条目的插入顺序和仍存活的显式 primary（它可能不是最后一个条目），anchor 随被删除 row 清理。该优化不改变删除范围或 Undo/Redo 的选择快照语义，也不表示整个删除过程为无分配或常数时间。
- Undo/Redo 恢复/重做对象、rows 及对应选择快照。成功模型操作标 dirty；Undo 回原值也不等于已保存。
- scene session 从 controller 同步真实 ECS/scene 投影；命令工厂不直接上传 GPU 数据。
- 结构提交及其 Undo/Redo 清除字段增量日志。跨结构版本的 `changed_objects_since` 返回 `None`，要求消费者全量重建，而不是把创建/删除误报为无变化。结构之后的纯字段编辑重新形成有界增量日志；相同 model revision 查询仍返回空增量。bootstrap 注册另外由 `shape_revision` 围栏识别。

同一 shape 内的后续属性编辑使用 retained GPU 路径：encode 只对 geometry/visibility dirty 并集执行 metadata `apply_patch`，读取多次同步后的最终状态；纯几何更新保留 transform binding，geometry+matrix 混合更新只上传一次已改变矩阵。`DesktopSceneStatus::scene_metadata_upserts` 按实际提交的 metadata item 累计，便于断言单网格修改不扫描提交全场景。结构增删、完整文档替换和 fresh attach/recovery 仍允许完整同步，不承诺结构 GPU 资源增量复用；计数定义和回归见 `prism_editor_scene_session_zh.md`。

## 持久化与真实边界

对象可经 mesh schema catalog 和既有 document codec 保存/加载。磁盘保存仍需显式 SceneProject/document session，不会因 Add 自动创建项目或保存。命令工厂属于宿主配置，不序列化，也不会从磁盘读取可执行回调。

CLI 的初次 create/open 安装 profile。后续 `PreparedDocumentLoad::install` / `install_ui` 在候选及 durable base 发布前继承当前宿主的 action/authoring 工厂和文档预算策略；回调经 `Arc` 共享，不执行命令工厂。预算针对加载候选的新根重新计量，不能复制旧 controller 的 used 缓存；容量、计量或总量失败保留原 controller/UI、历史和 durable base。成功 Reload 清空文档历史及临时 UI 状态，保留窗口逻辑尺寸、宿主命令及策略，因此创作按钮可在新 UI 中重新构建。直接替换 controller 的其他宿主必须自行转交配置，不能从 codec 推断 profile。

不包含任意 mesh 导入、其他 schema 的对象创建菜单、层级重排/复制、多选创作 UI、OFPA、World cell payload 的正式 runtime consumers、物理或 gameplay 集成。Delete 使用当前 selection；此 profile 不限制删除对象必须是 mesh。

## 验证状态

新增 `tests/scene_authoring.rs` 覆盖：无隐式示例、重复/冲突命令、Add/Delete/Undo/Redo、选择恢复、redo 丢弃后身份不复用、bootstrap/身份耗尽、预算失败保留状态、递归 alias 删除、真实反射文档 codec 往返，以及 CPU ECS/scene 初始和增量同步。

`desktop_ui.rs` 内包含 retained-layout 按钮分发、opt-in 可见性、draft/IME 保留及活动 controller gesture 拒绝测试。selection pruning 的选择顺序/primary 语义、结构删除/Undo/Redo 后 range anchor 恢复，以及 GPU dirty-union、transform binding 保留等 patch 已纳入下述主线统一验证。

最新预算切片已由 session `59915` 完成强制 GPU 的 app + viewport、all features/targets 统一回归：353 passed、0 failed/ignored，exit 0；包含全部 16 项聚合预算测试及 Reload 重计量、场景构造签名/准入变更。session `75350` 的同范围 strict Clippy（`--no-deps -- -D warnings`）亦 exit 0；依赖树未发现 Bevy。精确命令、日志和覆盖边界见 `prism_editor_document_budget_zh.md`。

上一已完成切片的验证记录（主线报告，早于本次预算接线）：

- session `36236`：`PRISM_REQUIRE_GPU=1 cargo test --offline -j1 -p prism_editor_app -p prism_editor_viewport --all-features --all-targets`，exit 0；各结果为 `146/17/2/15/14/6/23/8/4/11/9/18/14/0/42`，共 329 项通过，0 failed/ignored。日志：`/private/tmp/prism-editor-authoring-final-gpu.log`。
- session `40272`：同两包 `--all-features --all-targets` 的 `cargo clippy --no-deps -- -D warnings`，exit 0。日志：`/private/tmp/prism-editor-authoring-strict.log`。
- `cargo tree` 检查同两包所有 features 的 normal/build/dev 依赖，无 `bevy` 匹配。

较早历史验收记录（均早于 latest GPU patch，保留作版本边界，不与当前结果相加）：

- app + viewport 强制 GPU 回归共 325 项通过。
- 结构定向回归 12 项通过；这是单独的历史记录，不与 325 简单相加。
- native smoke 已呈现 3 帧，但没有原生输入 E2E，不能证明鼠标/键盘创作、选择与 Undo/Redo 的完整桌面闭环。

本次仅依据主线结果更新文档，未另外运行 Cargo、测试或 GPU 验证。native 仍仅有 3 帧呈现、无 input E2E；原生窗口/设备输入、World 接线及上述未包含功能仍须分别验收，不宣称完整 Editor 完成。
