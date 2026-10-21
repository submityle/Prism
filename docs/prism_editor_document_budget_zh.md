# Editor 文档聚合预算

## 提交前的资源契约

`ControllerDocumentBudget` 是宿主显式安装的、引擎中立的聚合资源策略。
策略由静态 ID、`u64` 总量上限和只读反射对象计量回调组成；回调不能修改
模型、执行外部操作或依赖可变外部状态。它必须对相同 identity/root 给出
相同结果。策略与命令工厂一样属于宿主配置，不从文档读取或序列化。

`register_document_budgets` 安装策略；`register_authoring_profile` 把整组命令
和策略作为一个原子配置安装。任何 ID、容量、计量、总量或 revision 错误
均不发布其中任何部分。默认最多 16 个策略，硬上限 64；零上限允许
零贡献对象，但不允许正贡献对象。`document_budget_usage` 读取缓存的
used/limit，不扫描或克隆对象。

策略 ID 必须非空且在 action 文本长度上限内。同一策略的 clone 共享
回调身份，可重复安装且不推进 revision；相同 ID 的不同回调或不同上限
不能覆盖既有策略。新策略只能在 undo/redo 都为空时加入，避免已有历史
缺少逆向计量。重新安装已有同一策略不受这条历史限制。

## 稀疏计量与原子发布

- 首次安装只遍历当前文档一次，验证完整初始总量。
- bootstrap 注册、结构创建/删除与属性批次都在模型发布前验证总量。
- 属性批次只计量最终发生变化的候选根，不按字段重复计量或扫描未变化根。
- 一批替换先减去所有被替换对象的旧贡献，再累计全部新贡献，允许同批
  grow/shrink 平衡，不因对象身份排序而误拒绝。
- 所有加减使用 checked arithmetic。溢出、回调失败与超限均保留模型、
  selection/anchor、历史、dirty、身份高水位及 revisions。
- 标量历史保存涉及对象的贡献逆补丁，Undo/Redo 不重新调用计量回调或
  为计量克隆根；总量从当前基线推导，保留其后 bootstrap 注册的贡献。
- 结构历史仍移动真实根的所有权；恢复时只重新计量本次涉及的根，
  不为预算克隆几何或扫描整个文档。

`PreparedControllerEdit` 携带策略配置围栏；prepare 后加入策略会使旧候选
提交失败，即使旧候选是空操作。既有 controller identity、shape 和 model
revision 围栏继续适用。不存在绕过预算的直接可变模型 getter。

## 重载与场景 profile

`PreparedDocumentLoad::install/install_ui` 在发布候选及 durable base 前继承
当前宿主的策略与命令。策略必须针对加载候选的新根重新计量，而不能复制
旧文档的 used 值。失败保留原 controller/UI、历史和文件适配器状态。
成功重载清空文档历史，不清除宿主策略，也不从文件加载可执行代码。

indexed-mesh profile 安装 `scene.geometry_bytes`：所有 `EditorSceneMesh`
（包括隐藏对象）的 vertex/index payload 总量最多 64 MiB，其他 schema
贡献零。策略使用共享不可变回调，因此 profile 安装和 scene session attach
可以重复使用同一配置。投影仍明确拒绝不支持的 schema，预算的零贡献
不代表该对象能被场景 profile 渲染。

`EditorSceneSession::new/with_camera` 接收可变 controller：先验证和 stage
完整投影，再以最后一个可失败步骤安装策略，然后发布 scene。未预装策略
且已经有作者历史的 controller 不能事后 attach；宿主应在首次编辑前
安装 profile/预算。scene session 的累计预检仍保留为防御性投影边界，
不能用它替代 controller 提交前 admission。

## 资源边界

几何贡献计算的是实际顶点/索引 payload，不是容器 capacity、序列化字节、
GPU 驻留、readback staging、历史根或 retained UI 的内存。各层仍独立执行
自己的预算。这个契约不把所有 Editor 内存等同于 64 MiB，也不构成任意
资源导入、World/OFPA 恢复、原生输入 E2E 或完整 Editor 交付证明。

## 验证（2026-10-22）

限定 target 为 `/private/tmp/prism-editor-world-wake-target`，离线、单 job、
关闭 incremental 与 dev/test debug info；以下结果针对本次未提交预算实现，
不是其他并行模块或整个 Editor 产品的完成证明。

- `PRISM_REQUIRE_GPU=1 cargo test --offline -j 1 -p prism_editor_app -p prism_editor_viewport --all-features --all-targets`：session `59915`，exit 0；
  共 353 passed、0 failed、0 ignored。包含 16 项 typed/dynamic 聚合预算回归、
  24 项文档、15 项场景作者、21 项场景会话回归，以及既有真实 GPU 路径。
  日志 `/private/tmp/prism-editor-budget-final-gpu.log`。强制开关不允许用无 GPU
  时的可选跳过充当通过；这仍不是原生鼠标/键盘 E2E。
- 同两包 `cargo clippy --offline -j 1 --all-features --all-targets --no-deps -- -D warnings`：
  session `75350`，exit 0。日志 `/private/tmp/prism-editor-budget-final-strict.log`；
  依赖包的既有 warning 不属于这项 `--no-deps` 零告警结论。
- 同两包 `cargo tree --offline --all-features --edges normal,build,dev`：
  无 `bevy` 包匹配，日志 `/private/tmp/prism-editor-budget-final-dependencies.log`。

预算回归覆盖首次安装/ bootstrap 超限、贡献错误与整数溢出、批量 grow/shrink、
多独立预算原子性、create/delete/Undo/Redo、稀疏回调、后续 bootstrap 基线、
命令与预算联合注册、Arc 身份幂等、历史限制、空/no-op prepared 配置围栏及
跨 controller 拒绝。文档回归验证两个 reload 安装入口对新根重计量，失败保留
原模型、UI、历史及 durable base。
