# EditorSceneMesh：可选的显式编辑器几何 profile

## 范围与接线边界

`pkg/prism_editor_app/src/scene_mesh.rs` 提供业务中立的 indexed triangle mesh 反射模型，不是 gameplay schema，也不把任意项目组件推断为几何。模型不携带 Character、Weapon、物理、authority、材质或项目层级规则。

本切片只维护该模块与本文。由主线负责 `desktop` cfg 模块声明/导出、ECS → `SceneDocumentAdapter`、场景候选的原子装配、默认入口和 changed objects 的差分同步。这里不隐式安装全局 schema，不创建 ECS entity，不修改活跃场景。

## API

```rust
#[derive(Clone, Debug, PartialEq, Reflect, Component)]
pub struct EditorSceneMesh {
    pub positions: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
    pub matrix: [[f32; 4]; 4],
    pub visible: bool,
}

impl EditorSceneMesh {
    pub fn validate(&self) -> Result<(), DesktopError>;
}

pub fn editor_scene_mesh_registry() -> TypeRegistry;
pub fn editor_scene_mesh_catalog() -> Result<DocumentSchemaCatalog, DocumentError>;
pub fn editor_scene_mesh_validate(root: &dyn Reflect) -> Result<(), DesktopError>;
pub fn editor_scene_mesh_matrix(root: &dyn Reflect) -> Result<SceneIdMatrix, DesktopError>;
pub(crate) fn editor_scene_mesh_geometry_bytes(root: &dyn Reflect) -> Result<u64, DesktopError>;

pub struct EditorSceneMeshTransform;
impl SceneTransformAdapter for EditorSceneMeshTransform { /* model / edits */ }
```

registry 显式注册根、`Vec<[f32; 3]>`、`[f32; 3]`、`Vec<u32>`、`u32`、`[[f32; 4]; 4]`、`[f32; 4]`、`f32` 和 `bool`。catalog 只 allowlist 本根的 schema version 1，使用完整 Rust type path；改变类型路径或持久化结构时，主线须提供明确迁移策略。

geometry、matrix、visible 默认均可编辑，不安装 readonly metadata。registry 在根 registration 中插入 `ControllerObjectValidator::new(...)`；主线 controller 在对象注册（含 codec 解码注册）和最终隔离 edits 候选上调用该准入规则。仅构造 registry/catalog 不执行校验。反射 codec 的预算和场景几何预算是两套独立限制，本模块不放宽 `DocumentLimits` 或 controller limits。

## 借用式准入与路径分流

`EditorSceneMesh::validate` 委托 `editor_scene_mesh_validate`。后者支持 typed 根和 codec 解码后的 dynamic 根：typed 借用容器，dynamic 严格检查根表示类型、恰好四个字段、容器表示类型/形状和叶子类型；完整扫描 finite vertices、triangle indices/bounds、单模型预算、visible 类型及 matrix。成功路径不分配几何副本，不调用整根 `FromReflect`、`apply` 或 `reflect_clone`。dynamic 几何先检查两个 list 的长度与预算，再访问任何元素。

注册回调接受最终候选根及全部 changed `FieldPath`，按以下规则分流：

- 空 paths、根路径或未知顶层路径：完整验证，用于注册、解码和保守回退。
- positions 或 indices 任一字段及其子路径：共同验证全部 positions、indices、预算和 bounds，不允许索引修改漏掉顶点约束。
- matrix 及其子路径：只读取并验证 16 个矩阵分量，不访问 positions、indices、visible；固定大小校验，无几何 clone。
- visible：只检查 bool 类型，不访问 geometry 或 matrix。
- 混合路径取上述校验组的并集，不因第一个字段通过而提前成功；任意组失败都拒绝候选，不返回部分修改。

局部分流依赖 controller 的前置不变量：原根已经完整准入、未列出的字段没有变化、提供全部有效 changed paths。它不是对任意畸形根只传 matrix 路径就能完成的全量验证；独立调用者应使用 `editor_scene_mesh_validate`。controller 在所有字段应用到最终候选后才调用回调，因此可以接受合法最终状态，而不要求每次单字段中间状态都合法。本模块不承诺 controller 内部候选事务也不克隆根。

## 全文档分配前的 geometry 预算预检

crate 内部 `editor_scene_mesh_geometry_bytes` 从 typed 根直接借用 positions/indices 的长度；dynamic 根检查表示类型、恰好四个字段及两个 list 的表示类型/容器种类，只读取长度。复用完整 validator 的长度准入与检查算术，要求非空、计数适配 u32、完整三角形，并返回不超过单网格 64 MiB 的 `positions.len() * 12 + indices.len() * 4`。畸形计数（含 `usize::MAX`）在算术或元素访问前被拒绝；通过 u32 计数约束后，u64 字节算术仍保留 checked 防护。

该函数不读取几何元素、matrix、visible，也不 clone 或执行 `FromReflect`；返回值只表示长度/预算通过，不证明有限顶点、索引 bounds、叶子类型、matrix 或 visible 有效，不能替代完整准入。主线 scene_session 的 `document_geometry_bytes` 先检查 ExactSizeIterator 的对象数，再逐个验证 entity identity、借用计数，以 checked 累加并检查场景总字节上限；production `build_projection` 在任何 `FromReflect`、descriptor 或 ECS 装配前调用它。全文档累计预检和接线由主线维护，本切片提供单根计数 API 及跨模块哨兵回归。

## 矩阵读取与差分同步

`editor_scene_mesh_matrix` 的 typed 快路径只复制 16 个 f32；dynamic 路径要求根表示 `EditorSceneMesh`，借用名为 matrix 的字段，严格检查数组表示类型、外层 4 列、内层 4 行及 f32 叶子。codec 解码会保留这些表示类型；手工构造 DynamicArray 时也须设置。不会对整个根运行 `apply`、`FromReflect`、`reflect_clone`，也不读取 positions、indices、visible。调用方应使用 controller 的借用根，而不是先 snapshot 整个模型。

生产 session 已使用 controller 的 changed-fields 日志分流：仅 matrix/visible 的更新借用读取对应字段，不访问几何；geometry 字段变更才执行完整 `FromReflect` 并替换描述符。因此不再提供逐元素 geometry 比较 helper。准入前的借用验证不需要该转换，matrix-only 变化通过 getter 同步。主线仍须管理对象增删、identity、总预算、失败回滚和场景发布原子性。本模块不承诺现有 controller 内部的候选事务或调用方没有其他 clone。

## 校验规则

- positions 与 indices 均非空，计数不超过 u32；indices 长度为 3 的倍数，每个索引严格小于 positions 长度。
- 所有 position 分量有限；隐藏对象也必须通过相同检查。不额外禁止退化三角形、共享顶点，不限定项目坐标范围。
- 单个模型的有效顶点/索引负载 `positions.len() * 12 + indices.len() * 4` 不超过 64 MiB，使用检查算术。与 `DesktopScene` 一样不按 Vec capacity 或反射序列化开销计数。完整三角形和顶点记录均以 12 字节递增，最接近上限的可达有效负载为 64 MiB − 4 字节。
- 场景累计最多 4096 objects / 64 MiB 仍由主线和 `DesktopScene` 控制；单模型通过不意味着多模型组合通过。
- matrix 列主序，所有分量有限，最后一行严格为 `[0, 0, 0, 1]`，3×3 basis 不得奇异。以 f64 精确承载两个 f32 的乘积，通过 FMA 分离第三次乘法的舍入残差，再用固定栈空间的精确展开累加判零，避免普通 f64 行列式相消后把奇异矩阵误判为可逆。此检查不分配堆内存，不设置任意最小 scale；允许最小正次正规 scale、shear 与镜像，不要求正行列式。可逆不等于后续任意病态矩阵求逆都有数值稳定性保证。

## 累计 TRS 与编辑原子性

`EditorSceneMeshTransform::model` 委托 matrix getter；`edits` 每次从调用方提供的 drag-start 根读取矩阵。顺序和算术与 `examples/native_gizmo/support.rs` 一致：局部三个 basis 分别乘 scale，依次绕世界 X/Y/Z 轴旋转，最后按世界坐标增加 translation。旋转作用于自身 pivot，不旋转/缩放 translation，不推断父子关系或共享选择 pivot。

scale milliunits 必须逐轴严格大于零，1000 是单位缩放；`TransformDelta::default()` 的零 scale 无效，调用方需要显式给 `[1000; 3]`。translation 为 milliunits，rotation 为 millidegrees。输入矩阵和运算结果都校验；scale 导致非有限值或下溢为奇异矩阵时拒绝。

全部校验后才生成 16 个 `ControllerFieldEdit`，路径为 `matrix[column][row]`（FieldPath 内部会规范化为 `.matrix[...]`）。任何错误均返回 Err，没有部分批次，也不修改原根。主线将批次交给 controller 的 prepare/commit 协议管理 Undo/Redo 和场景候选验证。

## 测试与验证状态

主线已于 2026-10-22 导出并接入本模块，强制 GPU 的 app library/CLI/controller/scene session 定向回归共 193 项通过（exit 0），包含本模块全部单元测试及生产累计预算哨兵。命令、日志与成本边界见 `prism_editor_scene_session_zh.md`。下文“未运行”指并行模块作者交接时的历史状态，不是主线当前验证结果。

模块内单元测试覆盖完整递归 registry、Component 约束、真实 `DocumentCodec` 往返后的 DynamicStruct/DynamicArray、typed/dynamic matrix 与 TRS、世界 XYZ 旋转顺序、累计预览不修改原根、批次提交与 Undo/Redo、geometry 默认可编辑、有限/affine/singular 检查、非法 scale 与输出 overflow/underflow、geometry/index/budget 边界与畸形动态字段。新增借用式完整 validator、TypeData 注册/clone、空路径及根路径全量校验、混合路径及 geometry 耦合、controller 非法注册/最终候选拒绝回归用例。会 panic 的反射哨兵检查 matrix getter/TRS 及 matrix-only 注册回调不访问 geometry/visible 的类型、值或 clone，visible-only 不访问 geometry/matrix。移除已被 changed-fields 快路径替代的 geometry 比较 helper、专属测试和混合测试中的相关调用，其余 validator/getter 回归保留。

预算边界测试使用实际 validator 的长度 helper 和禁止 clone 的合成借用列表，直接验证 dynamic 完整/路径回调先拒绝超预算长度再读取元素，不分配 64 MiB 的测试网格。字节计数 API 另覆盖真实 codec dynamic/typed 一致性、接近单网格上限的大长度成功计数、positions/indices 任一超预算及组合超预算、空/非三角形/u32 越界/`usize::MAX` 计数拒绝、错误根与容器 schema。两个合成 list 禁止 clone 和任何元素读取，matrix/visible 也使用禁止访问的哨兵，验证成功计数和拒绝路径均不触碰这些数据。只对该模块执行定向 rustfmt；按本任务限制不运行 cargo，因此本轮测试代码尚未编译/执行，也不宣称全包集成、GPU 或原生窗口闭环已经验收。

全文档预检测试直接调用生产 `scene_session::document_geometry_bytes`，复用禁止元素读取/clone 的 `length_only_root`：空文档返回 0，小文档精确返回 84 字节，大长度但累计预算内返回 64 MiB − 16 字节；两个各自通过单网格预算的根累计超限时拒绝。超过 4096 objects 时迭代器一旦前进即 panic，验证计数拒绝先于任何 root 扫描；asset/sub-object identity 搭配禁止访问的根，验证身份拒绝先于根读取。非法长度用例还同时交给完整 validator 和 geometry-edit 回调，约束它们与字节 helper 的拒绝规则一致。以上均为新增测试源码，不代表已运行通过。
