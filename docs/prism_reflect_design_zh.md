# Prism Reflect 顶级次世代 AAA 级反射 / 序列化 / 类型系统设计方案

> 面向 Prism（脱离 Bevy 后的独立引擎）的 **静态+动态反射 + 类型注册表 + 反射驱动序列化 + 属性元数据 + 路径访问 + 函数反射** 内核设计。它是 `bevy_reflect` 的自研替代，是场景序列化、编辑器检视器、脚本桥、网络复制、ECS 动态组件的统一类型基座。
> 借形态不抄码。借鉴：
> - **Rust 反射人体工学**：Bevy reflect（`Reflect`/`TypeRegistry`/`FromReflect`/动态结构/`reflect_trait`/函数反射/`apply` 局部更新）
> - **运行时元类型**：flecs meta（运行时类型描述符 + 反射驱动序列化 + 反射驱动 UI）
> - **编辑器暴露**：Unreal `UPROPERTY/UCLASS`（元数据驱动编辑器/复制/序列化）、Unity SerializedProperty/Inspector
> - **序列化**：serde（derive 心智）但走**反射驱动、无需每类型手写**；带 schema/版本化迁移
> - **C++ 反射**：RTTR（类型注册 + 动态属性/方法调用）
> 本文为纯经典类型系统/序列化路线，**不含任何 AI/ML 内容**。

- 版本: v0.2（核心 M0–M6 已落地并验证；§24 高级增补仍为设计阶段；v0.1→v0.2 新增第 24 章「AAA 高级功能增补」：derive 期静态 TypeInfo 零注册成本/热路径访问器代码生成/二进制零拷贝序列化/部分 patch 与反射 diff/反射驱动字段级网络增量/脚本与编辑器属性桥/自定义特性属性/不可信反序列化安全边界；均为 PLANNED，无代码）
- 适用引擎: Prism（后 Bevy 时代，独立运行时）
- 关键依赖: `prism_utils`（稳定哈希/句柄/小容器）、`prism_math`（为数学类型实现反射），可选 `prism_diagnostic`
- 层级定位: ECS 文档 L1 地基；服务 ECS（动态组件/快照）、场景、编辑器、脚本、网络
- 明确约束: 核心 `no_std + alloc`；`std` / `serialize` / `functions` / `documentation` / `determinism` 为 feature；**不依赖任何 `bevy_*` crate**

---

## 目录

1. 设计哲学与目标
2. 参考产品取舍
3. 档位化（capability / quality tier）
4. 分层架构
5. 核心模型：Reflect / TypeInfo / TypeRegistry
6. 反射种类（struct / enum / tuple / list / map / 值 / 不透明）
7. 动态值与 FromReflect（Dynamic* ↔ 具体类型）
8. 局部更新 apply / patch（prefab 覆盖 / 热重载 / 回滚）
9. 路径访问（"transform.translation.x" / 索引 / 键）
10. 反射驱动序列化（无需手写 serde + schema + 版本化迁移）
11. 属性元数据（范围 / 提示 / 分类 / 只读 / 隐藏）
12. 函数反射与方法调用（脚本桥）
13. 反射 trait 对象（动态 trait 分发）
14. 稳定类型 ID 与跨构建/网络一致
15. 与 ECS / 场景 / 编辑器 / 脚本 / 网络集成
16. 可观测性与诊断
17. 高级功能增补
18. 性能工程
19. 易用性与 Bevy 迁移策略
20. crate 分层与模块布局
21. 契约、不变量与版本化
22. 路线图（M0–M6）与基准即规格
23. 诚实边界与风险
24. AAA 高级功能增补（v0.2）

---

## 1. 设计哲学与目标

反射是把「Rust 的静态类型世界」接到「数据驱动的动态世界」（编辑器、脚本、序列化、网络）的唯一桥梁。没有它，ECS 动态组件、场景保存/加载、编辑器 inspector、脚本读写、网络复制都得各自手写一遍——`prism_reflect` 让这些**统一走一套反射协议**，组件只需 `#[derive(Reflect)]` 一次。

**一句话定位**：`prism_reflect` 是 Prism 的「类型真相层」——编译期零成本的静态类型信息 + 运行时可注册/可查询/可动态读写/可序列化的类型系统；一次 derive，处处可用（序列化、inspector、脚本、复制、diff/patch）。

四条总目标（按权重）：

1. **性能**：静态 `TypeInfo` 编译期生成（`OnceLock` 缓存，零运行时构造）；反射访问走索引而非字符串查表（路径预编译）；序列化流式零中间分配。
2. **效果（能力）**：动态结构/枚举、局部 patch、函数反射、稳定类型 ID、schema 版本化迁移，支撑编辑器/脚本/网络/存档兼容等 AAA 刚需。
3. **易用**：`#[derive(Reflect)]` 一行；序列化无需每类型写 serde；元数据用属性标注；与 `bevy_reflect` 近乎一致。
4. **可移植 + 档位化**：核心 `no_std + alloc`；序列化/函数反射/文档等按 feature 裁剪，移动端可只带最小反射。

非目标：不做跨语言 IDL；不替代 serde（可互操作）；不保证反射热路径与原生字段访问同速（反射用于工具/边界，非内层循环）。

---

## 2. 参考产品取舍

| 产品 | 吸收 | 规避 |
|---|---|---|
| Bevy reflect | `Reflect`/`TypeRegistry`/`FromReflect`/Dynamic*/`reflect_trait`/函数反射/`apply` | 字符串路径偶有开销、动态分配 |
| flecs meta | 运行时元类型、反射驱动序列化/UI、可运行时定义类型 | C ABI 风格 |
| Unreal 反射 | 元数据驱动编辑器/复制/序列化、UHT 生成 | UObject 耦合、代码生成器重 |
| Unity Serialized | Inspector 绑定、SerializedProperty 路径 | C#、GC |
| serde | derive 心智、流式 Serializer/Deserializer | 需每类型实现、无运行时类型注册 |
| RTTR | 类型注册、动态属性/方法调用、变体 | C++、手动注册啰嗦 |

综合：**Bevy reflect 人体工学 + flecs meta 运行时元类型 + Unreal 元数据驱动 + serde 流式序列化** 四支柱，叠加 **schema 版本化 + 稳定类型 ID + 函数反射脚本桥** 的 AAA 能力层，全部走 feature/档位门控。

---

## 3. 档位化（capability / quality tier）

| 维度 | 说明 | 示例 |
|---|---|---|
| **capability** | 运行时环境 | 是否带编辑器/脚本（决定是否注册函数反射与文档元数据） |
| **quality tier** | 携带形态 | runtime-min（只反射需序列化的类型）/ editor（全元数据+函数+文档）/ tool（离线烘焙全反射） |
| **feature flag** | 编译期裁剪 | `serialize` / `functions` / `documentation` / `determinism` |

目标：发行版运行时只带「序列化/网络/存档」所需反射；编辑器构建叠加元数据、函数反射、文档字符串。

---

## 4. 分层架构

```
L5  集成     场景序列化 / 编辑器 inspector / 脚本桥 / 网络复制 / ECS 动态组件
L4  序列化   反射驱动 Serializer/Deserializer + schema + 版本化迁移
L3  访问     路径解析 / 字段遍历 / apply-patch / FromReflect
L2  类型     TypeRegistry + TypeInfo + TypeData（可插拔反射能力）
L1  反射核   Reflect trait + ReflectRef/Mut（struct/enum/list/map/value/opaque）
L0  derive   prism_reflect_macros：#[derive(Reflect)] + 属性解析
```

依赖严格向下；L0–L3 可 `no_std + alloc`，L4 序列化在 `serialize` 档。

---

## 5. 核心模型：Reflect / TypeInfo / TypeRegistry

```rust
pub trait Reflect: Any + Send + Sync {
    fn type_info(&self) -> &'static TypeInfo;   // 静态缓存，零构造成本
    fn reflect_ref(&self) -> ReflectRef<'_>;    // 下探：Struct/Enum/List/Map/Value...
    fn reflect_mut(&mut self) -> ReflectMut<'_>;
    fn apply(&mut self, value: &dyn Reflect);   // 局部更新（见 §8）
    fn as_any(&self) -> &dyn Any;
    fn reflect_hash(&self) -> Option<u64>;      // 反射哈希（去同步/diff）
    fn reflect_partial_eq(&self, other: &dyn Reflect) -> Option<bool>;
}

pub enum TypeInfo { Struct(..), TupleStruct(..), Enum(..), List(..), Map(..), Value(..), Opaque(..) }

pub struct TypeRegistry {
    types: HashMap<StableTypeId, TypeRegistration>,  // 类型 → 注册项
    // 每注册项挂若干 TypeData：ReflectSerialize/ReflectComponent/ReflectDefault/用户 trait...
}
```

- **静态 `TypeInfo`**：derive 生成、`OnceLock` 缓存——字段名/类型/属性在编译期定好，运行时零构造。
- **`TypeRegistry`**：按稳定类型 ID 查类型；每类型挂可插拔 **TypeData**（如 `ReflectSerialize`、`ReflectDefault`、`ReflectComponent`、用户 `reflect_trait` 产物），插件化扩展反射能力。

---

## 6. 反射种类

统一的 `ReflectRef/ReflectMut` 下探：`Struct`（具名字段）/ `TupleStruct` / `Enum`（变体+字段）/ `Tuple` / `List`（动态数组/`Vec`）/ `Array`（定长）/ `Map`（`HashMap`）/ `Set` / `Value`（叶子：数值/bool/字符串）/ `Opaque`（不透明：手动实现）。每种暴露统一遍历/索引/插入接口，使序列化、inspector、diff 用一套代码处理所有类型。

---

## 7. 动态值与 FromReflect

- **Dynamic\***（`DynamicStruct/Enum/List/Map`）：无具体 Rust 类型的「鸭子类型」反射值，服务脚本/编辑器/网络构造任意数据。
- **`FromReflect`**：从任意 `&dyn Reflect`（含 Dynamic*）构造具体类型实例——反序列化、prefab 实例化、脚本回传的落地点。
- 动态值可与具体类型互相 `apply`，是 ECS 动态组件（ECS §16.2）的数据载体。

---

## 8. 局部更新 apply / patch

`apply(&mut self, patch: &dyn Reflect)`：只覆盖 patch 中出现的字段，其余保持——这是三大刚需的共同底座：

- **prefab 覆盖**（ECS §16.3 `IsA`）：实例在模板上 apply 覆盖字段。
- **热重载/实时调参**（ECS §16.2）：编辑器改一个字段即 apply 到运行实例。
- **回滚/网络增量**（ECS §16.5）：只传/只还原变化字段。

配合 `reflect_partial_eq`/`reflect_hash` 做 **diff**（算出最小 patch）。

---

## 9. 路径访问

```rust
value.path("transform.translation.x")       // 具名字段链
value.path("children[2].name")              // 索引
value.path("inventory[\"sword\"].damage")   // 键
```

- 路径可**预编译**成 `ParsedPath`（一串索引/键 token），避免每次字符串解析——inspector 绑定、脚本访问、网络字段选择器都用它。
- 支持读/写/类型查询，越界/类型不符返回错误而非 panic。

---

## 10. 反射驱动序列化

**无需每类型手写 serde**：Serializer 遍历 `ReflectRef` 把任意反射值写出；Deserializer 用 `TypeRegistry` + `FromReflect` 读回。

- **格式无关**：抽象 `Serializer`/`Deserializer` trait，可接 RON/JSON/二进制/自有紧凑格式。
- **schema + 版本化迁移**：每类型带 `schema_version`；读旧版数据时按注册的 **migration** 链逐版升级（存档/网络向后兼容的 AAA 刚需）。
- **流式零中间分配**：直接从反射遍历写入字节流，不建中间 `Value` 树（大场景序列化省内存）。
- **可与 serde 互操作**：对已实现 serde 的类型可桥接，不强制二选一。

---

## 11. 属性元数据

derive 属性标注，供编辑器 inspector 与校验使用：

```rust
#[derive(Reflect)]
struct Light {
    #[reflect(@0.0..=100000.0)]            // 范围（inspector 滑条 + 校验）
    intensity: f32,
    #[reflect(tooltip = "色温(K)", category = "外观")]
    temperature: f32,
    #[reflect(readonly)] id: u64,          // 只读
    #[reflect(skip)] cache: Cache,         // 不序列化/不暴露
}
```

元数据包括：数值范围、提示/文档、分类分组、只读、隐藏/跳过、默认值、单位——`documentation` 档还可携带字段/类型的文档字符串。

---

## 12. 函数反射与方法调用

`functions` 档：把自由函数/方法注册进反射，脚本/编辑器可按名动态调用：

```rust
registry.register_function("Transform::looking_at", Transform::looking_at);
let result = registry.call("Transform::looking_at", args)?;   // args: &[Box<dyn Reflect>]
```

- 参数/返回值经反射装箱/拆箱；类型不符在调用点报错。
- 这是**脚本桥**（`prism_script`）与**编辑器可调用操作**的底座，无需为每个 API 手写 FFI 胶水。

---

## 13. 反射 trait 对象

`#[reflect_trait]`：为某 trait 生成 `ReflectMyTrait` 的 TypeData，使持有 `&dyn Reflect` 时能动态取回 `&dyn MyTrait` 并调用——实现「数据驱动的多态分发」（如编辑器对所有实现 `Drawable` 的组件统一画 gizmo）。

---

## 14. 稳定类型 ID 与跨构建/网络一致

- Rust 的 `TypeId` 跨编译不稳定，不能用于序列化/网络。`prism_reflect` 为每反射类型生成 **StableTypeId**（基于类型路径 + 版本的稳定哈希），存档/网络用它标识类型。
- 支持类型重命名映射（旧 StableTypeId → 新类型），配合 §10 迁移保证存档兼容。
- `determinism` 档下类型 ID 与字段顺序确定，保证跨平台网络一致。

---

## 15. 与 ECS / 场景 / 编辑器 / 脚本 / 网络集成

- **ECS**：`ReflectComponent` TypeData 让反射值能插入/读取 World 组件（ECS §16.2/16.5 动态组件/快照的落地）。
- **场景**：反射驱动序列化整个场景/prefab；`apply` 做实例覆盖。
- **编辑器**：inspector 用 `TypeInfo` + 路径 + 元数据自动生成 UI（接 `prism_ui_inspector`）。
- **脚本**：函数反射 + 路径访问暴露给脚本层（`prism_script`）。
- **网络**：StableTypeId + diff/patch + 迁移做复制与存档兼容（接 `prism_replication`）。

---

## 16. 可观测性与诊断

- 类型注册表转储（已注册类型/TypeData/schema 版本），排查「类型未注册」错误。
- 序列化 schema 导出（供外部工具/文档）。
- 反射访问 trace（路径解析耗时、动态分配计数），定位工具侧性能热点。

---

## 17. 高级功能增补

- **泛型/const 泛型反射**：为单态化实例注册（如 `Vec<f32>` vs `Vec<u8>` 各自 TypeInfo）。
- **运行时定义类型**（flecs 形态）：脚本/数据驱动在运行时注册新反射类型（name+字段 layout），服务 ECS 动态组件。
- **反射 diff/merge**：算两值最小差异 patch（§8），服务回滚增量、协作编辑、版本合并。
- **校验与约束**：元数据范围/非空/枚举集在反序列化/编辑时校验，拒绝非法数据。
- **文档门控**：`documentation` 档携带文档字符串，发行版剥离省体积。
- **默认与克隆**：`ReflectDefault`/反射克隆，支持「重置为默认」「反射深拷贝」。

---

## 18. 性能工程

- **静态 TypeInfo + OnceLock**：类型信息编译期定、首次访问缓存，运行时零构造。
- **预编译路径**：`ParsedPath` 复用，避免热点字符串解析。
- **流式序列化**：不建中间树，直接遍历写字节。
- **索引访问**：字段按序号而非名字访问（名字→序号查一次缓存）。
- **边界使用原则**：反射用于工具/序列化/脚本边界，内层仿真循环仍走原生字段，不把反射放进热路径。

诚实边界：序列化吞吐、大场景内存、脚本调用开销需真实数据集压测，标注 PLANNED。

---

## 19. 易用性与 Bevy 迁移策略

对外 API 贴近 `bevy_reflect`：`#[derive(Reflect)]`、`Reflect`/`FromReflect`、`TypeRegistry::register`、`reflect_trait`、`DynamicStruct`、路径访问、`apply`。

迁移：
1. `prism_reflect::prelude` 近同名导出。
2. 为 `prism_math` 类型实现反射（替代 bevy_reflect 对 glam 的实现）。
3. ECS 动态组件/快照、场景、编辑器 inspector 底层切到本 crate。

---

## 20. crate 分层与模块布局

```
pkg/prism_reflect/
  src/
    reflect.rs                 # Reflect trait、ReflectRef/Mut
    kinds/                     # struct/enum/tuple/list/array/map/set/value/opaque
    type_info.rs               # TypeInfo、字段/变体描述、OnceLock 缓存
    registry.rs type_data.rs   # TypeRegistry、可插拔 TypeData
    dynamic.rs from_reflect.rs # Dynamic*、FromReflect
    apply.rs diff.rs           # 局部更新 apply/patch、diff/merge
    path.rs                    # 路径解析 ParsedPath、读写
    serde/                     # 反射驱动 Serializer/Deserializer、schema、migration（serialize 档）
    attributes.rs              # 元数据：范围/提示/分类/只读/隐藏/默认/单位
    func/                      # 函数反射、按名调用（functions 档）
    stable_id.rs               # StableTypeId、重命名映射
    impls/                     # 标准库 + prism_math 类型的反射实现
  features = ["std","serialize","functions","documentation","determinism"]

pkg/prism_reflect_macros/      # #[derive(Reflect)] + #[reflect_trait] + 属性解析
```

依赖：`prism_utils`，可选 `prism_math`/`prism_diagnostic`。**不碰任何 `bevy_*`。**

---

## 21. 契约、不变量与版本化

- **TypeInfo 静态且稳定**：同类型多次访问返回同一 `&'static`；字段顺序为序列化不变量。
- **StableTypeId 稳定**：同类型跨构建一致；重命名须经映射表，不得静默变更。
- **apply 语义**：只覆盖出现字段，类型不符报错不 panic。
- **迁移单调**：schema 版本只增；每版须有迁移或显式声明兼容。
- **版本化契约**：`TypeInfo`、`StableTypeId`、`ReflectSerialize`/`ReflectComponent`/`ReflectDefault` TypeData、`ParsedPath`、序列化格式/schema。

---

## 22. 路线图（M0–M6）与基准即规格

- **M0 反射核**：`Reflect` + struct/tuplestruct/value 种类 + `#[derive(Reflect)]` + TypeInfo/OnceLock → 字段遍历 + 单测。
- **M1 全种类 + 注册表**：enum/list/array/map/set、TypeRegistry + TypeData、标准库 + `prism_math` impls。
- **M2 动态 + FromReflect + apply**：Dynamic*、FromReflect、apply/patch、路径访问（ParsedPath）。
- **M3 序列化**：反射驱动 Serializer/Deserializer（接二进制 + RON）、StableTypeId。
- **M4 元数据 + schema**：属性元数据、schema 版本化 + 迁移链、校验。
- **M5 函数反射 + trait 对象**：函数注册/按名调用、`reflect_trait`、运行时定义类型、diff/merge。
- **M6 集成**：ECS 动态组件/快照、场景序列化、编辑器 inspector、脚本桥、网络复制接线；bevy_reflect 兼容 prelude。

**基准即规格**：序列化/反序列化吞吐、路径访问开销、FromReflect 构造成本、迁移正确性（旧版数据读回等价）、确定性双跑类型 ID/字段序一致。核心价值在 **M3（序列化）+ M4（schema 迁移）+ M5（函数反射）**。

---

## 23. 诚实边界与风险

- M0–M6 核心路线图**已全部落地并通过验证**：实现 + 单测（113 项 lib 测试全绿）+ 基准，`cargo clippy --all-targets` 零告警、`cargo test` 零失败。状态随代码演进；§24「AAA 高级功能增补」仍为 PLANNED，按本文优先级随消费方接线落地。
- **高风险项**：
  1. **StableTypeId 稳定性（M3）**：跨构建/平台/网络必须一致；哈希算法与类型路径规范化要定死，一旦发行就不能变（变则旧存档失效），须早定早冻结。
  2. **schema 版本化迁移（M4）**：存档/网络向后兼容是长期负债；迁移链须有测试覆盖「每旧版 → 当前」读回等价，漏一环就是玩家存档损坏。
  3. **动态分配**：Dynamic*/序列化易产生大量小分配；须竞技场/池化，否则大场景序列化内存与 GC 压力大。
  4. **函数反射安全（M5）**：按名调用 + 反射装箱是 `unsafe`/类型擦除要害，参数类型校验必须严格，错配会 UB；脚本边界尤其危险。
  5. **反射 vs 原生性能**：须纪律性地只在边界用反射；若误入仿真热路径会显著掉帧。
- **与既有文档关系**：本 crate 是 ECS 动态组件（§16.2）、快照（§16.5）、prefab（§16.3）与场景/编辑器/脚本/网络的共同类型基座；`apply`/`diff`/StableTypeId 的契约须与这些消费方一致。
- 所有 Prism crate 不含任何 Unreal Engine / Unity 源码或衍生代码；仅借鉴公开架构形态与经典数值。

---

## 24. AAA 高级功能增补（v0.2）

本章补齐顶级反射系统常被忽视、却在真实 AAA 项目里缺一不可的能力。均 feature/档位门控，默认不付成本；与前文静态+动态反射、反射驱动序列化、StableTypeId、schema 迁移、函数反射互补。

### 24.1 derive 期静态 TypeInfo（零运行时注册成本）

反射信息在 `#[derive(Reflect)]` 编译期生成为 `const`/`static` 的 `TypeInfo`，而非运行时逐字段 push 注册：

- 启动零注册开销，类型信息随二进制常驻，可 `const` 求值。
- 类型注册表仅做「StableTypeId → &'static TypeInfo」映射登记，不构造数据。
- 对标 Rust 零成本抽象理念,规避 Bevy 早期运行期注册的启动开销。

### 24.2 热路径访问器代码生成

动态字段访问若每次走「名字查哈希 → 偏移 → 类型擦除指针」会慢。为热路径生成直达访问器：

- derive 期为每字段生成 `offset_of` 常量 + 类型化 getter/setter，动态路径（如 `"transform.translation.x"`）首次解析后**缓存成偏移链**，后续 O(1) 直取。
- 序列化/网络/编辑器批量访问同一类型时复用缓存访问计划，避免重复解析。

### 24.3 二进制零拷贝序列化

文本（RON/JSON）用于编辑器/调试；运行时存档/网络走紧凑二进制：

- 定长 POD 字段**整块 memcpy**（零拷贝），变长字段（String/Vec）走长度前缀。
- 版本化二进制头 + StableTypeId，配合 §schema 迁移做跨版本读取。
- 与 `prism_tasks` 异步 I/O（tasks §24.5）协同做流式反/序列化。

### 24.4 部分 Patch 与反射 Diff

对两个同类型值做反射级 diff，产出**最小变更集**；或把部分 patch 应用到实例：

```rust
let patch = reflect_diff(&old, &new);   // 仅含变化字段
reflect_apply(&mut target, &patch);     // 部分覆盖
```

- 用途:热重载(改一个字段不重建整对象)、编辑器撤销/重做、网络增量(§24.5)、预制体覆盖(prefab override)。
- diff 递归进嵌套结构/集合,带路径定位。

### 24.5 反射驱动的字段级网络增量

复制系统（`prism_replication`）借反射自动算「哪些字段脏了」并只发增量：

- 字段级脏标记 + §24.4 diff → 每实体每帧只序列化变化字段。
- `#[reflect(replicate)]` 标注哪些字段参与复制，`#[reflect(no_replicate)]` 排除。
- 与 ECS 变更检测、`prism_tasks` 确定性并行协同,量化压缩交给 `prism_transform` §24.2 式编码。

### 24.6 脚本与编辑器属性桥

反射是脚本/编辑器通用访问的唯一门面：

- **编辑器**:自动生成属性面板(字段名/类型/范围/枚举下拉),无需为每类型手写 UI(对标 Unreal `UPROPERTY` 反射驱动 Details 面板)。
- **脚本**(Lua/WASM):经反射安全读写组件字段、调用反射注册的函数(接前文函数反射),脚本无需编译期知道 Rust 类型。
- 属性元数据(范围/步进/工具提示)由 §24.7 自定义特性携带。

### 24.7 自定义特性属性（Attributes）

derive 支持字段/类型级特性,驱动序列化、UI、校验:

| 特性 | 作用 |
|---|---|
| `#[reflect(rename = "...")]` | 序列化别名(兼容改名,接 schema 迁移) |
| `#[reflect(default = ...)]` | 缺字段时默认值(向后兼容) |
| `#[reflect(skip)]` | 不参与序列化/反射 |
| `#[reflect(clamp(0..=1))]` | 编辑器/反序列化范围钳制 |
| `#[reflect(tooltip = "...")]` | 编辑器提示 |

### 24.8 不可信反序列化安全边界

反射反序列化是**攻击面**（存档/网络来自不可信源）：

- **递归深度限幅** + 集合长度上限,防「十亿笑脸」式膨胀/栈溢出。
- 未知类型/字段按策略(忽略/拒绝/迁移)处理,不 panic。
- 分配上限 + 超时,防内存耗尽 DoS。
- 与 `prism_tasks` 背压(tasks §24.8)协同对流式输入限速。

### 24.9 诚实边界

本章全部为 PLANNED 设计目标,无代码。**24.1 静态 TypeInfo + 24.2 访问器代码生成**是性能地基,建议随 M0/M1 优先落地;24.3 二进制序列化、24.8 安全边界随 `prism_asset` 存档/加载落地;24.4 diff/patch 与 24.5 网络增量随 `prism_replication` 落地;24.6 属性桥、24.7 特性随编辑器(`prism_editor`)落地。反射的误用易成性能黑洞(热路径走动态查名),§24.2 缓存与 lint 必须兜底。所有 Prism crate 不含任何 Unreal Engine / Unity 源码或衍生代码;仅借鉴公开架构形态与经典数值。

