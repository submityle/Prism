# Shader executable 验证链

## 实现

`VerifiedShaderArtifact::from_spirv` 使用 Naga 30 SPIR-V 前端的严格能力解析，再启用全部验证标记检查整个模块。只有验证通过且 manifest 入口唯一存在，才产生私有字段的证明对象。不存在 unchecked 构造或可直接反序列化的证明；载入字节需要重新验证。

证明保留不可变 `Arc<[u32]>`，克隆不复制 shader 字节。内容身份采用带域分隔的 BLAKE3，对每个 word 的规范 little-endian 编码计算，与文件名、源 revision、包名称分离。

`ShaderPackageRegistry::register_verified` 原子地注册证明和 manifest；ABI 验证仍需独立通过。原有 metadata 注册仍用于包发现，但不能作为可执行响应的依据。`ResponseRegistry` 的注册和解析都检查 Ready、完整 manifest、ABI 和 executable digest；因此同名同 revision 的代码替换也不会静默混用。

## 验收

真实最小 compute SPIR-V fixture覆盖完整解析和验证；同时检查 metadata Ready 伪造、有效代码替换、结构可反射但类型错误的函数、proof 克隆共享字节。仅名称反射不再是响应注册的充分条件。

Naga 验证证明类型与控制流有效，不证明响应数学满足 OpenPBR，也不证明所有 GPU 都支持模块。设备特性与限额由实际 pipeline 创建验证，响应数学由对应对拍套件验证；这三个证据不互相替代。
