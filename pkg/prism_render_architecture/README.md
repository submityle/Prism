# Prism Render Architecture

This package contains backend-neutral contracts for Prism's Vulkan-first renderer. It deliberately contains no Unreal Engine source or derived shader code.

The package is split into foundation, scheduling, scene, virtualization, shading, lighting, presentation, and production-support modules. Feature implementations should depend on these contracts instead of directly coupling ECS data to Vulkan resources.

Current status: the first integration milestone is implemented. The retained GPU Scene has stable generational identity, transactional CPU mirroring, upload planning and completion-based reclamation. The frame graph compiler emits resource versions, hazards, queue batches and transient alias offsets. These APIs remain experimental while higher-level consumers are added.
