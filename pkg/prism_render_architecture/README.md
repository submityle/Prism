# Prism Render Architecture

This package contains backend-neutral contracts for Prism's Vulkan-first renderer. It deliberately contains no Unreal Engine source or derived shader code.

The package is split into foundation, scheduling, scene, virtualization, shading, lighting, presentation, and production-support modules. Feature implementations should depend on these contracts instead of directly coupling ECS data to Vulkan resources.

Current status: architecture scaffold. Types may change until the GPU Scene, ABI, and frame graph reach their first integration milestone.

