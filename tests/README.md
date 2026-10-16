# 测试脚本

仓库测试集中在本目录，按功能分组。仿真、渲染、场景生成和烘焙功能模块仍位于 `demos/`。

```text
tests/fluid/
├── boundary/          # 地形与容器生成、网格检查、SDF 烘焙与独立 CLI
├── pbf/               # PBF 配置、控制、SDF 文件名选择与资源接入
├── mmpbf/             # MMPBF 公式与行为
└── pbf2waycoupling/   # 双向耦合、墙体钳制、渲染与接触参考实现
```

在仓库根目录、Conda `warplab` 环境中运行全部测试：

```powershell
conda run -n warplab --no-capture-output python -m unittest discover -s tests -t . -v
```

仅运行 PBF 和场景边界测试：

```powershell
conda run -n warplab --no-capture-output python -m unittest discover -s tests/fluid/pbf -t . -v
conda run -n warplab --no-capture-output python -m unittest discover -s tests/fluid/boundary -t . -v
```

运行单个模块：

```powershell
conda run -n warplab --no-capture-output python -m unittest tests.fluid.boundary.test_sdf_baker -v
conda run -n warplab --no-capture-output python -m unittest tests.fluid.boundary.test_bake_sdf -v
```

NanoVDB 测试需要 CUDA，无 CUDA 时跳过；双向耦合的渲染测试需要 CUDA 和可用的 OpenGL 上下文。渲染测试的截图保存在 `outputs/pbf2way/render-check/`。`test_contact_reference.py` 是供等价性测试和接触性能基准调用的参考 kernel。
