# supreme pbf impl 
```
OpenGL 主循环
└─ PlaybackScheduler.advance()
   └─ Example.step()                         # 一个完整物理步
      └─ ScopedTimer("sub-step")
         ├─ Prologue
         │  ├─ 读取本步 dt
         │  ├─ clear_accelerations           # 重力及外力初始化
         │  ├─ predict_positions             # 只推进 Active 粒子
         │  ├─ fluid_grid.build
         │  ├─ boundary_grid / 边界模型更新
         │  ├─ fluid-fluid 邻域查询
         │  ├─ fluid-boundary 邻域查询
         │  └─ 可选的邻域数据预计算
         │
         ├─ Solver：PBF pressureSolve
         │  └─ while 未收敛且未达到最大迭代数
         │     ├─ compute_density
         │     ├─ compute_lambda
         │     ├─ compute_delta_x
         │     │  ├─ 流体—流体约束
         │     │  └─ 流体—边界约束
         │     ├─ apply_delta_x               # 只修正 Active 粒子
         │     └─ density_error reduction     # 判断是否继续迭代
         │
         └─ Epilogue
            ├─ reconstruct_velocity           # 一阶或二阶
            ├─ recompute_density
            ├─ compute_non_pressure_forces
            │  ├─ viscosity                  # 选择一种黏滞模型
            │  ├─ vorticity                  # 可选
            │  ├─ surface_tension             # 可选
            │  ├─ drag / elasticity           # 可选模块
            │  └─ boundary viscosity / reaction
            ├─ integrate_non_pressure_forces
            ├─ CFL reduction
            ├─ update_next_dt
            │
            ├─ EmitterSystem.step
            │  ├─ AnimatedByEmitter → Active
            │  ├─ 回收越界粒子（可选）
            │  └─ 遍历所有 emitter
            │     ├─ 推进喷口区域内的旧粒子
            │     ├─ 判断本步是否应发射
            │     ├─ 激活新的一层粒子
            │     ├─ 初始化各求解模块的数据
            │     └─ 扩大邻域搜索活跃范围
            │
            ├─ animateParticles              # 外部动画粒子/边界
            └─ sim_time += dt
```