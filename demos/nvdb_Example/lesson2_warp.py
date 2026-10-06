import struct
import numpy as np
import warp as wp


@wp.kernel
def fill_sphere(
    volume_id: wp.uint64,
    lo: wp.vec3i,
    radius: float,
    band: float,
):
    # 三维启动：每个线程负责一个体素。
    x, y, z = wp.tid()
    i = x + lo[0]
    j = y + lo[1]
    k = z + lo[2]

    # 在该体素的世界位置计算球体 SDF。
    world_point = wp.volume_index_to_world(
        volume_id,
        wp.vec3(float(i), float(j), float(k)),
    )

    # 截断深处的距离值，但不改变 active 状态。
    sdf = wp.clamp(wp.length(world_point) - radius, -band, band)
    wp.volume_store_f(volume_id, i, j, k, sdf)


@wp.kernel
def sample_points(
    volume_id: wp.uint64,
    world_points: wp.array(dtype=wp.vec3),
    samples: wp.array(dtype=wp.float32),
):
    # 一维启动：每个线程负责一个随机点。
    tid = wp.tid()

    # 采样函数需要 index-space 坐标，且允许小数。
    index_point = wp.volume_world_to_index(
        volume_id, world_points[tid]
    )

    # LINEAR 表示三线性插值。
    # 每个线程只写自己的 samples[tid]，没有写入冲突。
    samples[tid] = wp.volume_sample_f(
        volume_id, index_point, wp.Volume.LINEAR
    )


def get_world_bbox(volume):
    """
    延续练习 1 的元数据读取方式。

    Warp 高层接口未直接暴露 worldBBox；
    NanoVDB v32 的 GridData 在偏移 560 处保存六个 double。
    这里只复制头部的 608 字节，不复制整个网格。
    """
    header = wp.empty(608, dtype=wp.uint8, device="cpu")
    wp.copy(header, volume.array(), count=608)

    # CPU 接下来要读取复制结果，因此先等待复制完成。
    wp.synchronize_device(volume.device)
    data = header.numpy().tobytes()

    version = struct.unpack_from("<I", data, 16)[0]
    if version >> 21 != 32:
        raise ValueError("This helper expects NanoVDB v32")

    bbox = struct.unpack_from("<6d", data, 560)
    return np.array(bbox[:3]), np.array(bbox[3:])


def main():
    wp.init()
    device = "cuda:0"

    sample_count = 1000
    radius = 10.0
    voxel_size = 0.1
    band = 3.0 * voxel_size

    # 与练习 1 相同：完整覆盖且对齐到 8 格块边界。
    lo = (-104, -104, -104)
    hi = (103, 103, 103)
    shape = tuple(hi[a] - lo[a] + 1 for a in range(3))

    # 1. 建立拓扑并分配 float 体素存储。
    volume = wp.Volume.allocate(
        min=lo,
        max=hi,
        voxel_size=voxel_size,
        bg_value=band,
        translation=(0.0, 0.0, 0.0),
        points_in_world_space=False,
        device=device,
    )

    # 2. 在 GPU 上填充截断后的球体 SDF。
    wp.launch(
        fill_sphere,
        dim=shape,
        inputs=[volume.id, wp.vec3i(*lo), radius, band],
        device=device,
    )

    bbox_lo, bbox_hi = get_world_bbox(volume)

    # 3. 在 CPU 上生成均匀分布的随机世界坐标。
    # 每行是一个点，数组形状为 (1000, 3)。
    rng = np.random.default_rng(42)
    points_host = rng.uniform(
        low=bbox_lo,
        high=bbox_hi,
        size=(sample_count, 3),
    ).astype(np.float32)

    # 转成 GPU Warp 数组。
    # dtype=wp.vec3 将每行三个 float 作为一个向量元素。
    points_device = wp.array(
        points_host,
        dtype=wp.vec3,
        device=device,
    )
    samples = wp.empty(
        sample_count,
        dtype=wp.float32,
        device=device,
    )

    # 4. 每个线程采样一个点。
    wp.launch(
        sample_points,
        dim=sample_count,
        inputs=[volume.id, points_device],
        outputs=[samples],
        device=device,
    )

    # 5. CPU 要计算统计量，因此取回采样结果。
    # 本例使用默认执行方式；numpy() 会等待必要的 GPU 工作。
    values = samples.numpy()

    # 体素值是 float32，统计累加使用 float64。
    # ddof=0 对应 C++ 版本中的“除以 N”。
    mean = values.mean(dtype=np.float64)
    stddev = values.std(dtype=np.float64, ddof=0)

    print("Sample count:", sample_count)
    print("World bbox min:", bbox_lo)
    print("World bbox max:", bbox_hi)
    print(f"Mean: {mean:.6f}")
    print(f"Std:  {stddev:.6f}")


if __name__ == "__main__":
    main()