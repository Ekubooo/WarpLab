
import struct
import warp as wp


@wp.kernel
def fill_sphere(
    volume_id: wp.uint64,
    lo: wp.vec3i,
    radius: float,
    band: float,
):
    x, y, z = wp.tid()
    i = x + lo[0]
    j = y + lo[1]
    k = z + lo[2]

    # 整数 index 坐标 → 世界坐标。
    p = wp.volume_index_to_world(
        volume_id, wp.vec3(float(i), float(j), float(k))
    )

    sdf = wp.clamp(wp.length(p) - radius, -band, band)
    wp.volume_store_f(volume_id, i, j, k, sdf)


@wp.kernel
def query_values(
    volume_id: wp.uint64,
    center: wp.vec3i,
    result: wp.array(dtype=wp.float32),
):
    result[0] = wp.volume_lookup_f(
        volume_id, center[0], center[1], center[2]
    )

    # 球面上：世界位置 (10, 0, 0)，对应 index (100, 0, 0)。
    result[1] = wp.volume_lookup_f(volume_id, 100, 0, 0)

    # 远离已分配区域：返回背景值。
    result[2] = wp.volume_lookup_f(
        volume_id, 1000000, 1000000, 1000000
    )


def copy_bytes_to_host(volume, offset, count):
    """仅供元数据检查：复制少量网格字节到 CPU。"""
    src = volume.array()
    dst = wp.empty(count, dtype=wp.uint8, device="cpu")

    wp.copy(dst, src, src_offset=offset, count=count)
    wp.synchronize_device(volume.device)

    return dst.numpy().tobytes()


def read_metadata(volume):
    """
    Warp 高层接口未暴露全部所需字段，
    因此按 NanoVDB v32 的固定布局读取头部。
    这是检查工具，平时采样不需要这样做。
    """
    # GridData 672 字节，随后是 TreeData 64 字节。
    header = copy_bytes_to_host(volume, 0, 736)

    version = struct.unpack_from("<I", header, 16)[0]
    grid_type = struct.unpack_from("<I", header, 636)[0]
    if version >> 21 != 32 or grid_type != 1:
        raise ValueError("This helper expects a NanoVDB v32 float grid")

    name = header[40:296].split(b"\0", 1)[0].decode("utf-8")
    grid_class = struct.unpack_from("<I", header, 632)[0]
    world_bbox = struct.unpack_from("<6d", header, 560)
    active_count = struct.unpack_from("<Q", header, 672 + 56)[0]

    # Root 的偏移相对于 TreeData 起点。
    root_relative = struct.unpack_from("<q", header, 672 + 24)[0]
    root_offset = 672 + root_relative
    root = copy_bytes_to_host(volume, root_offset, 32)

    index_bbox = struct.unpack_from("<6i", root, 0)
    background = struct.unpack_from("<f", root, 28)[0]

    return name, grid_class, active_count, index_bbox, world_bbox, background


def main():
    wp.init()
    device = "cuda:0"

    radius = 10.0
    voxel_size = 0.1
    band = 3.0 * voxel_size

    # 边界对齐到 8 个体素，避免额外分配的槽位没有被填充。
    lo = (-104, -104, -104)
    hi = (103, 103, 103)
    shape = tuple(hi[a] - lo[a] + 1 for a in range(3))

    volume = wp.Volume.allocate(
        min=lo,
        max=hi,
        voxel_size=voxel_size,
        bg_value=band,
        translation=(0.0, 0.0, 0.0),
        points_in_world_space=False,
        device=device,
    )

    wp.launch(
        fill_sphere,
        dim=shape,
        inputs=[volume.id, wp.vec3i(*lo), radius, band],
        device=device,
    )
    wp.synchronize_device(device)

    name, cls, active_count, ibox, wbox, background = read_metadata(volume)

    bbox_lo = ibox[:3]
    bbox_hi = ibox[3:]
    center = tuple(
        bbox_lo[a] + (bbox_hi[a] - bbox_lo[a]) // 2
        for a in range(3)
    )

    result = wp.empty(3, dtype=wp.float32, device=device)
    wp.launch(
        query_values,
        dim=1,
        inputs=[volume.id, wp.vec3i(*center)],
        outputs=[result],
        device=device,
    )
    wp.synchronize_device(device)
    values = result.numpy()

    class_names = {
        0: "Unknown",
        1: "LevelSet",
        2: "FogVolume",
        7: "VoxelVolume",
    }

    print("Name:", name or "(unnamed)")
    print("Class:", class_names.get(cls, f"class #{cls}"))
    print("Active voxels:", active_count)
    print("Index bbox:", bbox_lo, bbox_hi)
    print("World bbox:", wbox[:3], wbox[3:])
    print("Background:", background)
    print("BBox center coordinate:", center)
    print("Value at bbox center:", float(values[0]))
    print("Value at sphere surface:", float(values[1]))
    print("Value far outside:", float(values[2]))


if __name__ == "__main__":
    main()