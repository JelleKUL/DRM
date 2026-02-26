def txt_pcd_to_ply(txt_path, ply_path):
    with open(txt_path, "r") as f:
        rows = [line.strip().split() for line in f if line.strip()]

    with open(ply_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(rows)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for r in rows:
            if len(r) != 9:
                raise ValueError(f"Expected 9 values per row, got {len(r)}")

            f.write(" ".join(r) + "\n")