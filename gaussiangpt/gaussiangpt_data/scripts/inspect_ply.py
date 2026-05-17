from plyfile import PlyData
from pathlib import Path
import sys

ply_path = Path(sys.argv[1])

print("Reading:", ply_path)

ply = PlyData.read(str(ply_path))

print("\nPLY elements:")
for elem in ply.elements:
    print(" ", elem.name, "count =", elem.count)

v = ply["vertex"].data

print("\nNum gaussians:", len(v))

print("\nFields:")
for name in v.dtype.names:
    print(" ", name)

print("\nFirst gaussian values:")
for name in v.dtype.names:
    print(name, v[name][0])