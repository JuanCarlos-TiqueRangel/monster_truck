from pathlib import Path

# Folder containing all the optitrack folders
parent_folder = Path("/ros_ws/src/rc_car/rc_car/real_car/wheelie/results")

# Find all directories starting with "optitrack_"
folders = sorted([
    f for f in parent_folder.iterdir()
    if f.is_dir() and f.name.startswith("optitrack_")
])

# First rename to temporary names to avoid name collisions
temp_folders = []

for i, folder in enumerate(folders, start=1):
    temp_name = parent_folder / f"__temp_optitrack_{i:04d}"
    folder.rename(temp_name)
    temp_folders.append(temp_name)

# Rename to final names
for i, folder in enumerate(temp_folders, start=1):
    new_name = parent_folder / f"optitrack_{i:02d}"
    folder.rename(new_name)
    print(f"{folder.name} -> {new_name.name}")

print(f"\nDone! Renamed {len(folders)} folders.")
