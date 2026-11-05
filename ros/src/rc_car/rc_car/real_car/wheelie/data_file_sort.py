from pathlib import Path

parent_folder = Path(
    "/ros_ws/src/rc_car/rc_car/real_car/wheelie/data_rls_collection"
)

files = sorted([
    f for f in parent_folder.iterdir()
    # if f.is_file() and f.name.startswith("rls_identification_")
    if f.is_file() and f.name.startswith("rls_id_")
])

temp_files = []

for i, file in enumerate(files, start=1):
    extension = file.suffix
    temp_name = parent_folder / f"__temp_rls_identification_{i:04d}{extension}"

    if temp_name.exists():
        raise FileExistsError(f"{temp_name} already exists")

    file.rename(temp_name)
    temp_files.append(temp_name)

for i, file in enumerate(temp_files, start=1):
    extension = file.suffix
    new_name = parent_folder / f"rls_id_{i:03d}{extension}"

    if new_name.exists():
        raise FileExistsError(f"{new_name} already exists")

    file.rename(new_name)
    print(f"{file.name} -> {new_name.name}")

print(f"\nDone! Renamed {len(files)} files.")