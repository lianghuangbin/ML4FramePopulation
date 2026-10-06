import os
import re
import csv
import shutil


def parse_gm_filename(filename):
    """
    解析类似：
    GM_ACC_OESMD_19760506_000055_1_000X_P034985_00001-03654_010_1021_FRIULI_0020_MW65_023_000_MS.txt
    尽量提取出常用字段。
    若某些字段格式略有不同，可后续微调。
    """
    name = os.path.splitext(filename)[0]
    parts = name.split("_")

    # 先给默认值，避免报错
    info = {
        "original_filename": filename,
        "event_date_raw": "",
        "waveform_id": "",
        "component": "",
        "pga_mps2": "",
        "start_step": "",
        "end_step": "",
        "dt": "",
        "duration_s": "",
        "vs30_mps": "",
        "event_site": "",
        "station_id": "",
        "Mw": "",
        "epicentral_distance_km": "",
        "depth_km": "",
        "shock_type": "",
    }

    # 基于你给的示例格式：
    # 0 GM
    # 1 ACC
    # 2 OESMD
    # 3 19760506
    # 4 000055
    # 5 1
    # 6 000X
    # 7 P034985
    # 8 00001-03654
    # 9 010
    # 10 1021
    # 11 FRIULI
    # 12 0020
    # 13 MW65
    # 14 023
    # 15 000
    # 16 MS

    if len(parts) >= 17:
        info["event_date_raw"] = parts[3]
        info["waveform_id"] = str(int(parts[4])) if parts[4].isdigit() else parts[4]

        comp_raw = parts[6]
        info["component"] = comp_raw[-1] if len(comp_raw) > 0 else comp_raw

        pga_raw = parts[7]
        # P034985 -> 3.4985 m/s²
        info["pga_mps2"] = float(pga_raw[1:]) / 10000.0

        range_raw = parts[8]
        if "-" in range_raw:
            st, ed = range_raw.split("-")
            info["start_step"] = int(st)
            info["end_step"] = int(ed)

        dt_raw = parts[9]
        # 010 -> 0.01 s, 020 -> 0.02 s
        if dt_raw.isdigit():
            info["dt"] = float(dt_raw) / 1000.0

        if info["start_step"] != "" and info["end_step"] != "" and info["dt"] != "":
            npts = int(info["end_step"]) - int(info["start_step"]) + 1
            info["duration_s"] = npts * float(info["dt"])

        if parts[10].isdigit():
            info["vs30_mps"] = int(parts[10])

        info["event_site"] = parts[11]

        if parts[12].isdigit():
            info["station_id"] = int(parts[12])

        mw_raw = parts[13]
        # MW65 -> 6.5
        if mw_raw.upper().startswith("MW") and mw_raw[2:].isdigit():
            info["Mw"] = float(mw_raw[2:]) / 10.0

        if parts[14].isdigit():
            info["epicentral_distance_km"] = int(parts[14])

        if parts[15].isdigit():
            info["depth_km"] = int(parts[15])

        info["shock_type"] = parts[16]

    return info


def prepare_ground_motions(
    source_dir="02_ground_motions/original",
    raw_dir="02_ground_motions/raw",
    metadata_csv="02_ground_motions/gm_metadata.csv"
):
    os.makedirs(raw_dir, exist_ok=True)

    files = sorted([
        f for f in os.listdir(source_dir)
        if os.path.isfile(os.path.join(source_dir, f)) and f.lower().endswith(".txt")
    ])

    rows = []

    for i, fname in enumerate(files, start=1):
        gm_id = f"gm_{i:04d}"
        new_filename = f"{gm_id}.txt"

        src = os.path.join(source_dir, fname)
        dst = os.path.join(raw_dir, new_filename)

        # 复制并重命名
        shutil.copy2(src, dst)

        # 解析文件名信息
        info = parse_gm_filename(fname)

        row = {
            "gm_id": gm_id,
            "new_filename": new_filename,
            "new_filepath": os.path.join("raw", new_filename),
            "original_filename": fname,
            "event_date_raw": info["event_date_raw"],
            "waveform_id": info["waveform_id"],
            "component": info["component"],
            "dt": info["dt"],
            "duration_s": info["duration_s"],
            "pga_mps2": info["pga_mps2"],
            "vs30_mps": info["vs30_mps"],
            "event_site": info["event_site"],
            "station_id": info["station_id"],
            "Mw": info["Mw"],
            "epicentral_distance_km": info["epicentral_distance_km"],
            "depth_km": info["depth_km"],
            "shock_type": info["shock_type"],
        }
        rows.append(row)

        print(f"Prepared {gm_id} <- {fname}")

    # 写 metadata csv
    if rows:
        fieldnames = list(rows[0].keys())
        with open(metadata_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nSaved metadata: {metadata_csv}")
    print(f"Total motions prepared: {len(rows)}")


if __name__ == "__main__":
    prepare_ground_motions(
        source_dir="02_ground_motions/original",
        raw_dir="02_ground_motions/raw",
        metadata_csv="02_ground_motions/gm_metadata.csv"
    )