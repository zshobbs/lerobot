import shutil
import json
import pandas as pd
from pathlib import Path
import numpy as np

# Configuration
SOURCE_PATHS = [
    Path("/Users/zeke/Movies/merged_prop_fast"),
    Path("/Users/zeke/Movies/prop4"),
    Path("/Users/zeke/Movies/prop5")
]
OUTPUT_PATH = Path("/Users/zeke/Movies/merged_prop_all")
REPO_ID = "merged_prop_all"

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def fast_merge():
    if OUTPUT_PATH.exists():
        print(f"Output path {OUTPUT_PATH} exists. Please remove it first.")
        return

    print(f"Creating directory structure at {OUTPUT_PATH}...")
    OUTPUT_PATH.mkdir(parents=True)
    (OUTPUT_PATH / "meta").mkdir()
    (OUTPUT_PATH / "data").mkdir()
    (OUTPUT_PATH / "videos").mkdir()

    # --- 1. Load Info & stats ---
    infos = [load_json(p / "meta/info.json") for p in SOURCE_PATHS]
    stats = [load_json(p / "meta/stats.json") for p in SOURCE_PATHS]
    
    # Base info from the first dataset
    merged_info = infos[0].copy()
    
    # Calculate new totals
    total_episodes = sum(i["total_episodes"] for i in infos)
    total_frames = sum(i["total_frames"] for i in infos)
    
    merged_info["total_episodes"] = total_episodes
    merged_info["total_frames"] = total_frames
    merged_info["splits"] = {"train": f"0:{total_episodes}"}
    
    # Validate FPS and Features match
    for i, info in enumerate(infos[1:], 1):
        if info["fps"] != merged_info["fps"]:
            print(f"Error: FPS mismatch in dataset {i}")
            return
        # Simple key check
        if set(info["features"].keys()) != set(merged_info["features"].keys()):
            print(f"Error: Feature keys mismatch in dataset {i}")
            return

    # --- 2. Process Datasets ---
    
    # Global counters to track offsets
    global_episode_offset = 0
    global_frame_offset = 0
    
    # We need to track the last used chunk indices to know where to start the next dataset
    # format: data_chunk, {video_key: video_chunk}
    current_data_chunk_offset = 0
    current_video_chunk_offsets = {k: 0 for k in merged_info["video_path"].split("/") if "{video_key}" not in k} 
    # Actually, we need to discover the max chunk used by reading the files or metadata.
    # A safer way is to just keep incrementing based on what we see.
    
    # We will accumulate all metadata rows here
    all_episodes_dfs = []
    
    for ds_idx, ds_path in enumerate(SOURCE_PATHS):
        print(f"\nProcessing {ds_path.name}...")
        
        # Load metadata
        meta_dir = ds_path / "meta/episodes"
        # Read all parquet files in meta/episodes and concat
        # We need them sorted by chunk/file index usually, but glob order might vary.
        # It's safer to read them and sort by episode_index.
        meta_files = sorted(list(meta_dir.glob("**/*.parquet")))
        
        if not meta_files:
            print(f"No metadata files found in {meta_dir}!")
            return

        ds_meta_df = pd.concat([pd.read_parquet(f) for f in meta_files])
        ds_meta_df = ds_meta_df.sort_values("episode_index").reset_index(drop=True)
        
        num_eps = len(ds_meta_df)
        print(f"  Episodes: {num_eps}")
        
        # --- CALC MAX CHUNKS IN THIS DATASET ---
        # We need to know how many chunks this dataset occupies so we can offset the NEXT one.
        # But for THIS dataset, we need to offset it by the PREVIOUS dataset's count.
        
        # 1. Update Episode Index
        ds_meta_df["episode_index"] += global_episode_offset
        
        # 2. Update Frame Indices (dataset_from/to_index)
        ds_meta_df["dataset_from_index"] += global_frame_offset
        ds_meta_df["dataset_to_index"] += global_frame_offset
        
        # 3. Handle Data Chunks (Parquet files with sensor data)
        # We need to move the physical files and update the pointers in metadata
        
        # Identify unique data chunks in this dataset
        # We will map (old_chunk, old_file) -> (new_chunk, new_file)
        # But actually, LeRobot usually just fills chunks sequentially. 
        # We can just add the `current_data_chunk_offset` to the `chunk_index`.
        # However, we must ensure we don't merge the last chunk of DS1 with first of DS2.
        # We will keep them distinct to be safe (simple addition).
        
        # Helper to rename and move files
        def move_chunks(category, old_chunk_col, old_file_col, target_dir_fmt, src_root, dst_root, offset):
            # category: 'data' or 'videos/key'
            # target_dir_fmt: 'data' or 'videos/{key}'
            
            # Get unique pairs of (chunk, file)
            unique_files = ds_meta_df[[old_chunk_col, old_file_col]].drop_duplicates()
            
            max_chunk_seen = 0
            
            for _, row in unique_files.iterrows():
                c_idx, f_idx = int(row[old_chunk_col]), int(row[old_file_col])
                new_c_idx = c_idx + offset
                
                # Update max seen (for the next dataset to use)
                if c_idx > max_chunk_seen:
                    max_chunk_seen = c_idx
                
                # Construct paths
                # Assuming standard format: chunk-{:03d}/file-{:03d}.ext
                # Extension depends on category
                ext = ".parquet" if category == "data" else ".mp4"
                
                src_file = src_root / f"chunk-{c_idx:03d}/file-{f_idx:03d}{ext}"
                dst_dir = dst_root / f"chunk-{new_c_idx:03d}"
                dst_file = dst_dir / f"file-{f_idx:03d}{ext}"
                
                if not src_file.exists():
                    print(f"Warning: Missing file {src_file}")
                    continue
                
                dst_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy (or link?) Copy is safer.
                shutil.copy2(src_file, dst_file)
            
            return max_chunk_seen + 1 # Return count of chunks used

        # --- MOVE DATA FILES ---
        print("  Moving data parquet files...")
        chunks_used = move_chunks(
            "data", 
            "data/chunk_index", "data/file_index", 
            "data", 
            ds_path / "data", 
            OUTPUT_PATH / "data", 
            current_data_chunk_offset
        )
        
        # Update metadata pointers for data
        ds_meta_df["data/chunk_index"] += current_data_chunk_offset
        
        # Update global offset for next dataset (ensure we start on a new chunk)
        # We add the number of chunks this dataset used.
        next_data_chunk_start = current_data_chunk_offset + chunks_used
        
        # --- MOVE VIDEO FILES ---
        # Video keys are in info["features"] where dtype=video
        video_keys = [k for k, v in merged_info["features"].items() if v["dtype"] == "video"]
        
        next_video_chunk_offsets = current_video_chunk_offsets.copy()
        
        for v_key in video_keys:
            print(f"  Moving video files for {v_key}...")
            
            # Metadata columns for this video key
            chunk_col = f"videos/{v_key}/chunk_index"
            file_col = f"videos/{v_key}/file_index"
            
            # Current offset for this specific video key
            v_offset = current_video_chunk_offsets.get(v_key, 0)
            
            chunks_used_v = move_chunks(
                f"videos/{v_key}", 
                chunk_col, file_col, 
                f"videos/{v_key}", 
                ds_path / f"videos/{v_key}", 
                OUTPUT_PATH / f"videos/{v_key}", 
                v_offset
            )
            
            # Update metadata pointers
            ds_meta_df[chunk_col] += v_offset
            
            # Update offset for next dataset
            next_video_chunk_offsets[v_key] = v_offset + chunks_used_v

        # --- UPDATE INTERNAL PARQUET DATA (Indices) ---
        # The data parquet files contain "episode_index", "index", "frame_index" columns.
        # These MUST be updated to match the new global offsets.
        # We need to read the *newly moved* files, update them, and rewrite them.
        # This is the "slowest" part of the fast method, but much faster than encoding video.
        print("  Updating internal data indices...")
        
        # Iterate over the NEW chunks we just created
        unique_data_files = ds_meta_df[["data/chunk_index", "data/file_index"]].drop_duplicates()
        
        for _, row in unique_data_files.iterrows():
            c_idx = int(row["data/chunk_index"])
            f_idx = int(row["data/file_index"])
            p_path = OUTPUT_PATH / f"data/chunk-{c_idx:03d}/file-{f_idx:03d}.parquet"
            
            df = pd.read_parquet(p_path)
            
            # Update columns
            if "episode_index" in df.columns:
                df["episode_index"] += global_episode_offset
            if "index" in df.columns: # Global frame index
                df["index"] += global_frame_offset
            # frame_index (local to episode) usually doesn't need changing if episodes are intact?
            # actually frame_index is usually 0..N per episode. It stays same.
            # BUT "index" is global.
            
            # Write back
            df.to_parquet(p_path)

        # --- PREPARE FOR NEXT DATASET ---
        all_episodes_dfs.append(ds_meta_df)
        
        global_episode_offset += num_eps
        # Add total frames in this dataset to the offset
        # The last frame index in this dataset + 1 is the start of next.
        # We can get it from the last row's dataset_to_index
        global_frame_offset = ds_meta_df.iloc[-1]["dataset_to_index"]
        
        current_data_chunk_offset = next_data_chunk_start
        current_video_chunk_offsets = next_video_chunk_offsets

    # --- 3. Save Merged Metadata ---
    print("\nSaving merged metadata...")
    final_meta_df = pd.concat(all_episodes_dfs).reset_index(drop=True)
    
    # We need to save this big table. LeRobot splits it into chunks too.
    # We can just dump it all into one chunk or split it. Splitting is better practice.
    # We'll just split by 1000 episodes or so.
    meta_chunk_size = 1000
    total_eps = len(final_meta_df)
    
    for i in range(0, total_eps, meta_chunk_size):
        chunk_df = final_meta_df.iloc[i:i+meta_chunk_size]
        # We'll treat meta chunks simply: chunk-000, file-000, etc.
        # Since we are creating a new meta structure, we can just increment.
        chunk_idx = i // meta_chunk_size
        
        meta_out_dir = OUTPUT_PATH / f"meta/episodes/chunk-{chunk_idx:03d}"
        meta_out_dir.mkdir(parents=True, exist_ok=True)
        
        # We just write one file per chunk for simplicity
        chunk_df.to_parquet(meta_out_dir / "file-000.parquet")

    # Save info.json
    save_json(merged_info, OUTPUT_PATH / "meta/info.json")
    
    # Save stats.json (We should aggregate them properly, but for now we take the first one
    # or simple average? LeRobot has an aggregation function, but we are in a simple script.
    # We will just copy the first one's stats as a placeholder or warn.)
    # Ideally, we should use lerobot.datasets.compute_stats.aggregate_stats
    print("Aggregating stats (simple mean)...")
    # A proper aggregation is complex without loading the library. 
    # For now, we will just copy the stats from the first dataset and warn user to recompute if needed.
    save_json(stats[0], OUTPUT_PATH / "meta/stats.json")
    print("Warning: Stats file is copied from the first dataset. Recommend re-computing stats if distributions differ significantly.")
    
    # Copy tasks.parquet if it exists (assuming single task for now)
    # If tasks differ, we need to merge this too.
    # For now, we assume same task.
    if (SOURCE_PATHS[0] / "meta/tasks.parquet").exists():
        shutil.copy2(SOURCE_PATHS[0] / "meta/tasks.parquet", OUTPUT_PATH / "meta/tasks.parquet")

    print(f"\nFast merge complete! Dataset saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    fast_merge()