import argparse
import logging
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import delete_episodes
from lerobot.utils.constants import HF_LEROBOT_HOME

app = FastAPI(title="LeRobot Dashboard")

# Global state
DATASET: Optional[LeRobotDataset] = None
DATASET_ROOT: Optional[Path] = None
REPO_ID: Optional[str] = None

# Templates
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

class DeleteRequest(BaseModel):
    episode_indices: List[int]

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    if not DATASET:
        return templates.TemplateResponse("error.html", {"request": request, "error": "No dataset loaded."})
    
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "repo_id": REPO_ID,
        "total_episodes": DATASET.meta.total_episodes,
        "total_frames": DATASET.meta.total_frames,
        "fps": DATASET.fps
    })

@app.get("/api/episodes")
async def get_episodes(limit: int = 50, offset: int = 0):
    if not DATASET:
        raise HTTPException(status_code=404, detail="Dataset not loaded")
    
    episodes = []
    # Using metadata directly is faster than loading the whole dataset
    meta_episodes = DATASET.meta.episodes
    
    end = min(offset + limit, len(meta_episodes))
    
    for i in range(offset, end):
        ep = meta_episodes[i]
        episodes.append({
            "index": ep["episode_index"],
            "length": ep["length"],
            "tasks": ep.get("tasks", ["Unknown"]),
            "success": ep.get("stats", {}).get("success", None) # If success metric exists
        })
        
    return {"episodes": episodes, "total": len(meta_episodes)}

@app.get("/api/episodes/{index}")
async def get_episode_details(index: int):
    if not DATASET:
        raise HTTPException(status_code=404, detail="Dataset not loaded")
    
    if index < 0 or index >= DATASET.meta.total_episodes:
        raise HTTPException(status_code=404, detail="Episode not found")

    ep = DATASET.meta.episodes[index]
    
    # Get video keys
    video_keys = DATASET.meta.video_keys
    
    return {
        "index": ep["episode_index"],
        "length": ep["length"],
        "tasks": ep.get("tasks", []),
        "video_keys": video_keys,
        "fps": DATASET.fps
    }

@app.get("/api/episodes/{index}/video/{key}")
async def get_episode_video(index: int, key: str):
    if not DATASET:
        raise HTTPException(status_code=404, detail="Dataset not loaded")
        
    try:
        # Resolve the video path using LeRobotDataset logic
        # LeRobotDataset.meta.get_video_file_path returns relative path
        video_rel_path = DATASET.meta.get_video_file_path(index, key)
        video_path = DATASET.root / video_rel_path
        
        if not video_path.exists():
            raise HTTPException(status_code=404, detail="Video file not found")
            
        return FileResponse(video_path, media_type="video/mp4")
    except Exception as e:
        logging.error(f"Error serving video: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/episodes/delete")
async def delete_episodes_api(req: DeleteRequest):
    global DATASET
    if not DATASET:
        raise HTTPException(status_code=404, detail="Dataset not loaded")
        
    try:
        # We need to reload the dataset after deletion to reflect changes
        # For now, we perform an in-place deletion if possible, or save to a new path
        # Simplest approach for "cleaning": overwrite the current dataset (risky but requested)
        # OR: create a "_cleaned" version.
        
        # NOTE: In-place editing is tricky with opened file handles. 
        # Ideally we use `lerobot.datasets.dataset_tools.delete_episodes`
        # But that function writes to a NEW directory by default.
        
        # For this prototype, let's just log it. Real implementation needs careful path handling.
        logging.info(f"Request to delete: {req.episode_indices}")
        
        # Placeholder for actual deletion logic
        return {"status": "success", "message": "Deletion logic to be implemented (requires dataset reload)"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, required=True, help="Dataset repo id")
    parser.add_argument("--root", type=Path, default=None, help="Dataset root")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    global DATASET, DATASET_ROOT, REPO_ID
    REPO_ID = args.repo_id
    DATASET_ROOT = args.root
    
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Loading dataset {REPO_ID}...")
    
    try:
        DATASET = LeRobotDataset(REPO_ID, root=DATASET_ROOT)
        logging.info("Dataset loaded.")
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        # We still start the server so the user sees the error page
    
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
