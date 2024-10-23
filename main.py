from fastapi import FastAPI, File, UploadFile, Request, WebSocket
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import io
from PIL import Image
import os
import uuid
import json
import asyncio
from typing import List

# Import our modules
from modules.image_processor import YOLOImageProcessor
from modules.models import ArrowDetectionResult
from modules.csv_logger import CSVLogger  # Add this import

app = FastAPI()

# Create templates directory
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Mount static directories
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/temp_images", StaticFiles(directory="temp_images"),
          name="temp_images")

# Store active WebSocket connections
active_connections: List[WebSocket] = []

# Current processing state
current_result = None
direction_event = asyncio.Event()

# Initialize the image processor and CSV logger
image_processor = YOLOImageProcessor("best.pt")
csv_logger = CSVLogger("data.csv")


@app.on_event("startup")
async def startup_event():
    # Create temp_images directory if it doesn't exist
    os.makedirs("temp_images", exist_ok=True)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except:
        active_connections.remove(websocket)


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


@app.post("/process_images/")
async def process_image(image: UploadFile = File(...)):
    global current_result, direction_event

    # Reset event for new request
    direction_event = asyncio.Event()

    # Process image
    image_content = await image.read()
    img = Image.open(io.BytesIO(image_content))

    # Get results from YOLO processing
    result, plotted_image = await image_processor.process_image(img)

    # Save plotted image with .jpg extension
    image_id = f"{str(uuid.uuid4())}.jpg"
    plotted_image.save(f"temp_images/{image_id}")

    # Store current result
    current_result = result

    # Notify WebSocket clients about new image
    for connection in active_connections:
        try:
            await connection.send_text(json.dumps({
                "image_id": image_id,
                "arrow_count": result.arrow_count,
                "arrow_label1": result.arrow_label1 if result.arrow_count > 0 else None,
                "arrow_label2": result.arrow_label2 if result.arrow_count > 1 else None,
                "arrow_label3": result.arrow_label3 if result.arrow_count > 2 else None,
                "confidence1": result.confidence1 if result.arrow_count > 0 else None,
                "confidence2": result.confidence2 if result.arrow_count > 1 else None,
                "confidence3": result.confidence3 if result.arrow_count > 2 else None,
                "status": "waiting_direction"
            }))
        except:
            active_connections.remove(connection)

    # Wait for direction to be set
    await direction_event.wait()

    print(current_result)

    # Return the complete result with direction
    return current_result


@app.post("/update_direction")
async def update_direction(direction: str):
    global current_result

    if current_result is None:
        return {"error": "No active detection"}

    # Update direction in the current result
    current_result.direction = direction

    # Log the result to CSV
    csv_logger.log_result(current_result)

    # Set the event to unblock the waiting request
    direction_event.set()

    return {"status": "success", "direction": direction}
