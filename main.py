from fastapi import FastAPI, File, UploadFile
import uvicorn
from recognize import process
from PIL import Image
import io

app = FastAPI()


@app.post("/process_images/")
async def process_image(image: UploadFile = File(...)):
    image_content = await image.read()
    img = Image.open(io.BytesIO(image_content))
    direction = process(img)
    return {"direction": direction}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
