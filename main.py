from fastapi import FastAPI, File, UploadFile
from typing import List
import uvicorn
from recognize import find_objects

app = FastAPI()


@app.post("/process_images/")
async def process_images(images: List[UploadFile] = File(...)):
    results = []
    image = images[0]
    print(f"Processing image: {image.filename}")
    contents = await image.read()
    objects = find_objects(contents)
    results.append({"filename": image.filename, "objects": objects})

    return {"results": results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
