from fastapi import FastAPI, Form, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from contextlib import asynccontextmanager
from io import StringIO
import pandas as pd


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    print("at startup")
    yield
# app = FastAPI()
app = FastAPI(lifespan=lifespan)

# Set up the Jinja2Templates instance and specify the directory for templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...)):
    # Read the uploaded Excel file
    df = pd.read_csv(file.file)

    filtered_df = df.head()
    
    # Convert the filtered DataFrame to CSV
    output = StringIO()
    filtered_df.to_csv(output, index=False)
    output.seek(0)
    
    # Return the CSV file as a response
    response = StreamingResponse(output, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=search_results.csv"})
    return response


    


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
