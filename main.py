from fastapi import FastAPI, Form, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from contextlib import asynccontextmanager
from io import StringIO
import pandas as pd
import time
from modules import Trie, insert_data, search_in_trie, preprocess_text,process_batches,process_results

def process_file(df_test):
    # print(list(df_test))
    df_test=df_test.head()
    df_test=df_test.fillna('')
    df_test.drop_duplicates(inplace = True)
    df_test = df_test[["Name", "FatherName", "CitizenshipNumber", "BlacklistNumber"]]
    # Convert columns to strings
    df_test['Name'] = df_test['Name'].astype(str)
    df_test['FatherName'] = df_test['FatherName'].astype(str)
    df_test['CitizenshipNumber'] = df_test['CitizenshipNumber'].astype(str)
    df_test['BlacklistNumber'] = df_test['BlacklistNumber'].astype(str)
    # Concatenate data with removing whitespaces
    df_test['Combined'] = df_test['Name'].str.strip() + df_test['FatherName'].str.strip() + df_test['CitizenshipNumber'].str.strip()
    # Example usage:
    queries = df_test['Combined'].tolist()

    start_time = time.time()

    print(start_time)
    df = app.state.df
    # Concatenate data with removing whitespaces
    df['Combined'] = df['FName'].str.strip() +df['MiddleName'].str.strip()+df['LastName'].str.strip() + df['FatherName'].str.strip()+ df['IDNo'].astype(str)

    # Process the queries using your DataFrame 'df'
    results = process_batches(queries, df)

    # Measure execution time
    end_time = time.time()
    execution_time = end_time - start_time

    print(results)
    print(f"Execution time: {execution_time} seconds")
    # Assuming results is your dictionary and df_test is your DataFrame
    df_output = process_results(results, df_test)
    print(df_output)

    
    return df_output

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model

    print("at startup")
    path = r'E:\data\real-data-exact-match\combined_inner_join.csv'  #specify where the complete data is located where search is to be performed
    df = pd.read_csv(path)
    # df= df.head(100)
    if not df.empty:
        print('df loaded successfully')
    app.state.df = df
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
    # Process the file
    df_test = pd.read_excel(file.file)
    filtered_df = process_file(df_test)
    print(list(filtered_df))
    
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
