from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from mineru import MinerU

app = FastAPI()

miner = MinerU()

@app.post("/parse")
async def parse_pdf(file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    try:
        result = miner(pdf_bytes)
        return JSONResponse(content={"success": True, "data": result})
    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)