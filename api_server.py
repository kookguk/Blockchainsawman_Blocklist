# api_server.py
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import torch
import pandas as pd

from final import (
    load_model_and_explainer,
    run_pipeline
)

###############################
# FastAPI 초기화
###############################
app = FastAPI(title="Blocklist API")


###############################
# CORS 허용 (iOS 앱에서 호출 가능)
###############################
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # 프로덕션에서는 iOS domain만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


###############################
# 요청 Body 스키마
###############################
class WalletRequest(BaseModel):
    txIds: list[str]   # ["14324"] 또는 여러개


###############################
# 모델 로딩 (서버 켜질 때 1회만)
###############################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base = Path("/Users/kook/Desktop/Blockchainsawman_Blocklist")
model_path = base / "models/saved/elliptic_gat_best.pt"
explainer_path = base / "models/saved/explainer_pg.pt"

df = pd.read_csv(base / "data/df_merged.csv")
idx_to_txId = df["txId"].astype(str).tolist()

data = torch.load(base / "data/elliptic_data_v2.pt", map_location=device)
x, edge_index, y = data.x, data.edge_index, data.y

model, explainer = load_model_and_explainer(device, model_path, explainer_path)


###############################
# API 엔드포인트
###############################
@app.post("/v1/check_wallet")
def check_wallet(req: WalletRequest):

    results = []

    for txId in req.txIds:
        result = run_pipeline(
            txId,
            model,
            explainer,
            x, edge_index, y,
            idx_to_txId
        )
        results.append(result)

    return {"results": results}


###############################
# 로컬 실행
###############################
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
