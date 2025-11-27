import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from pathlib import Path
import torch
import pandas as pd

from final import load_model_and_explainer, run_pipeline


###############################
# FastAPI 초기화
###############################
app = FastAPI(
    title="Blocklist API",
    description="온체인 AML 위험도 분석 API",
    version="1.0.0"
)


###############################
# CORS 허용
###############################
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # 실제 배포 시 iOS 앱 origin으로 설정
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


###############################
# 요청 Body 스키마
###############################
class WalletRequest(BaseModel):
    txIds: List[str]


###############################
# 응답 스키마들 (Swagger 문서 생성용)
###############################
class Edge(BaseModel):
    source: str
    target: str
    importance: float

class Node(BaseModel):
    id: str

class EvidenceGraph(BaseModel):
    nodes: List[Node]
    edges: List[Edge]

class WalletResult(BaseModel):
    txId: str
    risk_score: float
    status: str
    explanation: str
    explanation_summary: str
    explanation_bullet: List[str]
    evidence_graph: EvidenceGraph

class WalletResponse(BaseModel):
    results: List[WalletResult]


###############################
# 모델 로딩 (서버 시작 시 1회)
###############################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ❗ 서버용 경로 (중요)
base = Path("/home/ubuntu/Blockchainsawman_Blocklist")

model_path = base / "models/saved/elliptic_gat_best.pt"
explainer_path = base / "models/saved/explainer_pg.pt"
csv_path = base / "data/df_merged.csv"
pt_path = base / "data/elliptic_data_v2.pt"

# CSV load
df = pd.read_csv(csv_path)
idx_to_txId = df["txId"].astype(str).tolist()

# Graph data load
data = torch.load(pt_path, map_location=device)
x, edge_index, y = data.x, data.edge_index, data.y

# Model + Explainer load
model, explainer = load_model_and_explainer(
    device,
    model_path,
    explainer_path
)


###############################
# API 엔드포인트
###############################
@app.post("/v1/check_wallet", response_model=WalletResponse)
def check_wallet(req: WalletRequest):
    results = []

    for txId in req.txIds:
        try:
            result = run_pipeline(
                txId,
                model,
                explainer,
                x, edge_index, y,
                idx_to_txId
            )
            results.append(result)

        except Exception as e:
            # 오류 발생 시에도 API는 정상적으로 JSON을 반환해야 함
            results.append({
                "txId": txId,
                "risk_score": 0.0,
                "status": "Error",
                "explanation": f"Error: {str(e)}",
                "explanation_summary": "",
                "explanation_bullet": [],
                "evidence_graph": {
                    "nodes": [],
                    "edges": []
                }
            })

    return {"results": results}


###############################
# 로컬 실행용
###############################
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )