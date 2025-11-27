import torch
import torch.nn.functional as F
import json
from pathlib import Path
from torch_geometric.nn import GATv2Conv
from torch_geometric.explain import Explainer, PGExplainer
import pandas as pd

from typing import TypedDict, List
from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


###############################################################
# 0) GAT 모델 정의
###############################################################
class GATNet(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out, heads=8):
        super().__init__()
        self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads, dropout=0.3)
        self.gat2 = GATv2Conv(dim_h * heads, dim_out,
                              heads=heads, concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        h = self.gat1(x, edge_index)
        h = F.elu(h)
        h = self.gat2(h, edge_index)
        return h


###############################################################
# 1) 모델 + PGExplainer 로드
###############################################################
def load_model_and_explainer(device, model_path, explainer_ckpt_path):
    ckpt = torch.load(model_path, map_location=device)
    meta = ckpt["meta"]

    model = GATNet(
        dim_in=meta["dim_in"],
        dim_h=meta["dim_h"],
        dim_out=meta["dim_out"],
        heads=meta["heads"]
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    target_epochs = 20
    explainer = Explainer(
        model=model,
        algorithm=PGExplainer(target_epochs, lr=0.003).to(device),
        explanation_type="phenomenon",
        edge_mask_type="object",
        model_config=dict(
            mode="multiclass_classification",
            task_level="node",
            return_type="raw"
        )
    )

    ckpt2 = torch.load(explainer_ckpt_path, map_location=device)
    state_dict = ckpt2.get("state_dict", ckpt2)
    explainer.algorithm.load_state_dict(state_dict)

    explainer.algorithm._curr_epoch = target_epochs
    explainer.algorithm.training = False

    return model, explainer


###############################################################
# 2) TypedDict 기반 스키마 (FastAPI와 절대 충돌 X)
###############################################################
class AMLAnalysis(TypedDict):
    explanation: str
    explanation_summary: str
    explanation_bullet: List[str]


###############################################################
# 3) GNN → 중간 JSON (Explainer 결과)
###############################################################
def get_explainer_output(explainer, model, x, edge_index, node_idx, idx_to_txId, y):

    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index)[node_idx]
        prob = torch.softmax(logits, dim=0)[1].item()

    risk_score = round(prob, 4)
    status = "High_Risk" if prob >= 0.8 else ("Medium_Risk" if prob >= 0.5 else "Low_Risk")

    explanation = explainer(
        x,
        edge_index,
        index=node_idx,
        target=torch.tensor([1], device=x.device)
    )

    edge_mask = explanation.edge_mask
    sub_edges = explanation.edge_index

    edges = []
    for i in range(sub_edges.shape[1]):
        src = sub_edges[0, i].item()
        dst = sub_edges[1, i].item()
        edges.append({
            "source": idx_to_txId[src],
            "target": idx_to_txId[dst],
            "importance": round(edge_mask[i].item(), 6)
        })

    edges.sort(key=lambda x: x["importance"], reverse=True)
    edges = edges[:10]

    top_nodes = set([node_idx])
    for e in edges:
        top_nodes.add(idx_to_txId.index(e["source"]))
        top_nodes.add(idx_to_txId.index(e["target"]))

    nodes_json = [{"id": idx_to_txId[n]} for n in top_nodes]

    return {
        "txId": idx_to_txId[node_idx],
        "risk_score": risk_score,
        "status": status,
        "evidence_graph": {
            "nodes": nodes_json,
            "edges": edges
        }
    }


###############################################################
# 4) Neo4j 2-hop Evidence Graph 생성
###############################################################
AURA_URI = "neo4j+s://f7844108.databases.neo4j.io"
AURA_USER = "neo4j"
AURA_PASSWORD = "RWZVkpn0rWZ8g2xN2qHeMohglmtocayP4UH5k0V_SiA"

driver = GraphDatabase.driver(AURA_URI, auth=(AURA_USER, AURA_PASSWORD))


def fetch_subgraph(tx_list):
    cypher = """
    MATCH (w:Wallet)
    WHERE w.txId IN $txList
    MATCH p=(w)-[*1..2]-(n)
    RETURN p
    """
    with driver.session() as s:
        return list(s.run(cypher, txList=[str(t) for t in tx_list]))


def build_evidence_json(records, tx_list):
    nodes, edges = {}, {}

    for r in records:
        p = r["p"]

        for nd in p.nodes:
            tx = nd.get("txId")
            cls = int(nd.get("class", 3)) if nd.get("class") else 3

            nodes[tx] = {
                "id": tx,
                "class": cls,
                "entity_type": nd.get("entity_type", "unknown"),
                "comment": nd.get("comment", "unknown")
            }

        for rel in p.relationships:
            s = rel.start_node.get("txId")
            t = rel.end_node.get("txId")
            edges[(s, t)] = {
                "source": s,
                "target": t,
                "amount": rel.get("amount", "unknown"),
                "time": rel.get("time", "unknown")
            }

    # fallback 처리
    for tx in tx_list:
        if tx not in nodes:
            nodes[tx] = {
                "id": tx,
                "class": 3,
                "entity_type": "unknown",
                "comment": "unknown"
            }
            edges[(tx, tx)] = {
                "source": tx,
                "target": tx,
                "amount": "unknown",
                "time": "unknown"
            }

    return {"nodes": list(nodes.values()), "edges": list(edges.values())}


###############################################################
# 5) LangChain LLM 분석 (BaseModel 제거됨)
###############################################################
def analyze_with_llm(evidence_json, risk_score, txId):

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.2,
    )

    prompt = ChatPromptTemplate.from_template("""
당신은 블록체인 AML 분석 전문가입니다.

지갑 {txId} 의 risk_score: {risk_score}

evidence_graph:
{evidence_json}

아래 형식으로 출력하세요:

1) explanation: 상세 설명 (문자열)
2) explanation_summary: 요약 (문자열, 20자 이내)
3) explanation_bullet: ['포인트1','포인트2','포인트3']
""")

    chain = prompt | llm
    raw = chain.invoke({
        "txId": txId,
        "risk_score": risk_score,
        "evidence_json": json.dumps(evidence_json, ensure_ascii=False)
    })

    # 문자열만 주어지므로 최소 파싱
    content = raw.content

    return {
        "explanation": content,
        "explanation_summary": "",
        "explanation_bullet": []
    }


###############################################################
# 6) 파이프라인 실행 (최종 JSON만 반환)
###############################################################
def run_pipeline(txid, model, explainer, x, edge_index, y, idx_to_txId):

    if txid not in idx_to_txId:
        raise ValueError(f"CSV에 없는 txId: {txid}")

    node_idx = idx_to_txId.index(txid)

    mid_json = get_explainer_output(
        explainer, model, x, edge_index, node_idx, idx_to_txId, y
    )

    node_list = [n["id"] for n in mid_json["evidence_graph"]["nodes"]]
    records = fetch_subgraph(node_list)
    evidence_json = build_evidence_json(records, node_list)

    llm_out = analyze_with_llm(evidence_json, mid_json["risk_score"], txid)

    return {
        "txId": txid,
        "risk_score": mid_json["risk_score"],
        "status": mid_json["status"],
        "explanation": llm_out["explanation"],
        "explanation_summary": llm_out["explanation_summary"],
        "explanation_bullet": llm_out["explanation_bullet"],
        "evidence_graph": evidence_json
    }