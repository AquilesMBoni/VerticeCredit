from fastapi.testclient import TestClient

GOOD = {"person_age": 29, "person_income": 52000, "person_home_ownership": "RENT", "person_emp_length": 4,
        "loan_intent": "EDUCATION", "loan_grade": "B", "loan_amnt": 9000, "loan_int_rate": 11.5,
        "cb_person_default_on_file": "N", "cb_person_cred_hist_length": 5}
RISKY = GOOD | {"person_income": 15000, "loan_amnt": 12000, "loan_grade": "E", "loan_int_rate": 18.0,
                "cb_person_default_on_file": "Y"}
BR_GOOD = {"idade": 34, "ocupacao": "Assalariado CLT", "renda_mensal": 3200, "tempo_emprego_meses": 30,
           "score_bureau": 720, "possui_restricao": "Não", "uf": "sp", "modalidade": "Consignado privado (CLT)",
           "valor_contratado": 6000, "prazo_meses": 24, "taxa_juros_am": 3.2, "rating": "A"}
BR_RISKY = BR_GOOD | {"modalidade": "Pessoal não consignado", "score_bureau": 300, "possui_restricao": "Sim",
                      "rating": "H", "taxa_juros_am": 7.5, "prazo_meses": 24, "valor_contratado": 15000}


def client():
    from vertice.api.main import app
    return TestClient(app)


def test_health(pipeline_run):
    body = client().get("/health").json()
    assert body["datasets"]["us"]["has_model"] and body["datasets"]["br"]["has_model"]


def test_predict_orders_risk_us(pipeline_run):
    c = client()
    low, high = c.post("/predict", json=GOOD).json(), c.post("/predict", json=RISKY).json()
    assert 0 <= low["probability_default"] < high["probability_default"] <= 1
    assert low["decision"] == "Aprovar" and high["decision"] == "Recusar"


def test_predict_orders_risk_br(pipeline_run):
    c = client()
    low, high = c.post("/br/predict", json=BR_GOOD).json(), c.post("/br/predict", json=BR_RISKY).json()
    assert low["dataset"] == "br" and low["probability_default"] < high["probability_default"]
    assert len(c.get("/predict/log", params={"dataset": "br"}).json()) >= 2


def test_br_rejects_regulatory_violations(pipeline_run):
    c = client()
    r = c.post("/br/predict", json=BR_GOOD | {"modalidade": "Consignado INSS", "taxa_juros_am": 2.4})
    assert r.status_code == 422
    msg = r.json()["detail"][0]["msg"]
    assert "Aposentado" in msg and "teto" in msg
    assert c.post("/br/predict", json=BR_GOOD | {"valor_contratado": 60000}).status_code == 422  # margem 35%


def test_data_endpoints(pipeline_run):
    c = client()
    page = c.get("/data/loans", params={"size": 5, "grade": "A"}).json()
    assert len(page["items"]) == 5 and all(i["loan_grade"] == "A" for i in page["items"])
    seg = c.get("/data/segments/rating", params={"dataset": "br"}).json()
    assert [s["segment"] for s in seg] == ["AA", "A", "B", "C", "D", "E", "F", "G", "H"]
    assert c.get("/data/segments/drop table").status_code == 422
    assert all(ch["status"] == "aprovado" for ch in c.get("/quality/checks", params={"stage": "limpo", "dataset": "br"}).json())
