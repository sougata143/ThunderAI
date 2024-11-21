import pytest
from fastapi.testclient import TestClient
from ..ml.models.bert import BERTModel
from ..ml.models.gpt import GPTModel
from ..ml.models.lstm import LSTMModel

@pytest.fixture
def bert_model():
    config = {
        "max_length": 512,
        "num_labels": 2,
        "batch_size": 32
    }
    return BERTModel(config)

@pytest.fixture
def gpt_model():
    config = {
        "max_length": 1024,
        "num_labels": 2,
        "temperature": 0.7
    }
    return GPTModel(config)

def test_bert_prediction(bert_model):
    text = "This is a test sentence for BERT model."
    result = bert_model.predict(text)
    
    assert "prediction" in result
    assert "confidence" in result
    assert isinstance(result["prediction"], int)
    assert isinstance(result["confidence"], float)
    assert 0 <= result["confidence"] <= 1

def test_gpt_prediction(gpt_model):
    text = "This is a test sentence for GPT model."
    result = gpt_model.predict(text)
    
    assert "prediction" in result
    assert "confidence" in result
    assert "probabilities" in result
    assert isinstance(result["prediction"], int)
    assert isinstance(result["confidence"], float)
    assert len(result["probabilities"]) == 2

@pytest.mark.asyncio
async def test_model_training_endpoint(client, test_user):
    training_data = {
        "texts": ["positive text", "negative text"],
        "labels": [1, 0]
    }
    
    response = await client.post(
        "/api/v1/models/train/bert",
        json=training_data,
        headers={"Authorization": f"Bearer {test_user['access_token']}"}
    )
    
    assert response.status_code == 200
    assert "message" in response.json()
    assert "model_name" in response.json() 