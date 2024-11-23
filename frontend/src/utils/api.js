const API_URL = 'http://localhost:8001';

export async function fetchHealthStatus() {
    const response = await fetch(`${API_URL}/health`);
    return await response.json();
}

export async function testDatabase() {
    const response = await fetch(`${API_URL}/db-test`);
    return await response.json();
}

export async function testPrediction(data) {
    const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
    });
    return await response.json();
}

export async function fetchMetrics() {
    const response = await fetch(`${API_URL}/metrics`);
    return await response.text();
}
