from fastapi.testclient import TestClient
from server.app import app

client = TestClient(app)

print("Testing GET /health")
health = client.get("/health")
print("Health Status:", health.status_code)
print("Health Body:", health.json())

print("\nTesting POST /reset with {}")
reset1 = client.post("/reset", json={})
print("Reset1 Status:", reset1.status_code)

print("\nTesting POST /reset with {'episode_id': 'audit_001'}")
reset2 = client.post("/reset", json={"episode_id": "audit_001"})
print("Reset2 Status:", reset2.status_code)
