"""
Quick smoke test for the SHL Assessment Recommender API.
Run with: python test_api.py
Requires the server to be running: uvicorn main:app --reload
"""

import json
import urllib.request
import urllib.error

BASE = "http://localhost:8000"

def post_chat(messages: list[dict]) -> dict:
    body = json.dumps({"messages": messages}).encode()
    req = urllib.request.Request(
        f"{BASE}/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())

def test_health():
    with urllib.request.urlopen(f"{BASE}/health", timeout=5) as resp:
        data = json.loads(resp.read())
    assert data == {"status": "ok"}, f"Health check failed: {data}"
    print("✓ /health OK")

def test_vague_query_no_recs():
    """Vague first message should NOT produce recommendations."""
    result = post_chat([{"role": "user", "content": "I need an assessment"}])
    assert isinstance(result["reply"], str) and len(result["reply"]) > 0
    assert result["recommendations"] == [], f"Expected empty recs, got: {result['recommendations']}"
    assert result["end_of_conversation"] == False
    print("✓ Vague query → no recommendations on turn 1")

def test_java_developer():
    """Multi-turn: Java developer hiring."""
    msgs = [
        {"role": "user", "content": "Hiring a mid-level Java developer, 4 years experience, works with stakeholders"},
    ]
    r1 = post_chat(msgs)
    print(f"  Turn 1 reply: {r1['reply'][:100]}...")
    print(f"  Turn 1 recs: {len(r1['recommendations'])}")

    msgs.append({"role": "assistant", "content": r1["reply"]})
    msgs.append({"role": "user", "content": "Yes, also need a personality assessment"})
    r2 = post_chat(msgs)
    print(f"  Turn 2 reply: {r2['reply'][:100]}...")
    print(f"  Turn 2 recs: {len(r2['recommendations'])}")
    if r2["recommendations"]:
        for rec in r2["recommendations"]:
            assert "shl.com" in rec["url"], f"Bad URL: {rec['url']}"
            print(f"    - {rec['name']} ({rec['test_type']}) → {rec['url']}")
    print("✓ Java developer conversation")

def test_off_topic_refused():
    """Off-topic question should be refused."""
    result = post_chat([{"role": "user", "content": "What is the best way to write a job description?"}])
    assert result["recommendations"] == []
    print(f"  Off-topic reply: {result['reply'][:120]}")
    print("✓ Off-topic refused")

def test_schema_compliance():
    """Response must always have all three fields."""
    result = post_chat([{"role": "user", "content": "Tell me about OPQ32r"}])
    assert "reply" in result
    assert "recommendations" in result
    assert "end_of_conversation" in result
    assert isinstance(result["recommendations"], list)
    assert isinstance(result["end_of_conversation"], bool)
    print("✓ Schema compliance")

if __name__ == "__main__":
    print("Running smoke tests against", BASE)
    print()
    try:
        test_health()
        test_vague_query_no_recs()
        test_java_developer()
        test_off_topic_refused()
        test_schema_compliance()
        print("\nAll tests passed!")
    except AssertionError as e:
        print(f"\nFAIL: {e}")
    except urllib.error.URLError as e:
        print(f"\nCannot connect to server: {e}")
        print("Start the server first: uvicorn main:app --reload")
