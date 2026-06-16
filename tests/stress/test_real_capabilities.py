"""Exercise the full capability surface end to end: vision, embeddings, rerank, OpenAI-SDK
compatibility, and the RAG pipeline — all against the real models."""

from __future__ import annotations

import base64
import io
import math

import pytest


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)


def _red_square_png() -> str:
    """A data URL of a red square on white — a deterministic image any VLM can describe."""
    pytest.importorskip("PIL")
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (224, 224), "white")
    ImageDraw.Draw(img).rectangle([48, 48, 176, 176], fill="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def test_vision_describes_a_real_image(server):
    vlm = server.caps().get("vlm")
    if not vlm:
        pytest.skip("no vision model in this config")
    data_url = _red_square_png()
    text, final = server.chat(
        vlm,
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What color is the shape in this image? One word."},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        max_tokens=40,
        temperature=0.0,
    )
    assert final["choices"][0]["finish_reason"] in ("stop", "length")
    assert "red" in text.lower(), f"VLM did not identify the color: {text!r}"


def test_embeddings_capture_semantic_similarity(server):
    embed = server.caps().get("embedding")
    if not embed:
        pytest.skip("no embedding model in this config")
    r = server.client.post(
        "/v1/embeddings",
        json={"model": embed, "input": ["a dog", "a puppy", "the stock market crashed today"]},
    )
    r.raise_for_status()
    vecs = [d["embedding"] for d in r.json()["data"]]
    assert len(vecs) == 3 and all(len(v) > 0 for v in vecs)
    near = _cosine(vecs[0], vecs[1])  # dog ~ puppy
    far = _cosine(vecs[0], vecs[2])  # dog ~ stock market
    assert near > far, f"embeddings not semantic: near={near:.3f} far={far:.3f}"


def test_rerank_ranks_the_relevant_document_first(server):
    rerank = server.caps().get("rerank")
    if not rerank:
        pytest.skip("no rerank model in this config")
    docs = [
        "Photosynthesis converts sunlight into chemical energy in plants.",
        "To reset your password, click 'Forgot password' and follow the email link.",
        "The Great Wall of China is visible across many provinces.",
    ]
    r = server.client.post(
        "/v1/rerank",
        json={"model": rerank, "query": "How do I reset my account password?", "documents": docs},
    )
    r.raise_for_status()
    results = r.json()["results"]
    assert results[0]["index"] == 1, f"reranker did not surface the password doc: {results}"


def test_openai_sdk_drop_in_compatibility(server):
    """An unmodified OpenAI client must work by only changing the base URL."""
    openai = pytest.importorskip("openai")
    lm = server.caps()["lm"]
    client = openai.OpenAI(base_url=server.base + "/v1", api_key="not-needed")

    models = {m.id for m in client.models.list().data}
    assert lm in models

    resp = client.chat.completions.create(
        model=lm,
        messages=[{"role": "user", "content": "Reply with exactly: pong"}],
        max_tokens=16,
        temperature=0.0,
    )
    assert resp.choices[0].message.content.strip(), "empty SDK completion"


def test_rag_grounded_answer_with_citations_and_dedup(server, tmp_path):
    caps = server.caps()
    if not (caps.get("embedding") and caps.get("lm")):
        pytest.skip("RAG needs an embedding + lm model")
    # A fact the model cannot know without the document.
    fact = "The Zephyr-9 reactor reaches peak output at exactly 4,217 kelvin."
    doc = tmp_path / "zephyr.txt"
    doc.write_text(
        "Internal engineering notes.\n" + fact + "\nIt was commissioned in the Vega facility.\n"
    )

    first = server.client.post("/rag/ingest", json={"paths": [str(doc)]})
    first.raise_for_status()
    # Ingesting the same file again must not duplicate it (dedup/replace on re-ingest).
    server.client.post("/rag/ingest", json={"paths": [str(doc)]}).raise_for_status()

    listed = server.client.get("/rag/documents").json()["documents"]
    matches = [d for d in listed if "zephyr" in str(d).lower()]
    assert len(matches) == 1, f"document was indexed more than once: {matches}"

    q = server.client.post("/rag/query", json={"query": "At what temperature does Zephyr-9 peak?"})
    q.raise_for_status()
    answer = q.json()
    assert "4,217" in answer["answer"] or "4217" in answer["answer"], (
        f"grounded answer missed the fact: {answer['answer']!r}"
    )
    assert answer.get("sources"), "no citations returned"
