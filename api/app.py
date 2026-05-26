from flask import Flask, request, jsonify
import os
import tempfile
from src.pipeline import ingest
from src.retriever import answer
from src.memory import delete_session

app = Flask(__name__)


@app.route("/ingest", methods=["POST"])
def ingest_doc():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        file.save(tmp.name)
        try:
            n_chunks = ingest(tmp.name)
        finally:
            os.unlink(tmp.name)

    return jsonify({"message": f"Indexed {n_chunks} chunks from {file.filename}"}), 200


@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    question = data.get("question", "").strip()
    session_id = data.get("session_id", "default")

    if not question:
        return jsonify({"error": "question required"}), 400

    result = answer(question, session_id)
    return jsonify(result), 200


@app.route("/session/<session_id>", methods=["DELETE"])
def clear_session(session_id):
    delete_session(session_id)
    return jsonify({"message": "Session cleared"}), 200


if __name__ == "__main__":
    app.run(debug=True, port=5001)
