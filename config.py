from pathlib import Path

DATA_DIR = Path("data")
DB_DIR = Path("db")
DB_DIR.mkdir(exist_ok=True)

OLLAMA_MODEL = "mistral:7b"
OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "mxbai-embed-large"

CHROMA_PATH = str(DB_DIR / "ux_db")
TOP_K = 7

TEMPERATURE = 0