"""
Personal Memory Assistant
--------------------------
A conversational CLI assistant that remembers context about the user
across sessions using mem0 (vector + graph memory) backed by Qdrant
and Neo4j, with OpenAI for embeddings, LLM reasoning, and chat.

Run:
    docker-compose up -d      # starts Qdrant
    python memm.py
"""

import json
import logging
import os
import sys
from dataclasses import dataclass

from dotenv import load_dotenv
from mem0 import Memory
from openai import OpenAI

# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------

load_dotenv()

# Set DEBUG=true in .env to see full request/response logs from mem0,
# Qdrant, OpenAI, etc. By default, the CLI stays clean and only shows
# our own app-level messages.
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.WARNING,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("memory-assistant")
logger.setLevel(logging.INFO if not DEBUG else logging.DEBUG)

if not DEBUG:
    # These libraries log every HTTP call and internal warning at INFO/WARNING
    # level, which drowns out the actual conversation. Quiet them down.
    for noisy_logger in (
        "httpx",
        "httpcore",
        "openai",
        "mem0",
        "qdrant_client",
        "neo4j",
        "posthog",
        "urllib3",
    ):
        logging.getLogger(noisy_logger).setLevel(logging.ERROR)

    import warnings
    warnings.filterwarnings("ignore")


@dataclass(frozen=True)
class Settings:
    """Centralized, validated application configuration."""

    openai_api_key: str
    chat_model: str
    embedding_model: str

    qdrant_host: str
    qdrant_port: int

    neo4j_url: str
    neo4j_username: str
    neo4j_password: str

    default_user_id: str

    @classmethod
    def from_env(cls) -> "Settings":
        required = {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "NEO4J_URL": os.getenv("NEO4J_URL"),
            "NEO4J_USERNAME": os.getenv("NEO4J_USERNAME"),
            "NEO4J_PASSWORD": os.getenv("NEO4J_PASSWORD"),
        }
        missing = [key for key, value in required.items() if not value]
        if missing:
            raise EnvironmentError(
                f"Missing required environment variable(s): {', '.join(missing)}. "
                "Set them in a .env file (see .env.example)."
            )

        return cls(
            openai_api_key=required["OPENAI_API_KEY"],
            chat_model=os.getenv("CHAT_MODEL", "gpt-4.1-mini"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
            qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
            neo4j_url=required["NEO4J_URL"],
            neo4j_username=required["NEO4J_USERNAME"],
            neo4j_password=required["NEO4J_PASSWORD"],
            default_user_id=os.getenv("DEFAULT_USER_ID", "siddhesh"),
        )

    def mem0_config(self) -> dict:
        return {
            "version": "v1.1",
            "embedder": {
                "provider": "openai",
                "config": {
                    "api_key": self.openai_api_key,
                    "model": self.embedding_model,
                },
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "api_key": self.openai_api_key,
                    "model": self.chat_model,
                },
            },
            "graph_store": {
                "provider": "neo4j",
                "config": {
                    "url": self.neo4j_url,
                    "username": self.neo4j_username,
                    "password": self.neo4j_password,
                },
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "host": self.qdrant_host,
                    "port": self.qdrant_port,
                },
            },
        }


# --------------------------------------------------------------------------
# Core assistant
# --------------------------------------------------------------------------

class MemoryAssistant:
    """
    A lightweight chat assistant that retrieves relevant long-term
    memories before responding, then persists the new exchange back
    into memory for future recall.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.openai_client = OpenAI(api_key=settings.openai_api_key)
        self.memory = Memory.from_config(settings.mem0_config())

    def _retrieve_relevant_memories(self, query: str, user_id: str) -> list[str]:
        """Fetch memories relevant to the current query for this user."""
        try:
            results = self.memory.search(query=query, filters={"user_id": user_id})
        except Exception:
            logger.exception("Memory search failed; continuing without context.")
            return []

        memories = [
            f"ID: {item.get('id')}\nMemory: {item.get('memory')}"
            for item in results.get("results", [])
        ]
        logger.debug("Retrieved %d relevant memor%s.", len(memories), "y" if len(memories) == 1 else "ies")
        return memories

    def _build_system_prompt(self, memories: list[str]) -> str:
        return (
            "You are a helpful personal assistant. Use the following remembered "
            "context about the user, where relevant, to tailor your response. "
            "If nothing is relevant, ignore it.\n\n"
            f"User context:\n{json.dumps(memories, indent=2)}"
        )

    def _generate_response(self, user_query: str, memories: list[str]) -> str:
        response = self.openai_client.chat.completions.create(
            model=self.settings.chat_model,
            messages=[
                {"role": "system", "content": self._build_system_prompt(memories)},
                {"role": "user", "content": user_query},
            ],
        )
        return response.choices[0].message.content

    def _persist_exchange(self, user_query: str, ai_response: str, user_id: str) -> None:
        try:
            self.memory.add(
                user_id=user_id,
                messages=[
                    {"role": "user", "content": user_query},
                    {"role": "assistant", "content": ai_response},
                ],
            )
            logger.debug("Exchange saved to memory.")
        except Exception:
            logger.exception("Failed to persist exchange to memory.")

    def chat(self, user_query: str, user_id: str | None = None) -> str:
        """Run one full memory-augmented chat turn and return the reply."""
        user_id = user_id or self.settings.default_user_id

        memories = self._retrieve_relevant_memories(user_query, user_id)
        ai_response = self._generate_response(user_query, memories)
        self._persist_exchange(user_query, ai_response, user_id)

        return ai_response


# --------------------------------------------------------------------------
# CLI entrypoint
# --------------------------------------------------------------------------

def run_cli() -> None:
    try:
        settings = Settings.from_env()
    except EnvironmentError as exc:
        print(f"Configuration error: {exc}")
        sys.exit(1)

    print("Setting up memory assistant (Qdrant + Neo4j + OpenAI)...")
    assistant = MemoryAssistant(settings)

    print("=" * 50)
    print(" Memory Assistant — ready to chat")
    print(" Type 'exit' or press Ctrl+C to quit.")
    if DEBUG:
        print(" (debug mode: verbose logs enabled)")
    print("=" * 50)

    while True:
        try:
            user_query = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_query:
            continue
        if user_query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        print("Assistant is thinking...", end="\r")
        ai_response = assistant.chat(user_query)
        print(" " * 30, end="\r")  # clear the "thinking" line
        print(f"Assistant: {ai_response}")


if __name__ == "__main__":
    run_cli()