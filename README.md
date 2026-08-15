# 🧠 Memory Assistant

A personal AI assistant that **remembers you** across conversations — built with mem0, Qdrant, Neo4j, and OpenAI.

Unlike a plain chatbot that forgets everything once the session ends, this assistant stores meaningful facts from your conversations and recalls them in future chats — so it actually gets to know you over time.

---

## ⚙️ How It Works

1. You type a message.
2. The assistant searches its **memory store** for anything relevant to what you just said.
3. That context is quietly added to the system prompt sent to the LLM.
4. The LLM replies — now aware of your history.
5. The new exchange is saved back into memory for next time.

```
You → [Search Memory] → [LLM + Context] → Reply → [Save to Memory]
```

## 🧩 Tech Stack

| Component | Role |
|---|---|
| **[mem0](https://github.com/mem0ai/mem0)** | Memory layer — decides what's worth remembering and retrieves it later |
| **[Qdrant](https://qdrant.tech/)** | Vector database — stores memories as embeddings for semantic search |
| **[Neo4j](https://neo4j.com/)** | Graph database — stores relationships *between* memories (e.g. people, entities, facts) |
| **[OpenAI](https://platform.openai.com/)** | `gpt-4.1-mini` for chat, `text-embedding-3-small` for embeddings |
| **Docker** | Runs Qdrant locally with a single command |

---

## 📦 Prerequisites

- Python 3.10+
- Docker Desktop
- An [OpenAI API key](https://platform.openai.com/api-keys)
- A free [Neo4j AuraDB](https://neo4j.com/cloud/aura/) instance (or any Neo4j instance)

---

## 🔑 Environment Setup

Create a `.env` file in the project root:

```bash
# OpenAI
OPENAI_API_KEY=your-openai-api-key

# Neo4j (graph memory store)
NEO4J_URL=neo4j+s://your-instance-id.databases.neo4j.io
NEO4J_USERNAME=your-neo4j-username
NEO4J_PASSWORD=your-neo4j-password
```

---

## 🚀 Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/Siddhesh1732/mem0.git
cd mem0
```

**2. Set up a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux
```

**3. Install dependencies**
```bash
pip install mem0ai openai python-dotenv
```

**4. Start Qdrant (vector DB)**
```bash
docker-compose up -d
```

**5. Add your `.env` file** (see above)

**6. Run the assistant**
```bash
python memm.py
```

You should see:

```
==================================================
 Memory Assistant — ready to chat
 Type 'exit' or press Ctrl+C to quit.
==================================================

You:
```

---

## 💡 Example

```
You: I'm a software engineer who loves backend systems and Java.
Assistant: Nice! I'll remember that.

--- (new session) ---

You: Any project ideas for me?
Assistant: Since you're into backend systems and Java, how about
building a distributed task scheduler...
```

---


## 📄 License

MIT — free to use, modify, and build on.

---

## 👤 About Me

Built by **Siddhesh Karemore** while exploring how LLMs can be made more useful with memory and context.

Feel free to check out my [GitHub profile](https://github.com/Siddhesh1732) for more projects, or connect on [LinkedIn](https://www.linkedin.com/in/siddhesh-karemore-a62586229/).
