import os
from ingestion import ingest_pdf
from graph import build_graph

CHROMA_DIR = "./chroma_db"


def main():
    # Ingest only if ChromaDB doesn't exist yet
    if not os.path.exists(CHROMA_DIR) or not os.listdir(CHROMA_DIR):
        ingest_pdf()
    else:
        print("📂 ChromaDB already exists, skipping ingestion.")

    print("\n🤖 RAG Customer Support Bot")
    print("Type 'exit' to quit.\n")

    app = build_graph()

    while True:
        query = input("You: ").strip()

        if query.lower() == "exit":
            print("👋 Goodbye!")
            break

        if not query:
            continue

        result = app.invoke({
            "query": query,
            "context": [],
            "response": "",
            "escalated": False,
            "final_answer": ""
        })

        if result["escalated"]:
            print(f"\n✅ [Human Agent]: {result['final_answer']}\n")
        else:
            print(f"\n🤖 [Bot]: {result['final_answer']}\n")


if __name__ == "__main__":
    main()