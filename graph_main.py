from rag import setup_rag
from graph import build_graph

def main():
    setup_rag()
    graph = build_graph()

    while True:
        question = input("\nQuestion UX: ")
        if question.lower() == "quit":
            break

        try:
            result = graph.invoke({"question": question})
            print("\nRecommandations UX:")
            print(result["response"])
        except Exception as e:
            print(f"\nErreur : {e}")

if __name__ == "__main__":
    main()
