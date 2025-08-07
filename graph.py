from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langgraph.graph import StateGraph, END

from rag import search_rules
from prompt import ux_prompt, query_rewrite_prompt
import config

llm = ChatOllama(
    model=config.OLLAMA_MODEL,
    base_url=config.OLLAMA_URL,
    temperature=config.TEMPERATURE
)

rewrite_chain = query_rewrite_prompt | llm | StrOutputParser()
ux_response_chain = ux_prompt | llm | JsonOutputParser()


def rewrite_query_node(state):
    rewritten = rewrite_chain.invoke({"question": state["question"]})
    return {**state, "keywords": rewritten}

def retrieve_docs_node(state):
    rules = search_rules(state["keywords"])
    print(f"Mots-clés : {state['keywords']}")
    print(f"Règles trouvées ({len(rules)}):")
    for rule in rules:
        content = rule['content']
        titre = content.split('titre :')[1].split(' / catégorie')[0].strip()
        print(f"  - {titre}")

    context_rules = []
    for rule in rules:
        content = rule['content']
        titre = content.split('titre :')[1].split(' / catégorie')[0].strip()
        description = content.split(' / ')[-1].strip()
        context_rules.append(f"- {titre}: {description}")
    
    context = "\n".join(context_rules)
    return {**state, "context": context}

def generate_response_node(state):
    result = ux_response_chain.invoke({
        "context": state["context"],
        "question": state["question"]
    })
    return {**state, "response": result}


def build_graph():
    builder = StateGraph(dict)

    builder.add_node("rewrite", rewrite_query_node)
    builder.add_node("retrieve", retrieve_docs_node)
    builder.add_node("generate", generate_response_node)

    builder.set_entry_point("rewrite")
    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)

    return builder.compile()
