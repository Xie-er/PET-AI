import os
from typing import TypedDict, Annotated, List, Union
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

from src.agents.prompts import (
    ROUTER_SYSTEM_PROMPT, 
    HEALTH_AGENT_PROMPT, 
    DIET_AGENT_PROMPT, 
    CARE_AGENT_PROMPT,
    SAFETY_AGENT_PROMPT
)
from src.utils.rag import get_retriever

# Load environment variables
load_dotenv()

# Initialize LLM
# Ensure DASHSCOPE_API_KEY is set in .env
llm = ChatTongyi(
    model="qwen-turbo", # or qwen-max, qwen-plus
    temperature=0.7
)

# Initialize Retriever
retriever = None
try:
    retriever = get_retriever()
except Exception:
    pass # Fallback to LLM only if RAG fails

# Define State
class AgentState(TypedDict):
    messages: List[BaseMessage]
    next_node: str
    final_answer: str

# Define Nodes

def router_node(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1].content
    
    # Simple routing logic using LLM
    prompt = ChatPromptTemplate.from_messages([
        ("system", ROUTER_SYSTEM_PROMPT),
        ("user", "{input}")
    ])
    chain = prompt | llm
    response = chain.invoke({"input": last_message})
    decision = response.content.strip().lower()
    
    if "health" in decision:
        return {"next_node": "health_agent"}
    elif "diet" in decision:
        return {"next_node": "diet_agent"}
    elif "care" in decision:
        return {"next_node": "care_agent"}
    else:
        return {"next_node": "general_chat"}

def health_agent_node(state: AgentState):
    messages = state["messages"]
    query = messages[-1].content
    
    context = ""
    if retriever:
        try:
            docs = retriever.invoke(query)
            context = "\n".join([doc.page_content for doc in docs])
        except:
            pass
            
    prompt_text = HEALTH_AGENT_PROMPT
    if context:
        prompt_text += f"\n\n参考资料：\n{context}"
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_text),
        ("user", "{input}")
    ])
    chain = prompt | llm
    response = chain.invoke({"input": query})
    return {"final_answer": response.content, "next_node": "safety_agent"}

def diet_agent_node(state: AgentState):
    messages = state["messages"]
    query = messages[-1].content
    
    context = ""
    if retriever:
        try:
            docs = retriever.invoke(query)
            context = "\n".join([doc.page_content for doc in docs])
        except:
            pass

    prompt_text = DIET_AGENT_PROMPT
    if context:
        prompt_text += f"\n\n参考资料：\n{context}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_text),
        ("user", "{input}")
    ])
    chain = prompt | llm
    response = chain.invoke({"input": query})
    return {"final_answer": response.content, "next_node": "safety_agent"}

def care_agent_node(state: AgentState):
    messages = state["messages"]
    query = messages[-1].content
    
    context = ""
    if retriever:
        try:
            docs = retriever.invoke(query)
            context = "\n".join([doc.page_content for doc in docs])
        except:
            pass

    prompt_text = CARE_AGENT_PROMPT
    if context:
        prompt_text += f"\n\n参考资料：\n{context}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_text),
        ("user", "{input}")
    ])
    chain = prompt | llm
    response = chain.invoke({"input": query})
    return {"final_answer": response.content, "next_node": "safety_agent"}

def general_chat_node(state: AgentState):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"final_answer": response.content, "next_node": "safety_agent"}

def safety_agent_node(state: AgentState):
    answer = state.get("final_answer", "")
    
    prompt = ChatPromptTemplate.from_template(SAFETY_AGENT_PROMPT)
    chain = prompt | llm
    response = chain.invoke({"answer": answer})
    
    # Update the final message with the safety-checked answer
    return {"messages": [AIMessage(content=response.content)]}

# Build Graph
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("router", router_node)
workflow.add_node("health_agent", health_agent_node)
workflow.add_node("diet_agent", diet_agent_node)
workflow.add_node("care_agent", care_agent_node)
workflow.add_node("general_chat", general_chat_node)
workflow.add_node("safety_agent", safety_agent_node)

# Add Edges
workflow.set_entry_point("router")

workflow.add_conditional_edges(
    "router",
    lambda state: state["next_node"],
    {
        "health_agent": "health_agent",
        "diet_agent": "diet_agent",
        "care_agent": "care_agent",
        "general_chat": "general_chat"
    }
)

workflow.add_edge("health_agent", "safety_agent")
workflow.add_edge("diet_agent", "safety_agent")
workflow.add_edge("care_agent", "safety_agent")
workflow.add_edge("general_chat", "safety_agent")
workflow.add_edge("safety_agent", END)

# Compile
app_graph = workflow.compile()
