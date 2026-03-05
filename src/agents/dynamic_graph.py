import os
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, AIMessage
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

from src.utils.config import config_manager
from src.utils.rag import get_retriever
from src.agents.prompts import SAFETY_AGENT_PROMPT

# Load environment variables
load_dotenv()

# Initialize LLM
llm = ChatTongyi(
    model="qwen-turbo",
    temperature=0.7
)

# Define State
class AgentState(TypedDict):
    messages: List[BaseMessage]
    next_node: str
    final_answer: str

# ----------------------------------------------------------------------
# Dynamic Graph Builder
# ----------------------------------------------------------------------

def build_graph():
    """
    Rebuilds the LangGraph based on the current configuration in config_manager.
    """
    workflow = StateGraph(AgentState)
    
    # 1. Get all configured agents
    agents_config = config_manager.get_agents()
    
    # 2. Define Router Node
    def router_node(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1].content
        
        # Dynamically construct router prompt
        agent_list_text = ""
        for i, agent in enumerate(agents_config):
            agent_list_text += f"{i+1}. {agent['name']}: {agent['description']}\n"
        
        agent_names = ", ".join([a['name'] for a in agents_config])
        
        router_prompt = f"""你是一个智能路由助手。请根据用户输入，决定调用哪个专家智能体。

可用的专家智能体：
{agent_list_text}

如果用户的问题不属于以上任何类别（例如闲聊），请直接回复 "general_chat"。

输出格式要求：
只输出专家的名称（{agent_names}），或者 "general_chat"。不要输出其他内容。
"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", router_prompt),
            ("user", "{input}")
        ])
        chain = prompt | llm
        response = chain.invoke({"input": last_message})
        decision = response.content.strip()
        
        # Validate decision
        valid_nodes = [a['name'] for a in agents_config] + ["general_chat"]
        
        # Simple heuristic to find best match if exact match fails
        if decision not in valid_nodes:
            for node in valid_nodes:
                if node in decision:
                    decision = node
                    break
        
        if decision not in valid_nodes:
            decision = "general_chat"
            
        return {"next_node": decision}

    workflow.add_node("router", router_node)
    
    # 3. Define Expert Nodes dynamically
    # We need a factory function to create node functions that capture the specific agent config
    def create_agent_node(agent_config):
        def agent_node(state: AgentState):
            messages = state["messages"]
            query = messages[-1].content
            
            # RAG Logic
            context = ""
            kb_id = agent_config.get("kb_id")
            if kb_id:
                kb_meta = config_manager.get_kb_by_id(kb_id)
                if kb_meta:
                    retriever = get_retriever(kb_meta["persist_directory"])
                    if retriever:
                        try:
                            docs = retriever.invoke(query)
                            context = "\n".join([doc.page_content for doc in docs])
                        except Exception as e:
                            print(f"RAG Error: {e}")
            
            # Construct Prompt
            prompt_text = agent_config["system_prompt"]
            if context:
                prompt_text += f"\n\n参考资料（来自知识库）：\n{context}"
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", prompt_text),
                ("user", "{input}")
            ])
            chain = prompt | llm
            response = chain.invoke({"input": query})
            return {"final_answer": response.content, "next_node": "safety_agent"}
        
        return agent_node

    # Add each agent as a node
    for agent in agents_config:
        workflow.add_node(agent["name"], create_agent_node(agent))

    # 4. Define General Chat Node
    def general_chat_node(state: AgentState):
        messages = state["messages"]
        response = llm.invoke(messages)
        return {"final_answer": response.content, "next_node": "safety_agent"}
    
    workflow.add_node("general_chat", general_chat_node)

    # 5. Define Safety Agent Node
    def safety_agent_node(state: AgentState):
        answer = state.get("final_answer", "")
        prompt = ChatPromptTemplate.from_template(SAFETY_AGENT_PROMPT)
        chain = prompt | llm
        response = chain.invoke({"answer": answer})
        return {"messages": [AIMessage(content=response.content)]}

    workflow.add_node("safety_agent", safety_agent_node)

    # 6. Define Edges
    workflow.set_entry_point("router")
    
    # Conditional edges from router
    mapping = {a["name"]: a["name"] for a in agents_config}
    mapping["general_chat"] = "general_chat"
    
    workflow.add_conditional_edges(
        "router",
        lambda state: state["next_node"],
        mapping
    )
    
    # Edges to safety agent
    for agent in agents_config:
        workflow.add_edge(agent["name"], "safety_agent")
    workflow.add_edge("general_chat", "safety_agent")
    
    workflow.add_edge("safety_agent", END)
    
    return workflow.compile()
