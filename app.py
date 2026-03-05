import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from src.agents.graph import app_graph

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Pet Encyclopedia AI", page_icon="🐾")

st.title("🐾 宠物百科智能问答系统")
st.markdown("""
欢迎使用基于多智能体的宠物问答系统！
我可以回答关于 **宠物健康**、**饮食习惯**、**日常护理** 等方面的问题。
""")

# Sidebar for configuration
with st.sidebar:
    st.header("设置")
    api_key = st.text_input("DashScope API Key", type="password", value=os.getenv("DASHSCOPE_API_KEY", ""))
    if api_key:
        os.environ["DASHSCOPE_API_KEY"] = api_key
    
    st.info("本系统使用阿里云通义千问大模型。")
    if not api_key:
        st.warning("请输入 API Key 以开始使用。")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)

# Chat input
if prompt := st.chat_input("请输入您的问题..."):
    # Add user message to chat history
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("human"):
        st.markdown(prompt)

    # Generate response
    if not os.environ.get("DASHSCOPE_API_KEY"):
        st.error("请先配置 DashScope API Key。")
    else:
        with st.chat_message("ai"):
            with st.spinner("正在思考中..."):
                try:
                    # Invoke the graph
                    inputs = {"messages": st.session_state.messages}
                    config = {"recursion_limit": 50}
                    
                    # Run the graph
                    # We only care about the final state
                    final_state = app_graph.invoke(inputs, config=config)
                    
                    # The safety agent updates the messages list with the final answer
                    response_content = final_state["messages"][-1].content
                    
                    st.markdown(response_content)
                    
                    # Add AI response to chat history
                    st.session_state.messages.append(AIMessage(content=response_content))
                    
                except Exception as e:
                    st.error(f"发生错误: {e}")
                    # If it's an API key error, give a hint
                    if "401" in str(e) or "InvalidApiKey" in str(e):
                        st.error("API Key 无效，请检查设置。")
