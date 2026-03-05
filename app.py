import streamlit as st
import os
import shutil
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from src.agents.dynamic_graph import build_graph
from src.utils.config import config_manager
from src.utils.rag import build_vector_store_from_file, delete_vector_store

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Pet Encyclopedia AI Platform", page_icon="🐾", layout="wide")

# Sidebar Navigation
st.sidebar.title("🐾 宠物百科 AI 平台")
page = st.sidebar.radio("导航", ["智能问答", "智能体管理", "知识库管理"])

# API Key Config
with st.sidebar:
    st.markdown("---")
    st.header("设置")
    api_key = st.text_input("DashScope API Key", type="password", value=os.getenv("DASHSCOPE_API_KEY", ""))
    if api_key:
        os.environ["DASHSCOPE_API_KEY"] = api_key
    
    st.info("本系统使用阿里云通义千问大模型。")
    if not api_key:
        st.warning("请输入 API Key 以开始使用。")

# ----------------------------------------------------------------------
# Page: Smart Chat
# ----------------------------------------------------------------------
if page == "智能问答":
    st.title("💬 智能问答")
    st.markdown("您可以向已配置的智能体提问。系统会自动路由您的问题。")
    
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
                        # Rebuild graph dynamically (to catch latest config)
                        app_graph = build_graph()
                        
                        # Invoke the graph
                        inputs = {"messages": st.session_state.messages}
                        config = {"recursion_limit": 50}
                        
                        # Run the graph
                        final_state = app_graph.invoke(inputs, config=config)
                        
                        # The safety agent updates the messages list with the final answer
                        response_content = final_state["messages"][-1].content
                        
                        st.markdown(response_content)
                        
                        # Add AI response to chat history
                        st.session_state.messages.append(AIMessage(content=response_content))
                        
                    except Exception as e:
                        st.error(f"发生错误: {e}")
                        if "401" in str(e) or "InvalidApiKey" in str(e):
                            st.error("API Key 无效，请检查设置。")

# ----------------------------------------------------------------------
# Page: Agent Management
# ----------------------------------------------------------------------
elif page == "智能体管理":
    st.title("🤖 智能体管理")
    st.markdown("在这里添加、修改或删除您的智能体。")
    
    # List existing agents
    agents = config_manager.get_agents()
    kbs = config_manager.get_kbs()
    kb_options = {kb['id']: kb['name'] for kb in kbs}
    kb_options[""] = "无知识库"
    
    # Add New Agent Form
    with st.expander("➕ 添加新智能体"):
        with st.form("add_agent_form"):
            new_name = st.text_input("智能体名称 (英文 ID)", help="例如: dog_training_agent")
            new_desc = st.text_input("描述", help="用于路由识别，例如: 处理狗狗训练问题")
            new_prompt = st.text_area("系统提示词 (System Prompt)", height=150, help="定义智能体的人设和任务")
            new_kb = st.selectbox("关联知识库", options=list(kb_options.keys()), format_func=lambda x: kb_options[x])
            
            submitted = st.form_submit_button("创建智能体")
            if submitted:
                if not new_name or not new_desc or not new_prompt:
                    st.error("请填写所有必填字段。")
                elif any(a['name'] == new_name for a in agents):
                    st.error("智能体名称已存在。")
                else:
                    new_agent = {
                        "name": new_name,
                        "description": new_desc,
                        "system_prompt": new_prompt,
                        "kb_id": new_kb if new_kb else None
                    }
                    config_manager.add_agent(new_agent)
                    st.success(f"智能体 {new_name} 创建成功！")
                    st.rerun()

    st.markdown("### 现有智能体列表")
    for agent in agents:
        with st.container():
            col1, col2 = st.columns([4, 1])
            with col1:
                st.subheader(f"🔹 {agent['name']}")
                st.write(f"**描述**: {agent['description']}")
                kb_name = kb_options.get(agent.get("kb_id", ""), "无知识库")
                st.write(f"**知识库**: {kb_name}")
                with st.expander("查看提示词"):
                    st.code(agent['system_prompt'])
            with col2:
                if st.button("删除", key=f"del_{agent['name']}"):
                    config_manager.delete_agent(agent['name'])
                    st.success(f"已删除 {agent['name']}")
                    st.rerun()
            st.markdown("---")

# ----------------------------------------------------------------------
# Page: Knowledge Base Management
# ----------------------------------------------------------------------
elif page == "知识库管理":
    st.title("📚 知识库管理")
    st.markdown("上传文档以构建知识库，供智能体调用。")
    
    kbs = config_manager.get_kbs()
    
    # Create KB Form
    with st.expander("➕ 创建新知识库"):
        with st.form("create_kb_form"):
            kb_id = st.text_input("知识库 ID (英文)", help="例如: cat_diseases")
            kb_name = st.text_input("知识库名称", help="例如: 猫咪疾病大全")
            kb_desc = st.text_input("描述")
            uploaded_file = st.file_uploader("上传文档 (TXT/PDF)", type=['txt', 'pdf'])
            
            submitted = st.form_submit_button("构建知识库")
            if submitted:
                if not kb_id or not kb_name or not uploaded_file:
                    st.error("请填写完整信息并上传文件。")
                elif any(k['id'] == kb_id for k in kbs):
                    st.error("知识库 ID 已存在。")
                else:
                    # Save file temporarily
                    temp_dir = "temp_uploads"
                    if not os.path.exists(temp_dir):
                        os.makedirs(temp_dir)
                    
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Build Vector Store
                    persist_dir = f"chroma_db/{kb_id}"
                    status_placeholder = st.empty()
                    status_placeholder.info("正在构建向量索引，请稍候...")
                    
                    try:
                        if not os.environ.get("DASHSCOPE_API_KEY"):
                             st.error("请先配置 API Key。")
                        else:
                            build_vector_store_from_file(file_path, persist_dir)
                            
                            # Save Config
                            new_kb = {
                                "id": kb_id,
                                "name": kb_name,
                                "description": kb_desc,
                                "persist_directory": persist_dir
                            }
                            config_manager.add_kb(new_kb)
                            
                            status_placeholder.success(f"知识库 {kb_name} 创建成功！")
                            # Cleanup
                            os.remove(file_path)
                            st.rerun()
                    except Exception as e:
                        status_placeholder.error(f"构建失败: {e}")

    st.markdown("### 现有知识库列表")
    for kb in kbs:
        with st.container():
            col1, col2 = st.columns([4, 1])
            with col1:
                st.subheader(f"📕 {kb['name']} ({kb['id']})")
                st.write(f"**描述**: {kb['description']}")
                st.write(f"**存储路径**: `{kb['persist_directory']}`")
            with col2:
                if st.button("删除", key=f"del_kb_{kb['id']}"):
                    # Delete actual files
                    delete_vector_store(kb['persist_directory'])
                    # Delete from config
                    config_manager.delete_kb(kb['id'])
                    st.success(f"已删除 {kb['name']}")
                    st.rerun()
            st.markdown("---")
