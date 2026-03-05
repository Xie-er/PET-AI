# 基于多智能体的宠物百科问答系统 (Multi-Agent Pet Encyclopedia QA System)

## 1. 项目背景 (Project Background)
本项目旨在设计并实现一个基于多智能体（Multi-Agent）架构的宠物百科问答系统。系统针对宠物健康、饮食习惯、日常护理等领域，通过协作的智能体提供准确、专业的回答。该项目作为计算机科学与技术专业的毕业设计。

## 2. 系统架构 (System Architecture)

系统采用 **多智能体协作 (Multi-Agent Collaboration)** 架构，主要包含以下核心组件：

### 2.1 核心组件
1.  **用户界面 (User Interface)**: 
    - 使用 Streamlit 构建，提供友好的聊天界面。
    - 用户输入自然语言问题。

2.  **主智能体/路由智能体 (Router Agent)**:
    - 负责理解用户意图。
    - 将问题分发给最合适的专家智能体 (健康、饮食、护理)。

3.  **专家智能体 (Specialized Agents)**:
    - **健康顾问 (Health Agent)**: 专注于宠物疾病、疫苗、体检等问题。
    - **营养专家 (Diet Agent)**: 专注于宠物饲料、禁忌食物、营养搭配。
    - **护理专家 (Care Agent)**: 专注于洗澡、美容、训练、居住环境。

4.  **内容安全智能体 (Safety Agent)**:
    - 负责对输出内容进行审核。
    - 确保回答内容健康、安全、无害。

5.  **知识库 (Knowledge Base - RAG)**:
    - 包含宠物相关的专业文档。
    - 使用向量数据库 (ChromaDB) 进行存储和检索。

## 3. 技术栈 (Tech Stack)
- **编程语言**: Python 3.10+
- **大模型框架**: LangChain / LangGraph
- **界面框架**: Streamlit
- **向量数据库**: ChromaDB
- **大语言模型**: Alibaba Cloud DashScope (通义千问)

## 4. 快速开始 (Quick Start)

### 4.1 环境配置
1.  克隆项目到本地。
2.  安装依赖：
    ```bash
    pip install -r requirements.txt
    ```
3.  配置 API Key：
    - 复制 `.env` 文件并填入您的 DashScope API Key。
    - 或者在运行界面侧边栏输入。

### 4.2 运行系统
```bash
streamlit run app.py
```

## 5. 目录结构 (Directory Structure)
```
PET-AI/
├── data/               # 存放知识库文档
├── src/
│   ├── agents/         # 智能体实现代码 (graph.py, prompts.py)
│   ├── chains/         # RAG 检索链
│   ├── utils/          # 工具函数 (rag.py)
│   └── main.py         # 系统入口
├── app.py              # Streamlit 界面入口
├── requirements.txt    # 项目依赖
└── README.md           # 项目说明
```
