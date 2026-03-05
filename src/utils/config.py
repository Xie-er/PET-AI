import json
import os
from typing import List, Dict, Optional

CONFIG_DIR = "config"
AGENTS_FILE = os.path.join(CONFIG_DIR, "agents.json")
KBS_FILE = os.path.join(CONFIG_DIR, "knowledge_bases.json")

# Default agents
DEFAULT_AGENTS = [
    {
        "name": "health_agent",
        "description": "处理宠物疾病、疫苗、体检、身体不适等健康相关问题。",
        "system_prompt": """你是一个专业的宠物健康顾问。你的任务是回答用户关于宠物健康、疾病、疫苗等方面的问题。
请基于你的专业知识和提供的参考资料（如果有）来回答。回答要准确、富有同理心，并建议用户在紧急情况下咨询兽医。""",
        "kb_id": "pet_health_kb" # Example linking
    },
    {
        "name": "diet_agent",
        "description": "处理宠物饲料、禁忌食物、营养需求、喂食建议等饮食相关问题。",
        "system_prompt": """你是一个专业的宠物营养专家。你的任务是回答用户关于宠物饮食、营养搭配、禁忌食物等方面的问题。
请提供科学的喂养建议。""",
        "kb_id": "pet_diet_kb"
    },
    {
        "name": "care_agent",
        "description": "处理宠物洗澡、美容、训练、居住环境、日常护理等问题。",
        "system_prompt": """你是一个专业的宠物护理专家。你的任务是回答用户关于宠物日常护理、美容、训练、环境等方面的问题。
请提供实用的护理技巧。""",
        "kb_id": "pet_care_kb"
    }
]

# Default KBs (Empty initially, user needs to upload)
DEFAULT_KBS = [
    {
        "id": "default_kb",
        "name": "默认宠物知识库",
        "description": "包含基础宠物知识",
        "persist_directory": "chroma_db/default"
    }
]

class ConfigManager:
    def __init__(self):
        if not os.path.exists(CONFIG_DIR):
            os.makedirs(CONFIG_DIR)
        
        self._ensure_file(AGENTS_FILE, DEFAULT_AGENTS)
        self._ensure_file(KBS_FILE, DEFAULT_KBS)

    def _ensure_file(self, filepath: str, default_content: list):
        if not os.path.exists(filepath):
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(default_content, f, ensure_ascii=False, indent=4)

    def get_agents(self) -> List[Dict]:
        with open(AGENTS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)

    def save_agents(self, agents: List[Dict]):
        with open(AGENTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(agents, f, ensure_ascii=False, indent=4)

    def add_agent(self, agent: Dict):
        agents = self.get_agents()
        agents.append(agent)
        self.save_agents(agents)

    def delete_agent(self, agent_name: str):
        agents = self.get_agents()
        agents = [a for a in agents if a['name'] != agent_name]
        self.save_agents(agents)
        
    def update_agent(self, old_name: str, new_agent: Dict):
        agents = self.get_agents()
        for i, agent in enumerate(agents):
            if agent['name'] == old_name:
                agents[i] = new_agent
                break
        self.save_agents(agents)

    def get_kbs(self) -> List[Dict]:
        with open(KBS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)

    def save_kbs(self, kbs: List[Dict]):
        with open(KBS_FILE, 'w', encoding='utf-8') as f:
            json.dump(kbs, f, ensure_ascii=False, indent=4)

    def add_kb(self, kb: Dict):
        kbs = self.get_kbs()
        # Check if ID exists
        if any(k['id'] == kb['id'] for k in kbs):
            raise ValueError(f"Knowledge Base ID {kb['id']} already exists.")
        kbs.append(kb)
        self.save_kbs(kbs)

    def delete_kb(self, kb_id: str):
        kbs = self.get_kbs()
        kbs = [k for k in kbs if k['id'] != kb_id]
        self.save_kbs(kbs)
    
    def get_kb_by_id(self, kb_id: str) -> Optional[Dict]:
        kbs = self.get_kbs()
        for k in kbs:
            if k['id'] == kb_id:
                return k
        return None

config_manager = ConfigManager()
