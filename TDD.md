# 智能医疗问诊助手 - 技术设计文档 (TDD)

## 1. 技术架构

### 1.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                        用户界面层                             │
│                   Streamlit 聊天界面                          │
│                   (端口 8501)                                 │
└────────────────────┬────────────────────────────────────────┘
                     │ HTTP POST /process
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                        API 服务层                             │
│                   FastAPI REST API                            │
│                   (端口 8000)                                 │
└────────────────────┬────────────────────────────────────────┘
                     │ 调用
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   LangGraph 多 Agent 系统                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │意图识别   │→│症状检测   │→│科室推荐   │→│安全检查   │   │
│  │Agent     │  │Agent     │  │Agent     │  │Agent     │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │用药建议   │  │医学知识   │  │就医指导   │  │答案融合   │   │
│  │Agent     │  │Agent     │  │Agent     │  │Agent     │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└──────────┬──────────────┬───────────────────────────────────┘
           │              │
           ▼              ▼
┌──────────────┐  ┌──────────────┐
│  FAISS       │  │  Neo4j       │
│  向量索引    │  │  知识图谱    │
│  (本地文件)  │  │  (Docker)    │
└──────────────┘  └──────────────┘
```

### 1.2 架构说明

**分层设计**：
1. **用户界面层**：Streamlit 聊天应用，负责用户交互
2. **API 服务层**：FastAPI REST API，负责请求路由和参数校验
3. **Agent 系统层**：LangGraph 多 Agent 编排，负责业务逻辑
4. **数据访问层**：Neo4j 知识图谱 + FAISS 向量索引，负责数据存储和检索

**设计原则**：
- 前后端分离：Streamlit 通过 HTTP 调用 FastAPI
- 模块化设计：每个 Agent 职责单一，易于测试和维护
- 可扩展性：新增 Agent 不影响现有系统
- 安全性：所有医疗建议强制添加免责声明

### 1.3 部署架构

**开发环境**：
```bash
# 1. 启动 Neo4j（Docker）
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/12345678 \
  -v neo4j_data:/data \
  neo4j:5.26

# 2. 启动 FastAPI（终端 1）
python scripts/start_api.py

# 3. 启动 Streamlit（终端 2）
streamlit run src/frontend/chat_app.py
```

**生产环境**（可选）：
```yaml
# docker-compose.yml
version: '3.8'
services:
  neo4j:
    image: neo4j:5.26
    ports:
      - "7474:7474"
      - "7687:7687"
    volumes:
      - neo4j_data:/data
  
  api:
    build: .
    ports:
      - "8000:8000"
    depends_on:
      - neo4j
  
  frontend:
    build: .
    ports:
      - "8501:8501"
    depends_on:
      - api

volumes:
  neo4j_data:
```

---

## 2. 模块设计

### 2.1 项目结构

```
medical-consultation-assistant/
├── README.md                          # 项目说明
├── PRD.md                             # 产品需求文档
├── TDD.md                             # 技术设计文档
├── .env                               # 环境变量配置
├── requirements.txt                   # Python 依赖
│
├── config/                            # 配置文件
│   ├── __init__.py
│   └── settings.py                    # 全局配置（从 .env 读取）
│
├── src/                               # 源代码目录
│   ├── __init__.py
│   │
│   ├── common/                        # 公共模块
│   │   ├── __init__.py
│   │   ├── llm.py                     # LLM 客户端封装
│   │   ├── neo4j_client.py            # Neo4j 数据库客户端
│   │   └── embedding_model.py         # 向量嵌入模型
│   │
│   ├── crawler/                       # 数据采集模块
│   │   ├── __init__.py
│   │   ├── medical_crawler.py         # 医疗网站爬虫
│   │   └── data_parser.py             # 数据解析器
│   │
│   ├── extraction/                    # 实体关系抽取模块
│   │   ├── __init__.py
│   │   ├── disease_extractor.py       # 疾病实体抽取
│   │   ├── symptom_extractor.py       # 症状实体抽取
│   │   ├── drug_extractor.py          # 药物实体抽取
│   │   ├── department_extractor.py    # 科室实体抽取
│   │   └── schemas/                   # Pydantic 模型定义
│   │       ├── __init__.py
│   │       ├── disease.py
│   │       ├── symptom.py
│   │       ├── drug.py
│   │       └── relation.py
│   │
│   ├── knowledge_graph/               # 知识图谱模块
│   │   ├── __init__.py
│   │   ├── schema.py                  # Neo4j Schema 定义
│   │   ├── graph_builder.py           # 图谱构建
│   │   └── graph_validator.py         # 图谱验证
│   │
│   ├── agents/                        # 多 Agent 系统模块
│   │   ├── __init__.py
│   │   ├── state.py                   # Agent 状态定义
│   │   ├── graph.py                   # LangGraph 编排
│   │   ├── base_agent.py              # Agent 基类
│   │   │
│   │   ├── intent/                    # 意图识别
│   │   │   ├── __init__.py
│   │   │   └── classifier.py          # 意图分类器
│   │   │
│   │   ├── detection/                 # 症状检测
│   │   │   ├── __init__.py
│   │   │   └── symptom_detector.py
│   │   │
│   │   ├── recommendation/            # 推荐系统
│   │   │   ├── __init__.py
│   │   │   ├── department_recommender.py
│   │   │   └── medication_advisor.py
│   │   │
│   │   ├── knowledge/                 # 知识问答
│   │   │   ├── __init__.py
│   │   │   └── medical_knowledge.py
│   │   │
│   │   ├── guidance/                  # 就医指导
│   │   │   ├── __init__.py
│   │   │   └── pre_visit_advisor.py
│   │   │
│   │   ├── safety/                    # 安全检查
│   │   │   ├── __init__.py
│   │   │   └── safety_checker.py
│   │   │
│   │   ├── fusion/                    # 答案融合
│   │   │   ├── __init__.py
│   │   │   └── answer_fusion.py
│   │   │
│   │   └── chat/                      # 通用对话
│   │       ├── __init__.py
│   │       └── general_chat.py
│   │
│   ├── vector_store/                  # 向量检索模块
│   │   ├── __init__.py
│   │   ├── vector_store.py            # 向量存储
│   │   ├── index_builder.py           # 索引构建
│   │   └── search.py                  # 搜索功能
│   │
│   ├── api/                           # API 服务模块
│   │   ├── __init__.py
│   │   ├── main.py                    # FastAPI 应用
│   │   ├── routes/                    # API 路由
│   │   │   ├── __init__.py
│   │   │   ├── consultation.py        # 问诊接口
│   │   │   └── health.py              # 健康检查
│   │   └── schemas/                   # 请求/响应模型
│   │       ├── __init__.py
│   │       ├── request.py
│   │       └── response.py
│   │
│   └── frontend/                      # 前端模块
│       ├── __init__.py
│       └── chat_app.py                # Streamlit 聊天应用
│
├── data/                              # 数据目录
│   ├── raw/                           # 原始数据
│   ├── processed/                     # 处理后的数据
│   └── indexes/                       # FAISS 索引
│
├── models/                            # 本地模型文件
│   └── bge-m3/                        # BGE-M3 嵌入模型
│
├── scripts/                           # 脚本目录
│   ├── run_crawler.py                 # 运行爬虫
│   ├── run_extraction.py              # 运行实体抽取
│   ├── build_graph.py                 # 构建知识图谱
│   ├── build_index.py                 # 构建向量索引
│   ├── start_api.py                   # 启动 API 服务
│   ├── start_frontend.py              # 启动前端
│   └── validate_environment.py        # 环境验证
│
└── tests/                             # 测试目录
    ├── unit/                          # 单元测试
    └── integration/                   # 集成测试
```

### 2.2 核心模块详细说明

#### 模块 1：公共模块 (`src/common/`)

**职责**：提供全局共享的基础功能

**核心类**：

```python
# config/settings.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """全局配置类"""
    # LLM 配置
    model_base_url: str
    model_api_key: str
    model_name: str
    
    # Neo4j 配置
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    
    # 嵌入模型配置
    embedding_model_path: str
    
    # 日志配置
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"

settings = Settings()
```

```python
# common/llm.py
from langchain_openai import ChatOpenAI
from config.settings import settings

def get_llm() -> ChatOpenAI:
    """获取 LLM 客户端"""
    return ChatOpenAI(
        api_key=settings.model_api_key,
        base_url=settings.model_base_url,
        model=settings.model_name,
        temperature=0.7
    )
```

```python
# common/neo4j_client.py
from neo4j import GraphDatabase
from config.settings import settings

class Neo4jClient:
    """Neo4j 数据库客户端（单例模式）"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password)
            )
        return cls._instance
    
    def run(self, query: str, params: dict = None):
        """执行 Cypher 查询"""
        with self.driver.session() as session:
            result = session.run(query, params or {})
            return [record.data() for record in result]
    
    def close(self):
        """关闭连接"""
        self.driver.close()

neo4j_client = Neo4jClient()
```

```python
# common/embedding_model.py
from sentence_transformers import SentenceTransformer
from config.settings import settings

class EmbeddingModel:
    """向量嵌入模型"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model = SentenceTransformer(settings.embedding_model_path)
        return cls._instance
    
    def encode(self, texts: list[str]) -> list[list[float]]:
        """文本向量化"""
        return self.model.encode(texts)

embedding_model = EmbeddingModel()
```

---

#### 模块 2：数据采集模块 (`src/crawler/`)

**职责**：从医疗网站采集原始数据

**核心类**：

```python
# crawler/medical_crawler.py
import requests
from bs4 import BeautifulSoup
from typing import List, Dict
import time

class MedicalCrawler:
    """医疗网站爬虫"""
    
    def __init__(self, base_url: str, headers: dict):
        self.base_url = base_url
        self.headers = headers
        self.session = requests.Session()
    
    def crawl_disease_list(self, page: int = 1) -> List[Dict]:
        """爬取疾病列表"""
        url = f"{self.base_url}/diseases?page={page}"
        response = self.session.get(url, headers=self.headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        diseases = []
        for item in soup.find_all('div', class_='disease-item'):
            diseases.append({
                'name': item.find('h3').text,
                'url': item.find('a')['href'],
                'description': item.find('p').text
            })
        
        time.sleep(1)  # 控制爬取频率
        return diseases
    
    def crawl_disease_detail(self, url: str) -> Dict:
        """爬取疾病详情"""
        response = self.session.get(url, headers=self.headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        return {
            'name': soup.find('h1').text,
            'symptoms': self._extract_symptoms(soup),
            'causes': self._extract_causes(soup),
            'treatments': self._extract_treatments(soup),
            'departments': self._extract_departments(soup)
        }
    
    def _extract_symptoms(self, soup) -> List[str]:
        """提取症状列表"""
        symptoms = []
        for item in soup.find_all('span', class_='symptom'):
            symptoms.append(item.text)
        return symptoms
```

**数据格式**：
```json
{
  "name": "高血压",
  "url": "https://example.com/disease/gaoxueya",
  "description": "高血压是指...",
  "symptoms": ["头痛", "头晕", "心悸"],
  "causes": ["遗传因素", "不良生活习惯"],
  "treatments": ["药物治疗", "生活方式调整"],
  "departments": ["心血管内科"]
}
```

---

#### 模块 3：实体关系抽取模块 (`src/extraction/`)

**职责**：使用 LLM 从原始文本中提取结构化实体和关系

**核心类**：

```python
# extraction/schemas/disease.py
from pydantic import BaseModel, Field
from typing import List

class DiseaseEntity(BaseModel):
    """疾病实体"""
    name: str = Field(description="疾病名称")
    symptoms: List[str] = Field(description="症状列表")
    causes: List[str] = Field(description="病因列表")
    treatments: List[str] = Field(description="治疗方法")
    departments: List[str] = Field(description="就诊科室")
    examinations: List[str] = Field(description="检查项目")
    body_parts: List[str] = Field(description="影响的身体部位")
    description: str = Field(description="疾病描述")
```

```python
# extraction/disease_extractor.py
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from src.common.llm import get_llm
from src.extraction.schemas.disease import DiseaseEntity

class DiseaseExtractor:
    """疾病实体抽取器"""
    
    def __init__(self):
        self.llm = get_llm()
        self.parser = PydanticOutputParser(pydantic_object=DiseaseEntity)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个医学信息抽取专家。请从以下文本中提取疾病信息，并按照指定格式输出。

输出格式：
{format_instructions}

要求：
1. 疾病名称使用标准医学术语
2. 症状列表要完整
3. 科室名称使用标准科室名称（如"心血管内科"而非"心内科"）
4. 如果没有相关信息，对应字段返回空列表"""),
            ("user", "{text}")
        ])
    
    def extract(self, text: str) -> DiseaseEntity:
        """从文本中抽取疾病实体"""
        formatted_prompt = self.prompt.format(
            text=text,
            format_instructions=self.parser.get_format_instructions()
        )
        response = self.llm.invoke(formatted_prompt)
        return self.parser.parse(response.content)
```

**提示词模板**：
```python
DISEASE_EXTRACTION_PROMPT = """
你是一个医学信息抽取专家。请从以下文本中提取疾病信息，并按照指定格式输出。

文本内容：
{text}

输出格式：
{format_instructions}

要求：
1. 疾病名称使用标准医学术语
2. 症状列表要完整
3. 科室名称使用标准科室名称（如"心血管内科"而非"心内科"）
4. 如果没有相关信息，对应字段返回空列表
"""
```

---

#### 模块 4：知识图谱模块 (`src/knowledge_graph/`)

**职责**：将结构化数据导入 Neo4j，构建医疗知识图谱

**Neo4j Schema 设计**：

**节点类型（8 种）**：
```cypher
// 疾病节点
(d:Disease {
    name: "高血压",
    description: "高血压是指...",
    created_at: timestamp()
})

// 症状节点
(s:Symptom {
    name: "头痛",
    description: "头部疼痛...",
    severity: "medium"  // low/medium/high
})

// 药物节点
(dr:Drug {
    name: "氨氯地平",
    category: "钙通道阻滞剂",
    side_effects: ["脚踝水肿", "面部潮红"],
    contraindications: ["严重低血压"]
})

// 科室节点
(dep:Department {
    name: "心血管内科",
    description: "专门治疗心脏和血管疾病"
})

// 检查节点
(e:Examination {
    name: "血压测量",
    purpose: "检测血压水平",
    preparation: "安静休息 5 分钟后测量"
})

// 治疗方案节点
(t:Treatment {
    name: "药物治疗",
    description: "使用药物控制血压"
})

// 身体部位节点
(bp:BodyPart {
    name: "心脏",
    description: "循环系统的核心器官"
})

// 医学概念节点
(mc:MedicalConcept {
    name: "血压",
    definition: "血液在血管中流动时对血管壁的压力"
})
```

**关系类型（12 种）**：
```cypher
// 疾病-症状关系
(d:Disease)-[:HAS_SYMPTOM {frequency: "common"}]->(s:Symptom)

// 症状-疾病关系（反向）
(s:Symptom)-[:MAY_INDICATE {probability: "medium"}]->(d:Disease)

// 疾病-科室关系
(d:Disease)-[:BELONG_TO_DEPARTMENT {priority: 1}]->(dep:Department)

// 疾病-药物关系
(d:Disease)-[:TREATED_BY_DRUG {evidence_level: "A"}]->(dr:Drug)

// 药物-疾病关系（反向）
(dr:Drug)-[:TREATS_DISEASE]->(d:Disease)

// 药物-副作用关系
(dr:Drug)-[:HAS_SIDE_EFFECT {frequency: "common"}]->(s:Symptom)

// 药物-药物相互作用
(dr1:Drug)-[:INTERACTS_WITH {severity: "high"}]->(dr2:Drug)

// 疾病-检查关系
(d:Disease)-[:NEEDS_EXAMINATION {necessity: "required"}]->(e:Examination)

// 疾病-身体部位关系
(d:Disease)-[:AFFECTS_BODY_PART]->(bp:BodyPart)

// 治疗方案-疾病关系
(t:Treatment)-[:FOR_DISEASE]->(d:Disease)

// 科室-疾病关系（反向）
(dep:Department)-[:HANDLES_DISEASE]->(d:Disease)

// 疾病-疾病相关关系
(d1:Disease)-[:RELATED_TO {relation_type: "complication"}]->(d2:Disease)
```

**核心类**：

```python
# knowledge_graph/graph_builder.py
from src.common.neo4j_client import neo4j_client
from typing import Dict

class GraphBuilder:
    """知识图谱构建器"""
    
    def create_indexes(self):
        """创建索引，加速查询"""
        neo4j_client.run("""
            CREATE INDEX disease_name IF NOT EXISTS
            FOR (d:Disease) ON (d.name)
        """)
        neo4j_client.run("""
            CREATE INDEX symptom_name IF NOT EXISTS
            FOR (s:Symptom) ON (s.name)
        """)
        neo4j_client.run("""
            CREATE INDEX drug_name IF NOT EXISTS
            FOR (dr:Drug) ON (dr.name)
        """)
    
    def merge_disease_node(self, disease_data: Dict):
        """创建或更新疾病节点"""
        query = """
        MERGE (d:Disease {name: $name})
        ON CREATE SET 
            d.description = $description,
            d.created_at = timestamp()
        ON MATCH SET
            d.description = $description,
            d.updated_at = timestamp()
        """
        neo4j_client.run(query, disease_data)
    
    def create_relationship(self, start_node: str, rel_type: str, end_node: str, properties: Dict = None):
        """创建关系"""
        query = f"""
        MATCH (a {{name: $start_name}})
        MATCH (b {{name: $end_name}})
        MERGE (a)-[r:{rel_type}]->(b)
        """
        if properties:
            set_clause = ", ".join([f"r.{k} = ${k}" for k in properties.keys()])
            query += f" ON CREATE SET {set_clause}"
        
        params = {
            "start_name": start_node,
            "end_name": end_node,
            **(properties or {})
        }
        neo4j_client.run(query, params)
```

---

#### 模块 5：向量检索模块 (`src/vector_store/`)

**职责**：使用 FAISS 实现语义搜索

**核心类**：

```python
# vector_store/vector_store.py
import faiss
import numpy as np
from src.common.embedding_model import embedding_model
import json

class VectorStore:
    """向量存储"""
    
    def __init__(self):
        self.index = None
        self.entities = []  # 存储实体名称
    
    def build_index(self, entities: list[dict]):
        """构建 FAISS 索引"""
        # entities: [{"name": "高血压", "type": "Disease", "description": "..."}]
        texts = [f"{e['name']} {e.get('description', '')}" for e in entities]
        embeddings = embedding_model.encode(texts)
        
        # 构建 FAISS 索引
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings))
        self.entities = entities
    
    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """语义搜索"""
        query_embedding = embedding_model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding), top_k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.entities):
                entity = self.entities[idx].copy()
                entity["score"] = float(1 / (1 + distance))  # 转换为相似度分数
                results.append(entity)
        
        return results
    
    def save_index(self, path: str):
        """保存索引到文件"""
        faiss.write_index(self.index, f"{path}/faiss.index")
        with open(f"{path}/entities.json", "w", encoding="utf-8") as f:
            json.dump(self.entities, f, ensure_ascii=False, indent=2)
    
    def load_index(self, path: str):
        """从文件加载索引"""
        self.index = faiss.read_index(f"{path}/faiss.index")
        with open(f"{path}/entities.json", "r", encoding="utf-8") as f:
            self.entities = json.load(f)
```

---

#### 模块 6：多 Agent 系统模块 (`src/agents/`)

**职责**：使用 LangGraph 编排多个 Agent，实现智能问答

**状态定义**：

```python
# agents/state.py
from typing import TypedDict, List, Optional

class AgentState(TypedDict):
    """Agent 状态"""
    user_id: str
    session_id: str
    user_input: str
    
    # 意图识别结果
    intent: Optional[str]  # "appointment" | "treatment" | "knowledge" | "other"
    
    # 症状检测结果
    symptoms: Optional[List[str]]
    severity: Optional[str]  # "low" | "medium" | "high"
    is_emergency: Optional[bool]
    
    # 科室推荐结果
    recommended_departments: Optional[List[dict]]
    
    # 用药建议结果
    medication_advice: Optional[dict]
    
    # 医学知识回答
    knowledge_answer: Optional[str]
    
    # 就医指导
    pre_visit_advice: Optional[List[str]]
    
    # 安全检查结果
    disclaimers: Optional[List[str]]
    warnings: Optional[List[str]]
    
    # 最终回答
    final_answer: Optional[str]
    
    # 中间消息（用于调试）
    messages: List[str]
```

**Agent 实现示例**：

```python
# agents/intent/classifier.py
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from src.common.llm import get_llm

class IntentResult(BaseModel):
    """意图识别结果"""
    intent: str = Field(description="意图类型：appointment/treatment/knowledge/other")
    confidence: float = Field(description="置信度：0-1")

class IntentClassifierAgent:
    """意图识别 Agent"""
    
    def __init__(self):
        self.llm = get_llm()
        self.parser = PydanticOutputParser(pydantic_object=IntentResult)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个医疗问诊意图识别专家。请判断用户的意图类型：

意图类型：
1. appointment - 看病挂号：用户描述症状，想知道该挂什么科
2. treatment - 治疗建议：用户询问治疗方法或用药
3. knowledge - 医学知识：用户咨询医学概念或知识
4. other - 其他问题：非医疗相关问题

输出格式：
{format_instructions}"""),
            ("user", "{user_input}")
        ])
    
    def __call__(self, state: dict) -> dict:
        """判断用户意图"""
        formatted_prompt = self.prompt.format(
            user_input=state['user_input'],
            format_instructions=self.parser.get_format_instructions()
        )
        response = self.llm.invoke(formatted_prompt)
        result = self.parser.parse(response.content)
        
        state["intent"] = result.intent
        state["messages"].append(f"意图识别：{result.intent} (置信度: {result.confidence})")
        
        return state
```

```python
# agents/detection/symptom_detector.py
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
from src.common.llm import get_llm

class SymptomResult(BaseModel):
    """症状检测结果"""
    symptoms: List[str] = Field(description="症状列表")
    severity: str = Field(description="严重程度：low/medium/high")
    is_emergency: bool = Field(description="是否为急症")

class SymptomDetectorAgent:
    """症状检测 Agent"""
    
    EMERGENCY_SYMPTOMS = [
        "胸痛", "呼吸困难", "意识丧失", "严重出血",
        "剧烈头痛", "突发视力丧失", "癫痫发作"
    ]
    
    def __init__(self):
        self.llm = get_llm()
        self.parser = PydanticOutputParser(pydantic_object=SymptomResult)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个医学症状检测专家。请从用户描述中提取症状信息：

输出格式：
{format_instructions}

急症症状列表：胸痛、呼吸困难、意识丧失、严重出血、剧烈头痛、突发视力丧失、癫痫发作
如果包含急症症状，is_emergency 设为 true"""),
            ("user", "{user_input}")
        ])
    
    def __call__(self, state: dict) -> dict:
        """检测症状"""
        formatted_prompt = self.prompt.format(
            user_input=state['user_input'],
            format_instructions=self.parser.get_format_instructions()
        )
        response = self.llm.invoke(formatted_prompt)
        result = self.parser.parse(response.content)
        
        state["symptoms"] = result.symptoms
        state["severity"] = result.severity
        state["is_emergency"] = result.is_emergency
        state["messages"].append(f"症状检测：{result.symptoms} (严重程度: {result.severity})")
        
        return state
```

**LangGraph 编排**：

```python
# agents/graph.py
from langgraph.graph import StateGraph, END
from src.agents.state import AgentState
from src.agents.intent.classifier import IntentClassifierAgent
from src.agents.detection.symptom_detector import SymptomDetectorAgent
from src.agents.recommendation.department_recommender import DepartmentRecommenderAgent
from src.agents.recommendation.medication_advisor import MedicationAdvisorAgent
from src.agents.knowledge.medical_knowledge import MedicalKnowledgeAgent
from src.agents.guidance.pre_visit_advisor import PreVisitAdvisorAgent
from src.agents.safety.safety_checker import SafetyCheckerAgent
from src.agents.fusion.answer_fusion import AnswerFusionAgent
from src.agents.chat.general_chat import GeneralChatAgent

def create_medical_qa_graph():
    """创建医疗问答状态图"""
    workflow = StateGraph(AgentState)
    
    # 添加节点（Agent）
    workflow.add_node("intent_classifier", IntentClassifierAgent())
    workflow.add_node("symptom_detector", SymptomDetectorAgent())
    workflow.add_node("department_recommender", DepartmentRecommenderAgent())
    workflow.add_node("medication_advisor", MedicationAdvisorAgent())
    workflow.add_node("medical_knowledge", MedicalKnowledgeAgent())
    workflow.add_node("pre_visit_advisor", PreVisitAdvisorAgent())
    workflow.add_node("safety_checker", SafetyCheckerAgent())
    workflow.add_node("answer_fusion", AnswerFusionAgent())
    workflow.add_node("general_chat", GeneralChatAgent())
    
    # 设置入口
    workflow.set_entry_point("intent_classifier")
    
    # 添加条件边（根据意图路由）
    workflow.add_conditional_edges(
        "intent_classifier",
        lambda state: state["intent"],
        {
            "appointment": "symptom_detector",
            "treatment": "symptom_detector",
            "knowledge": "medical_knowledge",
            "other": "general_chat"
        }
    )
    
    # 看病挂号流程
    workflow.add_edge("symptom_detector", "department_recommender")
    workflow.add_edge("department_recommender", "pre_visit_advisor")
    workflow.add_edge("pre_visit_advisor", "safety_checker")
    workflow.add_edge("safety_checker", "answer_fusion")
    
    # 治疗建议流程
    workflow.add_conditional_edges(
        "symptom_detector",
        lambda state: state["intent"],
        {
            "appointment": "department_recommender",
            "treatment": "medication_advisor"
        }
    )
    workflow.add_edge("medication_advisor", "safety_checker")
    
    # 医学知识流程
    workflow.add_edge("medical_knowledge", "safety_checker")
    
    # 通用对话直接结束
    workflow.add_edge("general_chat", END)
    
    # 答案融合后结束
    workflow.add_edge("answer_fusion", END)
    
    # 编译状态图
    app = workflow.compile()
    return app

def call_langgraph_ai(user_id: str, session_id: str, user_content: str) -> str:
    """调用 LangGraph AI 系统"""
    app = create_medical_qa_graph()
    
    initial_state = {
        "user_id": user_id,
        "session_id": session_id,
        "user_input": user_content,
        "messages": []
    }
    
    final_state = app.invoke(initial_state)
    return final_state["final_answer"]
```

---

#### 模块 7：API 服务模块 (`src/api/`)

**职责**：提供 REST API 接口

**核心代码**：

```python
# api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes.consultation import router as consultation_router
from src.api.routes.health import router as health_router

app = FastAPI(
    title="智能医疗问诊助手",
    version="1.0.0",
    description="基于知识图谱的医疗咨询系统"
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(consultation_router, prefix="/api", tags=["问诊"])
app.include_router(health_router, tags=["健康检查"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

```python
# api/routes/consultation.py
from fastapi import APIRouter
from src.api.schemas.request import ConsultationRequest
from src.api.schemas.response import ConsultationResponse
from src.agents.graph import call_langgraph_ai

router = APIRouter()

@router.post("/process", response_model=ConsultationResponse)
def process_consultation(req: ConsultationRequest):
    """处理用户问诊请求"""
    answer = call_langgraph_ai(
        user_id=req.data.user_id,
        session_id=req.data.session_id,
        user_content=req.data.user_content
    )
    
    return ConsultationResponse(
        result={
            "answer": answer,
            "timestamp": datetime.now().isoformat()
        }
    )
```

```python
# api/schemas/request.py
from pydantic import BaseModel
from typing import Dict, Any

class ConsultationData(BaseModel):
    user_id: str
    session_id: str
    user_content: str

class ConsultationRequest(BaseModel):
    data: ConsultationData
```

```python
# api/schemas/response.py
from pydantic import BaseModel
from typing import Dict, Any

class ConsultationResponse(BaseModel):
    result: Dict[str, Any]
```

---

#### 模块 8：前端模块 (`src/frontend/`)

**职责**：提供用户友好的聊天界面

**核心代码**：

```python
# frontend/chat_app.py
import streamlit as st
import requests
import uuid

# 页面配置
st.set_page_config(
    page_title="智能医疗问诊助手",
    page_icon="🏥",
    layout="wide"
)

# 添加免责声明横幅
st.warning("""
⚠️ **重要声明**：本系统仅供健康咨询和就医指导，不能替代专业医疗诊断。
身体不适一定要去正规医院找医生看病，千万不能自行用药或治疗！
""")

# 初始化聊天记录
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 用户输入
if prompt := st.chat_input("请描述您的症状或健康问题..."):
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 获取 AI 回答
    with st.chat_message("assistant"):
        with st.spinner("AI 正在分析您的问题..."):
            # 调用 FastAPI 后端
            url = "http://127.0.0.1:8000/api/process"
            payload = {
                "data": {
                    "user_id": "user_01",
                    "session_id": str(uuid.uuid4()),
                    "user_content": prompt
                }
            }
            response = requests.post(url, json=payload)
            answer = response.json()["result"]["answer"]
            
            st.markdown(answer)
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })
```

---

## 3. 数据流

### 3.1 整体数据流

```
医疗网站数据
    │ requests + BeautifulSoup（爬虫）
    ▼
Excel + JSON 文件（原始数据）
    │ LangChain + LLM（实体抽取）
    ▼
JSON 文件（结构化数据）
    │ Neo4j + Cypher（图谱构建）
    ▼
Neo4j 知识图谱（8 种节点，12 种关系）
    │ LangGraph + FAISS（智能问答）
    ▼
FastAPI（REST API）
    │ HTTP 请求
    ▼
Streamlit（聊天界面）→ 用户
```

### 3.2 详细数据流

#### 流程 1：数据采集与处理

```
1. 爬虫采集
   输入：医疗网站 URL
   处理：requests + BeautifulSoup 解析 HTML
   输出：原始数据（Excel + JSON）
   存储：data/raw/

2. 实体抽取
   输入：原始数据（txt/json）
   处理：LangChain + LLM 提取结构化信息
   输出：结构化数据（JSON）
   存储：data/processed/

3. 图谱构建
   输入：结构化数据（JSON）
   处理：Cypher MERGE 语句创建节点和关系
   输出：Neo4j 知识图谱
   存储：Neo4j 数据库

4. 向量索引构建
   输入：Neo4j 中的实体数据
   处理：BGE-M3 生成向量，FAISS 构建索引
   输出：FAISS 索引文件
   存储：data/indexes/
```

#### 流程 2：智能问答

```
用户输入
    │
    ▼
[Streamlit 前端]
    │ HTTP POST /api/process
    ▼
[FastAPI 后端]
    │ 调用 call_langgraph_ai()
    ▼
[LangGraph Agent 系统]
    │
    ├─→ [意图识别 Agent]
    │     输入：用户问题
    │     输出：意图类型（appointment/treatment/knowledge/other）
    │
    ├─→ [症状检测 Agent]
    │     输入：用户问题
    │     输出：症状列表、严重程度、是否急症
    │
    ├─→ [科室推荐 Agent]
    │     输入：症状列表
    │     处理：FAISS 向量搜索 + Neo4j 图谱查询
    │     输出：推荐科室列表
    │
    ├─→ [用药建议 Agent]
    │     输入：疾病名称
    │     处理：Neo4j 图谱查询
    │     输出：药物信息、副作用、注意事项
    │
    ├─→ [医学知识 Agent]
    │     输入：用户问题
    │     处理：Neo4j 图谱查询
    │     输出：医学知识回答
    │
    ├─→ [就医指导 Agent]
    │     输入：症状、疾病
    │     输出：就医前注意事项
    │
    ├─→ [安全检查 Agent]
    │     输入：所有建议
    │     输出：免责声明、警告信息
    │
    └─→ [答案融合 Agent]
          输入：所有 Agent 的输出
          处理：整合信息，格式化输出
          输出：最终回答
    │
    ▼
[FastAPI 返回响应]
    │ HTTP Response
    ▼
[Streamlit 显示回答]
    │
    ▼
用户看到回答
```

### 3.3 状态流转图

```
初始状态
    │
    ▼
user_input: "我最近总是头疼"
intent: null
symptoms: null
...
    │
    ▼ [意图识别 Agent]
    │
intent: "appointment"
messages: ["意图识别：appointment"]
    │
    ▼ [症状检测 Agent]
    │
symptoms: ["头疼"]
severity: "medium"
is_emergency: false
messages: [..., "症状检测：['头疼']"]
    │
    ▼ [科室推荐 Agent]
    │
recommended_departments: [
    {"name": "神经内科", "priority": "首选", "reason": "..."},
    {"name": "眼科", "priority": "备选", "reason": "..."}
]
messages: [..., "科室推荐：神经内科、眼科"]
    │
    ▼ [就医指导 Agent]
    │
pre_visit_advice: [
    "记录头痛的时间、频率",
    "就医前不要自行服用止痛药"
]
messages: [..., "就医指导：已生成"]
    │
    ▼ [安全检查 Agent]
    │
disclaimers: ["本建议仅供参考，不能替代专业医疗诊断..."]
warnings: []
messages: [..., "安全检查：已添加免责声明"]
    │
    ▼ [答案融合 Agent]
    │
final_answer: "根据您的症状描述，建议您就诊以下科室：..."
messages: [..., "答案融合：完成"]
    │
    ▼
返回最终回答
```

---

## 4. 技术选型

### 4.1 技术栈总览

| 层级 | 技术 | 版本 | 用途 | 选型理由 |
|------|------|------|------|---------|
| 网页爬虫 | requests + BeautifulSoup | 最新 | 数据采集 | 简单易用，适合小规模爬取 |
| 数据处理 | pandas + openpyxl | 最新 | Excel 读写 | 功能强大，API 友好 |
| 大语言模型 | langchain + ChatOpenAI | 最新 | LLM 调用 | 统一接口，支持多种 LLM |
| 结构化输出 | pydantic | v2 | 数据校验 | 类型安全，自动生成文档 |
| 图数据库 | Neo4j | 5.26 LTS | 知识图谱存储 | 图数据库领导者，Cypher 语法简洁 |
| 向量检索 | sentence-transformers + FAISS | 最新 | 语义搜索 | BGE-M3 多语言支持，FAISS 高效 |
| Agent 编排 | LangGraph | 最新 | 多 Agent 状态图 | LangChain 生态，易于扩展 |
| Web API | FastAPI + Uvicorn | 最新 | REST 接口 | 高性能，自动生成文档 |
| 前端界面 | Streamlit | 最新 | 聊天界面 | 快速开发，无需前端知识 |
| 环境配置 | python-dotenv + pydantic-settings | 最新 | 环境变量 | 类型安全，配置验证 |

### 4.2 技术选型详细说明

#### 大语言模型：阿里云 DashScope API (qwen-plus)

**选型理由**：
- 中文能力强，适合医疗领域
- 成本相对较低
- API 兼容 OpenAI 格式，易于切换

**替代方案**：
- OpenAI GPT-4：性能更好，但成本高
- 本地部署（如 LLaMA）：隐私性好，但需要 GPU

#### 图数据库：Neo4j 5.26 LTS

**选型理由**：
- 图数据库领导者，市场占有率最高
- Cypher 查询语言简洁易懂
- 社区版免费，适合教学项目
- Docker 部署简单

**替代方案**：
- JanusGraph：开源，但配置复杂
- Amazon Neptune：云服务，但成本高

#### 向量检索：FAISS + BGE-M3

**选型理由**：
- FAISS：Facebook 开发，高效稳定
- BGE-M3：北京智源研究院开发，多语言支持，中文效果好
- 本地部署，无需网络请求

**替代方案**：
- Milvus：分布式向量数据库，适合大规模
- Pinecone：云服务，但需要网络

#### Agent 编排：LangGraph

**选型理由**：
- LangChain 生态，与现有工具无缝集成
- 状态图设计，逻辑清晰
- 支持条件边，灵活路由

**替代方案**：
- AutoGen：微软开发，多 Agent 对话
- CrewAI：角色扮演，但不够灵活

#### Web API：FastAPI

**选型理由**：
- 高性能（基于 Starlette）
- 自动生成 OpenAPI 文档
- 类型安全（基于 Pydantic）

**替代方案**：
- Flask：简单易用，但性能较低
- Django：功能全面，但较重

#### 前端界面：Streamlit

**选型理由**：
- 纯 Python 开发，无需前端知识
- 快速原型开发
- 内置聊天组件

**替代方案**：
- Gradio：类似 Streamlit，但组件较少
- React + Next.js：功能强大，但需要前端知识

---

## 5. 接口设计

### 5.1 模块间接口

#### 接口 1：爬虫 → 实体抽取

**输入**：
```python
# 输入：Excel 文件路径
input_file = "data/raw/diseases.xlsx"
```

**输出**：
```python
# 输出：JSON 文件
output_file = "data/processed/disease_entities.json"

# 输出格式
[
    {
        "name": "高血压",
        "symptoms": ["头痛", "头晕", "心悸"],
        "causes": ["遗传因素", "不良生活习惯"],
        "treatments": ["药物治疗", "生活方式调整"],
        "departments": ["心血管内科"]
    }
]
```

#### 接口 2：实体抽取 → 图谱构建

**输入**：
```python
# 输入：JSON 文件
input_file = "data/processed/disease_entities.json"
```

**输出**：
```cypher
// 输出：Neo4j 数据库
MERGE (d:Disease {name: "高血压"})
MERGE (s:Symptom {name: "头痛"})
MERGE (d)-[:HAS_SYMPTOM]->(s)
```

#### 接口 3：图谱构建 → 向量检索

**输入**：
```python
# 从 Neo4j 读取实体
entities = neo4j_client.run("""
    MATCH (d:Disease)
    RETURN d.name AS name, d.description AS description
""")

# entities: [{"name": "高血压", "description": "..."}]
```

**输出**：
```python
# 构建 FAISS 索引
vector_store.build_index(entities)
vector_store.save_index("data/indexes/")
```

#### 接口 4：向量检索 → Agent 系统

**输入**：
```python
# Agent 调用向量搜索
query = "头疼"
results = vector_store.search(query, top_k=5)
# results: [{"name": "头痛", "score": 0.95}, ...]
```

**输出**：
```python
# 返回搜索结果
return results
```

### 5.2 API 接口

#### 接口 1：POST /api/process

**请求**：
```json
{
  "data": {
    "user_id": "user_01",
    "session_id": "session_01",
    "user_content": "我最近总是头疼，应该挂什么科？"
  }
}
```

**响应**：
```json
{
  "result": {
    "answer": "根据您的症状描述，建议您就诊以下科室：...",
    "timestamp": "2026-07-24T10:30:00"
  }
}
```

**错误响应**：
```json
{
  "error": {
    "code": 400,
    "message": "用户输入不能为空"
  }
}
```

#### 接口 2：GET /health

**请求**：
```
GET /health
```

**响应**：
```json
{
  "status": "ok",
  "version": "1.0.0"
}
```

### 5.3 Agent 间接口

#### Agent 状态传递

**输入状态**：
```python
state = {
    "user_id": "user_01",
    "session_id": "session_01",
    "user_input": "我最近总是头疼",
    "intent": None,
    "symptoms": None,
    "messages": []
}
```

**输出状态**（经过意图识别 Agent）：
```python
state = {
    "user_id": "user_01",
    "session_id": "session_01",
    "user_input": "我最近总是头疼",
    "intent": "appointment",
    "symptoms": None,
    "messages": ["意图识别：appointment"]
}
```

**状态流转**：
```python
# 每个 Agent 接收 state，修改 state，返回 state
def __call__(self, state: dict) -> dict:
    # 处理逻辑
    state["intent"] = "appointment"
    state["messages"].append("意图识别：appointment")
    return state
```

---

## 6. 安全设计

### 6.1 免责声明

**强制添加**：所有医疗建议都必须包含免责声明

```python
# agents/safety/safety_checker.py
DISCLAIMER = """
🔴 **重要声明**：
本建议仅供参考，不能替代专业医疗诊断。身体不适一定要去正规医院找医生看病，
千万不要自行用药或治疗！如有紧急情况，请立即就医或拨打急救电话（120）。
"""

class SafetyCheckerAgent:
    def __call__(self, state: dict) -> dict:
        state["disclaimers"] = [DISCLAIMER]
        return state
```

### 6.2 急症识别

**急症症状列表**：
```python
EMERGENCY_SYMPTOMS = [
    "胸痛", "呼吸困难", "意识丧失", "严重出血",
    "剧烈头痛", "突发视力丧失", "癫痫发作"
]

def check_emergency(symptoms: List[str]) -> bool:
    """检查是否包含急症症状"""
    return any(s in EMERGENCY_SYMPTOMS for s in symptoms)
```

**急症响应**：
```markdown
🚨 **紧急提醒**：

您描述的症状可能是急症表现，请立即就医或拨打急救电话（120）！
```

### 6.3 药物相互作用检查

```python
def check_drug_interactions(drugs: List[str]) -> List[dict]:
    """检查药物相互作用"""
    query = """
    MATCH (d1:Drug)-[:INTERACTS_WITH]->(d2:Drug)
    WHERE d1.name IN $drugs AND d2.name IN $drugs
    RETURN d1.name AS drug1, d2.name AS drug2, r.severity AS severity
    """
    return neo4j_client.run(query, {"drugs": drugs})
```

---

## 7. 测试设计

### 7.1 单元测试

```python
# tests/unit/test_extractor.py
import pytest
from src.extraction.disease_extractor import DiseaseExtractor

def test_disease_extractor():
    extractor = DiseaseExtractor()
    text = "高血压是一种常见的慢性病，症状包括头痛、头晕..."
    result = extractor.extract(text)
    assert result.name == "高血压"
    assert "头痛" in result.symptoms
```

### 7.2 集成测试

```python
# tests/integration/test_agents.py
import pytest
from src.agents.graph import call_langgraph_ai

def test_medical_qa_system():
    answer = call_langgraph_ai(
        user_id="test_user",
        session_id="test_session",
        user_content="我最近总是头疼，应该挂什么科？"
    )
    assert "神经内科" in answer or "科室" in answer
    assert "免责声明" in answer or "就医" in answer
```

### 7.3 端到端测试

```python
# tests/e2e/test_api.py
import pytest
import requests

def test_api_endpoint():
    response = requests.post(
        "http://127.0.0.1:8000/api/process",
        json={
            "data": {
                "user_id": "test",
                "session_id": "test",
                "user_content": "高血压吃什么药？"
            }
        }
    )
    assert response.status_code == 200
    assert "result" in response.json()
```

---

## 8. 部署方案

### 8.1 开发环境

```bash
# 1. 启动 Neo4j
docker start neo4j

# 2. 启动 FastAPI
python scripts/start_api.py

# 3. 启动 Streamlit
streamlit run src/frontend/chat_app.py
```

### 8.2 生产环境

**Docker Compose**：
```yaml
version: '3.8'
services:
  neo4j:
    image: neo4j:5.26
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/12345678
    volumes:
      - neo4j_data:/data
  
  api:
    build: .
    ports:
      - "8000:8000"
    depends_on:
      - neo4j
    environment:
      - NEO4J_URI=bolt://neo4j:7687
  
  frontend:
    build: .
    ports:
      - "8501:8501"
    depends_on:
      - api

volumes:
  neo4j_data:
```

**启动命令**：
```bash
docker-compose up -d
```

---

## 9. 性能优化

### 9.1 向量检索优化

**批量构建索引**：
```python
# 批量处理，减少内存占用
batch_size = 1000
for i in range(0, len(entities), batch_size):
    batch = entities[i:i+batch_size]
    vector_store.build_index(batch)
```

**缓存搜索结果**：
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def search_cached(query: str, top_k: int = 5):
    return vector_store.search(query, top_k)
```

### 9.2 Neo4j 查询优化

**创建索引**：
```cypher
CREATE INDEX disease_name FOR (d:Disease) ON (d.name);
CREATE INDEX symptom_name FOR (s:Symptom) ON (s.name);
```

**使用 EXPLAIN 分析查询**：
```cypher
EXPLAIN MATCH (d:Disease {name: "高血压"})-[:HAS_SYMPTOM]->(s:Symptom)
RETURN s.name
```

---

**文档版本**：v1.0  
**创建日期**：2026-07-24  
**最后更新**：2026-07-24  
**状态**：已完成 ✅
