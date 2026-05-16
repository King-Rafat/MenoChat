from abc import ABC, abstractmethod
import chromadb
from FlagEmbedding import BGEM3FlagModel
import json

# setup Chroma in-memory, for easy prototyping. Can add persistence easily!
class AIPlatform(ABC):
    @abstractmethod
    def chat(self, prompt: str) -> str:
        pass


class BGE(AIPlatform):
    def __init__(self, system_prompt):
        # self.api_key = api_key
        self.sys_prompt = system_prompt
        client = chromadb.PersistentClient(path = 'meno_store')

        # with open('general_health_questions_distinct.json', 'r', encoding = 'utf-8') as file:
        #     data1 = json.load(file)
        # with open('menstrual_menopausal_health_questions_distinct.json', 'r', encoding = 'utf-8') as file:
        #     data2 = json.load(file)
        self.collection = client.get_or_create_collection("MenoChat")
        self.model = BGEM3FlagModel('BAAI/bge-m3',  use_fp16=True)
    def chat(self, prompt: str) -> str:
        if self.sys_prompt:
            prompt = self.sys_prompt +" "+prompt #change if required
        q = self.model.encode(prompt)
        values = self.collection.query(query_embeddings = q['dense_vecs'])
        return values['documents'][0][0]