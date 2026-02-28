from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import chromadb
from openai import OpenAI
from sentence_transformers import SentenceTransformer, CrossEncoder


@dataclass(frozen=True)
class AppConfig:
    chroma_path: str
    collection_name: str
    emb_model_name: str
    rerank_model_name: str
    openai_base_url: str
    openai_api_key: str
    safety: bool


class App:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

        # Models
        self.emb_model = SentenceTransformer(cfg.emb_model_name)
        self.reranker = CrossEncoder(cfg.rerank_model_name)

        # Chroma
        self.chroma_client = chromadb.PersistentClient(path=cfg.chroma_path)
        self.collection = self.chroma_client.get_collection(cfg.collection_name)

        self.openai_client = OpenAI(
            base_url=cfg.openai_base_url,
            api_key=cfg.openai_api_key,
        )

    # def __init__(self, cfg: AppConfig):
    #     self.cfg = cfg
    #
    #     self._emb_model: Optional[SentenceTransformer] = None
    #     self._reranker: Optional[CrossEncoder] = None
    #     self._chroma_client: Optional[chromadb.PersistentClient] = None
    #     self._openai_client: Optional[OpenAI] = None
    #     self._collection = None
    #
    # @property
    # def emb_model(self) -> SentenceTransformer:
    #     if self._emb_model is None:
    #         self._emb_model = SentenceTransformer(self.cfg.emb_model_name)
    #     return self._emb_model
    #
    # @property
    # def reranker(self) -> CrossEncoder:
    #     if self._reranker is None:
    #         self._reranker = CrossEncoder(self.cfg.rerank_model_name)
    #     return self._reranker
    #
    # @property
    # def chroma_client(self) -> chromadb.PersistentClient:
    #     if self._chroma_client is None:
    #         self._chroma_client = chromadb.PersistentClient(path=self.cfg.chroma_path)
    #     return self._chroma_client
    #
    # @property
    # def collection(self):
    #     if self._collection is None:
    #         self._collection = self.chroma_client.get_collection(self.cfg.collection_name)
    #     return self._collection
    #
    # @property
    # def openai_client(self):
    #     if self._openai_client is None:
    #         self._openai_client = OpenAI(
    #             base_url=self.cfg.openai_base_url,
    #             api_key=self.cfg.openai_api_key,
    #         )
    #     return self._openai_client

