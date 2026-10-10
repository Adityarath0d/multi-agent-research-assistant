from typing import Annotated

from fastapi import Depends, Request

from multi_agent_research_assistant.config import Settings
from multi_agent_research_assistant.llm import LLMClient
from multi_agent_research_assistant.search import SearchClient


def get_app_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_search_client(request: Request) -> SearchClient:
    client: SearchClient = request.app.state.search
    return client


def get_llm_client(request: Request) -> LLMClient:
    client: LLMClient = request.app.state.llm
    return client


SettingsDep = Annotated[Settings, Depends(get_app_settings)]
SearchDep = Annotated[SearchClient, Depends(get_search_client)]
LLMDep = Annotated[LLMClient, Depends(get_llm_client)]
