import logging
from dataclasses import asdict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from multi_agent_research_assistant.api.deps import LLMDep, SearchDep
from multi_agent_research_assistant.llm import LLMCallError, LLMNotConfiguredError
from multi_agent_research_assistant.pipelines.baseline import run_baseline
from multi_agent_research_assistant.search import (
    SearchCallError,
    SearchNotConfiguredError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/research", tags=["research"])


class ResearchRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=3, max_length=500)


class SourceOut(BaseModel):
    index: int
    title: str
    url: str


class ResearchResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceOut]
    model: str
    prompt_tokens: int
    completion_tokens: int
    search_seconds: float
    llm_seconds: float
    total_seconds: float


@router.post("/baseline", response_model=ResearchResponse)
async def baseline(
    body: ResearchRequest,
    search: SearchDep,
    llm: LLMDep,
) -> ResearchResponse:
    """Temporary single-agent endpoint. Replaced by the async job API in Phase 6."""
    try:
        result = await run_baseline(body.question, search, llm)
    except (SearchNotConfiguredError, LLMNotConfiguredError) as exc:
        logger.error("research_not_configured error=%s", type(exc).__name__)
        raise HTTPException(
            status_code=503, detail="The service is not configured correctly."
        ) from exc
    except (SearchCallError, LLMCallError) as exc:
        logger.warning("research_upstream_failed error=%s", type(exc).__name__)
        raise HTTPException(
            status_code=502, detail="An upstream provider failed. Please retry."
        ) from exc
    return ResearchResponse(**asdict(result))
