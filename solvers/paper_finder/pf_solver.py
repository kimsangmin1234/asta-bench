"""AstaBench Inspect solver — 자체 호스팅한 Paper Finder(무료 LLM 백엔드)를 평가한다.

원본 리포의 `astabench/solvers/search/paper_finder.py@ai2i_paper_finder` 는
현재 main 브랜치에서 삭제되었다. 이 파일이 그 대체물이다.

동작
----
    inspect eval
        -> 각 샘플마다 solve() 호출
        -> find_papers(query, base_url)  [HTTP]  -> Paper Finder 서버
        -> 응답의 doc_collection.documents 를 채점기 형식으로 변환
        -> state.output.completion 에 JSON 문자열로 주입

채점기 계약 (astabench/evals/paper_finder/paper_finder_utils.py:36-42)
---------------------------------------------------------------------
    raw = state.output.completion
    json_output = json.loads(raw)
    if "error" in json_output: raise ...
    output = json_output["output"]        # <- 최상위 "output" 키가 반드시 필요

    ExpectedAgentOutput = {
        "query_id": str,
        "results": [{"paper_id": CorpusId, "markdown_evidence": str}, ...]
    }

    * results 는 관련도 높은 순으로 정렬되어 있어야 한다.
    * 상위 250개(MAX_RESULTS_TO_CONSIDER)만 채점된다.

사용법
------
    export PAPER_FINDER_URL=http://localhost:8000

    inspect eval astabench/paper_finder_validation \
        -T with_search_tools=false \
        --solver solvers/paper_finder/pf_solver.py@free_paper_finder \
        -S base_url=$PAPER_FINDER_URL \
        --model openai/gpt-4o-mini \
        --max-samples 4
"""

from __future__ import annotations

import json
import logging
from typing import Any

from inspect_ai.model import ModelOutput
from inspect_ai.solver import Generate, Solver, TaskState, solver

from astabench.tools.paper_finder_ai2i import find_papers

logger = logging.getLogger(__name__)

# 채점기가 상위 250개만 본다 (paper_finder_utils.MAX_RESULTS_TO_CONSIDER)
MAX_RESULTS = 250

# Paper Finder 의 relevance_judgement 등급. 높을수록 관련성 높음.
# 서버가 이 값을 주면 정렬에 쓰고, 없으면 서버가 준 순서를 신뢰한다.
_REL_KEY = "relevance_judgement"


def _to_results(response: dict[str, Any]) -> list[dict[str, str]]:
    """Paper Finder 응답 -> 채점기가 먹는 results 리스트."""
    docs = (response.get("doc_collection") or {}).get("documents") or []

    # relevance_judgement 가 있으면 그것으로 내림차순 정렬한다.
    # Paper Finder 는 이미 정렬해서 주지만, 방어적으로 한 번 더 보장한다.
    if any(d.get(_REL_KEY) is not None for d in docs):
        docs = sorted(
            docs,
            key=lambda d: (d.get(_REL_KEY) if d.get(_REL_KEY) is not None else -1),
            reverse=True,
        )

    results: list[dict[str, str]] = []
    for d in docs[:MAX_RESULTS]:
        corpus_id = d.get("corpus_id")
        if not corpus_id:
            continue
        results.append(
            {
                "paper_id": str(corpus_id),
                # markdown 이 비면 채점기의 LLM judge 가 근거를 못 봐서 불리해진다.
                "markdown_evidence": d.get("markdown") or "",
            }
        )
    return results


@solver
def free_paper_finder(
    base_url: str,
    timeout: int = 600,
    read_results_from_cache: bool = False,
) -> Solver:
    """자체 호스팅 Paper Finder 를 호출하는 solver.

    Args:
        base_url: Paper Finder 서버 주소 (예: http://localhost:8000)
        timeout: 쿼리당 최대 대기(초). diligent 모드는 3분까지 갈 수 있다.
        read_results_from_cache: 서버측 캐시 사용 여부. 성능 측정을 여러 번
            반복할 때만 켠다. 기본은 꺼둔다 (매번 실제로 검색하게).
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        query = state.input_text
        query_id = str(state.sample_id)

        # 데이터 오염 방지 컷오프. task.py 가 set_insertion_date() 로 심어둔다.
        # (paper_finder 는 2025-06-01, litqa2 는 별도 날짜)
        inserted_before = (state.metadata or {}).get("insertion_date")
        if not inserted_before:
            logger.warning(
                "샘플 %s 에 insertion_date 가 없습니다. 컷오프 없이 검색하면 "
                "미래 논문이 섞여 점수가 부풀 수 있습니다.",
                query_id,
            )

        try:
            response = await find_papers(
                query,
                base_url=base_url,
                timeout=timeout,
                inserted_before=inserted_before,
                read_results_from_cache=read_results_from_cache,
            )
        except Exception as e:
            logger.error("Paper Finder 호출 실패 (%s): %s", query_id, e, exc_info=True)
            # 채점기는 "error" 키를 보면 예외를 던진다 -> 해당 샘플 0점.
            # 파이프라인 전체를 죽이지 않고 이 샘플만 실패 처리한다.
            state.output = ModelOutput.from_content(
                model="paper_finder",
                content=json.dumps({"error": f"{type(e).__name__}: {e}"}),
            )
            return state

        if response.get("error"):
            state.output = ModelOutput.from_content(
                model="paper_finder",
                content=json.dumps({"error": str(response["error"])}),
            )
            return state

        results = _to_results(response)
        logger.info("query_id=%s -> %d편 반환", query_id, len(results))

        state.output = ModelOutput.from_content(
            model="paper_finder",
            content=json.dumps(
                {"output": {"query_id": query_id, "results": results}},
                ensure_ascii=False,
            ),
        )
        return state

    return solve
