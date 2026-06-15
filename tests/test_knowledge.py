"""Medical-info RAG: offline retrieval, citations, grounding pre-filter, the tool."""

import auth
import knowledge
import tools


def test_search_retrieves_relevant_passage_with_citation():
    res = knowledge.search("chronic kidney disease")
    assert res["passages"]
    assert "kidney" in res["passages"][0]["content"].lower()
    assert res["citations"][0]["url"].startswith("https://medlineplus.gov")
    assert res["live"] is False  # USE_LIVE_SEARCH off in tests


def test_search_returns_nothing_for_unrelated_query():
    assert knowledge.search("weather on mars")["passages"] == []


def test_is_grounded_pre_filter():
    passages = knowledge.search("chronic kidney disease")["passages"]
    assert knowledge.is_grounded("CKD damages the kidneys and is managed with diet", passages)
    assert not knowledge.is_grounded("The stock market rallied on tech earnings", passages)


def test_search_medical_info_tool_ok_and_cited():
    out = tools.search_medical_info(auth.authenticate("raj", "demo123"),
                                    condition="high blood pressure")
    assert out["status"] == "ok"
    assert out["citations"] and "blood pressure" in out["summary"].lower()


def test_search_medical_info_tool_defers_when_unknown():
    out = tools.search_medical_info(auth.authenticate("raj", "demo123"),
                                    condition="dragonpox")
    assert out["status"] == "empty"
