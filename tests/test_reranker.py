from langchain_core.documents import Document

from rag.reranker import rerank


def test_reranker_ranks_relevant_higher() -> None:
    """The cross-encoder should score a relevant doc higher than an irrelevant one."""
    docs = [
        Document(page_content="Stari Most u Mostaru je proglašen najljepšim mostom."),
        Document(page_content="Poslovni sistem predstavlja skup proizvodnih i ekonomskih podsistema."),
    ]

    scored = rerank("Sta su poslovni sistemi?", docs)

    assert scored[0][0].page_content.startswith("Poslovni")
    assert scored[0][1] > scored[1][1]
