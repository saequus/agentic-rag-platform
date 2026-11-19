import pytest
from pydantic import ValidationError

from agentic_rag_platform.schemas import Answer, ChatRequest


def test_answer_accepts_bounds() -> None:
    low = Answer(answer="uncertain", citations=[], confidence=0.0)
    high = Answer(answer="sure", citations=["https://api.nbp.pl"], confidence=1.0)
    assert low.confidence == 0.0
    assert high.confidence == 1.0
    assert high.citations == ["https://api.nbp.pl"]


@pytest.mark.parametrize("confidence", [-0.01, 1.01, 2])
def test_answer_rejects_confidence_out_of_range(confidence: float) -> None:
    with pytest.raises(ValidationError):
        Answer(answer="x", citations=[], confidence=confidence)


def test_answer_rejects_blank() -> None:
    with pytest.raises(ValidationError):
        Answer(answer="   ", citations=[], confidence=0.5)


def test_chat_request_requires_message() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(message="")
    assert ChatRequest(message="What is 1 USD in PLN?").history == []
