from sentinel.agent.tools import build_toolset, calculator


def test_calculator_basic_arithmetic():
    assert calculator("2 + 3") == "5"


def test_calculator_respects_precedence():
    assert calculator("(2 + 3) * 4") == "20"


def test_calculator_rejects_unsafe_expressions():
    result = calculator("__import__('os').system('echo hi')")
    assert result.startswith("error:")


def test_calculator_rejects_attribute_access():
    result = calculator("().__class__")
    assert result.startswith("error:")


class _FakeStore:
    def query_text(self, query: str, top_k: int = 5):
        return []

    def query_images(self, vector, top_k: int = 3):
        return []


def test_build_toolset_exposes_expected_tools():
    tools = build_toolset(_FakeStore())
    names = {t.name for t in tools}
    assert names == {"retrieve_documents", "calculator"}
    for tool in tools:
        assert "type" in tool.input_schema
        assert tool.input_schema["type"] == "object"
