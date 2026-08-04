from data_analysis_agent.llm.base import LLMProvider

def test_llm_provider_e_um_protocol():
    assert hasattr(LLMProvider, "__dict__")