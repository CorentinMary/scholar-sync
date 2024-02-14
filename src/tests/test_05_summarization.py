import pytest

from ..engine.summarization import DummySummarizer, PromptSummarizer


@pytest.mark.dependency(
    name="dummy_summarizer",
    depends=["get_abstract_1", "get_abstract_2", "get_abstract_3", "get_abstract_4"],
    scope="session",
)
def test_dummy_summarizer():
    sep = "----"
    abs_name = "ABSTRACT"
    summarizer = DummySummarizer(max_summary_tokens=3, sep=sep, abstract_section_name=[abs_name])
    text_list = [
        f"""Some text before {sep} {abs_name} This is the abstract section. {sep} Some text after""",
        f"""{abs_name} Abstract content goes here. {sep} More text""",
        "No abstract section in this text. {sep} Another section",
        f"""Text with multiple sections. {sep} {abs_name} First abstract. {sep} Another section
        {sep} {abs_name} Second abstract. {sep} More text""",
        f"""{abs_name}   {sep} Some section content.""",
    ]
    expected_summaries = [
        "This is the",
        "Abstract content goes",
        "No abstract section",
        "First abstract. ",
        "Some section content.",
    ]
    assert summarizer.summarize(text_list) == expected_summaries


@pytest.mark.dependency(name="prompt_summarizer", scope="session")
def test_prompt_summarizer():
    summarizer = PromptSummarizer(model_name="openai", max_summary_tokens=5)
    text_list = [
        "This is a first article that should be summarized.",
        "This is another article that should be summarized.",
    ]
    summary_list = summarizer.summarize(text_list)
    assert len(summary_list) == 2
    assert all(summarizer.llm.get_num_tokens(summary) <= 5 for summary in summary_list)
