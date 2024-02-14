import pytest

from ..utils import fetch_files, get_abstract, get_title_from_path, remove_section_indicator, trim_leading_whitespace


@pytest.fixture
def create_files(tmpdir):
    # Create temporary files for testing
    files = ["file1.txt", "file2.pdf", ".gitkeep"]
    for file in files:
        tmpdir.join(file).write(file)
    return tmpdir


@pytest.mark.dependency(name="fetch_files", scope="session")
def test_fetch_files(create_files):
    files = fetch_files(create_files, ignore=[".gitkeep"])
    assert "file1.txt" in files
    assert "file2.pdf" in files
    assert ".gitkeep" not in files


@pytest.mark.dependency(name="get_title_from_path", scope="session")
def test_get_title_from_path():
    path_windows = "C:\\Users\\User\\Documents\\file.txt"
    assert get_title_from_path(path_windows) == "file"
    path_unix = "/home/user/documents/file.txt"
    assert get_title_from_path(path_unix) == "file"
    path_no_extension = "folder/file"
    assert get_title_from_path(path_no_extension) == "file"
    path_root = "/file.txt"
    assert get_title_from_path(path_root) == "file"
    path_nested_folders = "folder1/folder2/file.txt"
    assert get_title_from_path(path_nested_folders) == "file"
    assert get_title_from_path("") == ""


@pytest.mark.dependency(name="trim_whitespace", scope="session")
def test_trim_leading_whitespace():
    assert trim_leading_whitespace("") == ""
    assert trim_leading_whitespace("hello") == "hello"
    assert trim_leading_whitespace("  hello") == "hello"
    assert trim_leading_whitespace("\nhello") == "hello"
    assert trim_leading_whitespace("  ") == ""
    assert trim_leading_whitespace("\n\n") == ""
    assert trim_leading_whitespace("  hello") == "hello"
    assert trim_leading_whitespace("\n\nhello") == "hello"
    assert trim_leading_whitespace("   \nhello") == "hello"
    assert trim_leading_whitespace("  \n  hello") == "hello"


@pytest.mark.dependency(name="remove_indicator", depends=["trim_whitespace"], scope="session")
def test_remove_section_indicator():
    assert remove_section_indicator("Hello world") == "Hello world"
    assert remove_section_indicator("ABSTRACT  ") == ""
    assert remove_section_indicator("TITLE PARAGRAPH: Lorem ipsum") == "Lorem ipsum"
    assert remove_section_indicator("TITLE PARAGRAPH: Lorem ipsum ABSTRACT") == "Lorem ipsum ABSTRACT"


@pytest.mark.dependency(name="get_abstract_1", depends=["remove_indicator"], scope="session")
def test_get_abstract_with_abstract_section():
    sep = "----"
    abs_name = ["ABSTRACT"]
    text = f"""Some text before {sep} {abs_name[0]} This is the abstract section. {sep} Some text after"""
    expected_output = "This is the abstract section. "
    abstract = get_abstract(text, sep=sep, abstract_section_name=abs_name)
    assert abstract == expected_output


@pytest.mark.dependency(name="get_abstract_2", depends=["remove_indicator"], scope="session")
def test_get_abstract_without_abstract_section():
    sep = "----"
    text = "No abstract section in this text. ---- Another section"
    expected_output = "No abstract section in this text. "
    abstract = get_abstract(text, sep=sep)
    assert abstract == expected_output


@pytest.mark.dependency(name="get_abstract_3", depends=["remove_indicator"], scope="session")
def test_get_abstract_with_multiple_abstract_sections():
    sep = "----"
    abs_name = ["ABSTRACT", "INTRO"]
    text = f"""Text with multiple sections. {sep} {abs_name[0]} First abstract. {sep} Another section
        {sep} {abs_name[1]} Second abstract. {sep} More text"""
    expected_output = "First abstract. "
    abstract = get_abstract(text, sep=sep, abstract_section_name=abs_name)
    assert abstract == expected_output


@pytest.mark.dependency(name="get_abstract_4", depends=["remove_indicator"], scope="session")
def test_get_abstract_with_empty_abstract_sections():
    sep = "----"
    abs_name = ["ABSTRACT"]
    text = f"""{abs_name[0]}  {sep}   {sep} Non empty section"""
    expected_output = "Non empty section"
    abstract = get_abstract(text, sep=sep, abstract_section_name=abs_name)
    assert abstract == expected_output
