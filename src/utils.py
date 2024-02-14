import os
from typing import List


def fetch_files(path: str, ignore: List[str] = [".gitkeep"]) -> List[str]:
    """Returns a list of files at a given path.

    :param path: str.
        location to list files of.
    :param ignore: List[str], defaults to [".gitkeep"].
        list of file names to ignore.
    :return: list of file names.
    """
    obj_list = os.listdir(path)
    file_list = [file for file in obj_list if os.path.isfile(f"{path}/{file}") and file not in ignore]

    return file_list


def get_title_from_path(path: str) -> str:
    """Returns the name of a file (without folder nor extension) from its path.

    :param path: str.
        path to the file.
    :return: file name
    """
    # keeping only the part after folder(s)
    file = path.split("/")[-1].split("\\")[-1]
    # removing extension
    file_name = file.split(".")[0]

    return file_name


def trim_leading_whitespace(string: str) -> str:
    """Removes the whitespace(s) at the beginning of a string

    :param string: str.
        string to trim.
    :return: trimmed string.
    """
    # stopping when the string is empty or doesn't start with a whitespace/new line
    if len(string) == 0 or string[0] not in [" ", "\n"]:
        return string
    else:
        return trim_leading_whitespace(string[1:])


def remove_section_indicator(
    text: str, indicator_list: List[str] = ["ABSTRACT", "TITLE PARAGRAPH:", "DESCRIPTION TABLE:"]
) -> str:
    """Returns a text without words indicating the beginning of a section.

    :param text: str.
        text to remove indicators from.
    :param indicator_list: List[str], defaults to ["ABSTRACT", "TITLE PARAGRAPH:", "DESCRIPTION TABLE:"].
        list of words indicating the beginning of a section
    :return: text without indicators.
    """
    text_ = trim_leading_whitespace(text)
    for indicator in indicator_list:
        if text_.startswith(indicator):
            text_ = text_[len(indicator) :]
            break

    return trim_leading_whitespace(text_)


def get_abstract(text: str, sep: str = "----", abstract_section_name: List[str] = ["ABSTRACT"]) -> str:
    """Extracts the abstract section (or equivalent) of a text.

    :param sep: int, defaults to "----".
        delimiter for text sections.
    :param abstract_section_name: List[str], defaults to ["ABSTRACT"].
        section title(s) to filter on to get the abstract.
    :return: abstract content.
    """
    # splitting the text into sections which are delimited by the separator provided
    section_list = text.split(sep)
    clean_section_list = [remove_section_indicator(section) for section in section_list]
    # removing spaces and new lines for filtering
    trim_section_list = [section.replace("\n", "").replace(" ", "").upper() for section in section_list]
    # keeping the first section found that starts with one of the abstract section name
    abstract_section_idx = [
        i for i in range(len(section_list)) if trim_section_list[i].startswith(tuple(abstract_section_name))
    ]
    if len(abstract_section_idx) and len(clean_section_list[abstract_section_idx[0]]):
        return clean_section_list[abstract_section_idx[0]]
    # if no abstract section was found or the abstract section found is empty, return the first non empty section of
    # the text
    non_empty_section_list = [section for section in clean_section_list if len(section)]
    return non_empty_section_list[0]
