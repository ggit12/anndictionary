"""
unit tests for anndict.llm.parse_llm_response
"""

from anndict.llm import (
    extract_dictionary_from_ai_string,
    extract_list_from_ai_string,
    process_llm_category_mapping
)

def test_extract_dictionary_from_ai_string(raw_llm_response_with_mapping):
    """Test dictionary extraction from AI-generated string"""
    result = extract_dictionary_from_ai_string(raw_llm_response_with_mapping)
    expected = '''{
        "First Category": "Group A",
        "Second Category": "Group A",
        "Third Category": "Group B",
        "Fourth Category": "Group B",
        "Fifth Category": "Group C"
    }'''
    assert result == expected


def test_extract_dictionary_from_ai_string_no_dict():
    """Test dictionary extraction when no dictionary is present"""
    test_string = "This string has no dictionary"
    result = extract_dictionary_from_ai_string(test_string)
    assert result == ""


def test_extract_dictionary_from_ai_string_multiple_dicts():
    """Test dictionary extraction with multiple dictionaries - should return first one"""
    test_string = '''
    First dict: {"first": "value1", "second": "value2"}
    Second dict: {"third": "value3", "fourth": "value4"}
    '''
    result = extract_dictionary_from_ai_string(test_string)
    assert result == '{"first": "value1", "second": "value2"}'


def test_extract_list_from_ai_string(raw_llm_response_with_list):
    """Test list extraction from AI-generated string"""
    result = extract_list_from_ai_string(raw_llm_response_with_list)
    expected = '''[
        "Group A",
        "Group B",
        "Group C"
    ]'''
    assert result == expected


def test_extract_list_from_ai_string_no_list():
    """Test list extraction when no list is present"""
    test_string = "This string has no list"
    result = extract_list_from_ai_string(test_string)
    assert result == ""


def test_extract_list_from_ai_string_multiple_lists():
    """Test list extraction with multiple lists - should return first one"""
    test_string = '''
    First list: ["first", "second"]
    Second list: ["third", "fourth"]
    '''
    result = extract_list_from_ai_string(test_string)
    assert result == '["first", "second"]'


def test_process_llm_category_mapping_exact_matches(
    input_categories,
    sample_mapping,
    expected_mapping
):
    """Test category mapping with exact matches"""
    result = process_llm_category_mapping(input_categories, sample_mapping)
    assert result == expected_mapping


def test_process_llm_category_mapping_empty_inputs():
    """Test category mapping with empty inputs"""
    result = process_llm_category_mapping([], {})
    assert result == {} # pylint: disable=use-implicit-booleaness-not-comparison


def test_process_llm_category_mapping_no_matches():
    """Test category mapping when no matches are found"""
    original = ["Unmapped Category"]
    mapping = {"First Category": "Group A"}
    expected = {"Unmapped Category": "Unmapped Category"}
    result = process_llm_category_mapping(original, mapping)
    assert result == expected


def test_process_llm_category_mapping_case_insensitive():
    """Test category mapping is case insensitive"""
    original = ["FIRST CATEGORY", "first category"]
    mapping = {"First Category": "Group A"}
    expected = {
        "FIRST CATEGORY": "Group A",
        "first category": "Group A"
    }
    result = process_llm_category_mapping(original, mapping)
    assert result == expected


def test_preserves_categories_below_similarity_threshold():
    """
    Given a mix of categories above and below the 0.6 similarity threshold
    When processing the category mapping
    Then only categories above threshold should be mapped, others preserved
    """
    input_categories = [
        "Category A",           # Exact match
        "Cat Axxxxx",           # Below similarity threshold
        "Categories B",         # Above similarity threshold
        "Something Else"        # No match at all
    ]
    category_mapping = {
        "Category A": "Group A",
        "Category B": "Group B",
        "Category C": "Group C"
    }

    result = process_llm_category_mapping(input_categories, category_mapping)

    print(result)

    expected = {
        "Category A": "Group A",                # Exact match - should map
        "Cat Axxxxx": "Cat Axxxxx",             # Below threshold - should preserve
        "Categories B": "Group B",              # Similar enough - should map
        "Something Else": "Something Else"      # No match - should preserve
    }
    assert result == expected
