from deck_agent.toolcall_parse import extract_tool_calls, parse_arguments, strip_thinking

KNOWN = {"web_search", "add_bullet_slide"}


def test_strip_thinking():
    assert strip_thinking("<think>plan</think>\nHello") == ("Hello", "plan")
    assert strip_thinking("hidden reasoning</think>Answer") == ("Answer", "hidden reasoning")
    assert strip_thinking("plain") == ("plain", "")


def test_hermes_json_tool_call():
    text = 'Sure.\n<tool_call>\n{"name": "web_search", "arguments": {"query": "heat pumps"}}\n</tool_call>'
    calls, rest = extract_tool_calls(text, KNOWN)
    assert [(c.name, c.arguments) for c in calls] == [("web_search", {"query": "heat pumps"})]
    assert rest == "Sure."


def test_qwen_coder_xml_tool_call():
    text = (
        "<tool_call>\n<function=add_bullet_slide>\n<parameter=title>\nWhy now\n</parameter>\n"
        '<parameter=bullets>\n["a", "b"]\n</parameter>\n<parameter=position>\n3\n</parameter>\n'
        "</function>\n</tool_call>"
    )
    calls, _ = extract_tool_calls(text, KNOWN)
    assert calls[0].name == "add_bullet_slide"
    assert calls[0].arguments == {"title": "Why now", "bullets": ["a", "b"], "position": 3}


def test_multiple_and_unterminated_blocks():
    text = ('<tool_call>{"name":"web_search","arguments":{"query":"a"}}</tool_call>'
            '<tool_call>{"name":"web_search","arguments":{"query":"b"}}')
    calls, _ = extract_tool_calls(text, KNOWN)
    assert [c.arguments["query"] for c in calls] == ["a", "b"]


def test_fenced_and_bare_json_only_for_known_tools():
    fenced = 'I will search.\n```json\n{"name": "web_search", "arguments": {"query": "x"}}\n```'
    calls, rest = extract_tool_calls(fenced, KNOWN)
    assert calls[0].name == "web_search" and rest == "I will search."
    bare, _ = extract_tool_calls('{"name": "web_search", "parameters": {"query": "y"}}', KNOWN)
    assert bare[0].arguments == {"query": "y"}
    unknown, rest = extract_tool_calls('{"name": "rm_rf", "arguments": {}}', KNOWN)
    assert unknown == [] and rest.startswith("{")


def test_ordinary_answer_is_not_a_tool_call():
    calls, rest = extract_tool_calls("The deck is saved. It has 8 slides.", KNOWN)
    assert calls == [] and rest.startswith("The deck")


def test_parse_arguments_tolerates_slips():
    assert parse_arguments('{"a": 1,}') == ({"a": 1}, None)
    assert parse_arguments('"{\\"a\\": 2}"') == ({"a": 2}, None)  # double-encoded
    assert parse_arguments('noise {"a": 3} noise') == ({"a": 3}, None)
    assert parse_arguments({"a": 4}) == ({"a": 4}, None)
    assert parse_arguments("") == ({}, None)
    args, err = parse_arguments("{not json")
    assert args == {} and "not valid JSON" in err
