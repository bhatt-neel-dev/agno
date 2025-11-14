from agno.models.message import Message
from agno.models.openai.chat import OpenAIChat


def test_format_message_truncates_long_tool_call_ids():
    """Test that tool_call IDs longer than 40 chars are truncated"""
    model = OpenAIChat(id="gpt-4o-mini")

    # Create a message with a tool call ID longer than 40 chars (like from Responses API)
    long_id = "fc_0f0e50a1476a6c35006912c58dc9f8819192c25864eb16f07f"  # 53 chars
    message = Message(
        role="assistant",
        tool_calls=[
            {
                "id": long_id,
                "type": "function",
                "function": {"name": "test_function", "arguments": '{"arg": "value"}'},
            }
        ],
    )

    formatted = model._format_message(message)

    # Verify the ID was truncated to 40 chars
    assert "tool_calls" in formatted
    assert len(formatted["tool_calls"]) == 1
    assert len(formatted["tool_calls"][0]["id"]) == 40
    # Verify it's the first 40 chars
    assert formatted["tool_calls"][0]["id"] == long_id[:40]


def test_format_message_preserves_short_tool_call_ids():
    """Test that tool_call IDs under 40 chars are not modified"""
    model = OpenAIChat(id="gpt-4o-mini")

    short_id = "call_abc123def456"  # 16 chars
    message = Message(
        role="assistant",
        tool_calls=[
            {
                "id": short_id,
                "type": "function",
                "function": {"name": "test_function", "arguments": '{"arg": "value"}'},
            }
        ],
    )

    formatted = model._format_message(message)

    # Verify the ID was not modified
    assert formatted["tool_calls"][0]["id"] == short_id


def test_format_message_truncates_tool_call_id_in_tool_response():
    """Test that tool_call_id in tool messages is truncated"""
    model = OpenAIChat(id="gpt-4o-mini")

    long_id = "fc_0f0e50a1476a6c35006912c58dc9f8819192c25864eb16f07f"  # 53 chars
    message = Message(role="tool", tool_call_id=long_id, content="Tool output")

    formatted = model._format_message(message)

    # Verify the tool_call_id was truncated
    assert len(formatted["tool_call_id"]) == 40
    assert formatted["tool_call_id"] == long_id[:40]


def test_format_message_handles_multiple_tool_calls():
    """Test that multiple tool calls are all truncated correctly"""
    model = OpenAIChat(id="gpt-4o-mini")

    long_id_1 = "fc_0f0e50a1476a6c35006912c58dc9f8819192c25864eb16f07f"  # 53 chars
    long_id_2 = "fc_1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p7q8r9s0t1u2v3w4x"  # 53 chars
    short_id = "call_123abc"  # 11 chars

    message = Message(
        role="assistant",
        tool_calls=[
            {
                "id": long_id_1,
                "type": "function",
                "function": {"name": "func1", "arguments": "{}"},
            },
            {
                "id": long_id_2,
                "type": "function",
                "function": {"name": "func2", "arguments": "{}"},
            },
            {
                "id": short_id,
                "type": "function",
                "function": {"name": "func3", "arguments": "{}"},
            },
        ],
    )

    formatted = model._format_message(message)

    # Verify all IDs are handled correctly
    assert len(formatted["tool_calls"]) == 3
    assert len(formatted["tool_calls"][0]["id"]) == 40
    assert len(formatted["tool_calls"][1]["id"]) == 40
    assert len(formatted["tool_calls"][2]["id"]) == 11  # Short ID unchanged
    assert formatted["tool_calls"][0]["id"] == long_id_1[:40]
    assert formatted["tool_calls"][1]["id"] == long_id_2[:40]
    assert formatted["tool_calls"][2]["id"] == short_id