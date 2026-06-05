from med_assistant.services.rag_service import is_conversational_query, conversational_reply


def test_conversational_greetings():
    assert is_conversational_query("hi")
    assert is_conversational_query("hii")
    assert is_conversational_query("Hello!")
    assert is_conversational_query("hey there")


def test_conversational_thanks_and_bye():
    assert is_conversational_query("thanks")
    assert is_conversational_query("thank you")
    assert is_conversational_query("bye")


def test_medical_queries_not_conversational():
    assert not is_conversational_query("What is aplastic anemia?")
    assert not is_conversational_query("Explain influenza symptoms")
    assert not is_conversational_query("How is diabetes treated?")


def test_conversational_reply_greeting():
    reply = conversational_reply("hii")
    assert "MedAssist" in reply


def test_conversational_reply_thanks():
    reply = conversational_reply("thank you")
    assert "welcome" in reply.lower()
