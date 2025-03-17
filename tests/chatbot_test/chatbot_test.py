
import pytest
from chatbot import Chatbot

# Beispiel-System- und User-Prompt (je nach Konstruktor definieren, anpassen falls nötig)
system_prompt_test = "Du bist ein hilfreicher Assistent."
user_prompt_test = "Das ist ein Test. Schreibe als Antwort 'Test erfolgreich'."

@pytest.fixture
def chatbot_instance():
    return Chatbot(systemprompt=system_prompt_test, userprompt=user_prompt_test)

def test_chat_method(chatbot_instance):
    """Testet die Standard-Chat-Methode"""
    response = chatbot_instance.chat()
    assert isinstance(response, str)
    assert "Test erfolgreich" in response

def test_chat_with_streaming_method(chatbot_instance):
    """Testet die Chat-Methode mit Streaming"""
    response = chatbot_instance.chat_with_streaming()
    assert isinstance(response, str)
    assert "Test erfolgreich" in response

