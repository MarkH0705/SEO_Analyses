import os
import openai
from google.colab import userdata


os.environ['OPENAI_API_KEY'] = userdata.get('open_ai_api_key')

class Chatbot:

    def __init__(self, systemprompt, userprompt):
        self.client = openai.OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        self.systemprompt = systemprompt
        self.prompt = userprompt
        self.context = [{"role": "system", "content": systemprompt}]
        self.model = "gpt-4o-mini-2024-07-18"

    def chat(self):
        """
        Sendet den Prompt an das Chat-Interface und gibt den kompletten Antwort-String zurück.
        """
        self.context.append({"role": "user", "content": self.prompt})
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.context
            )
            response_content = response.choices[0].message.content
            self.context.append({"role": "assistant", "content": response_content})
            return response_content
        except Exception as e:
            print(f"Fehler bei der OpenAI-Anfrage: {e}")
            return ""


    def chat_with_streaming(self):
            """
            Interagiert mit OpenAI Chat Completion API und streamt die Antwort.
            """
            # Nachricht zur Konversation hinzufügen
            self.context.append({"role": "user", "content": self.prompt})


            try:
                # Streaming-Option aktivieren
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.context,
                    stream=True
                )

                streamed_content = ""  # Zum Speichern der gestreamten Antwort

                for chunk in response:
                    # Debugging: Anzeigen, was tatsächlich in jedem Chunk enthalten ist
                    delta = chunk.choices[0].delta
                    content = getattr(delta, "content", "")

                    if content:  # Verarbeite nur nicht-leere Inhalte
                        print(content, end="", flush=True)
                        streamed_content += content

                print()  # Neue Zeile am Ende

                # Gestreamte Antwort zur Konversation hinzufügen
                self.context.append({"role": "assistant", "content": streamed_content})

                # Return the streamed content
                return streamed_content # This line was added

            except Exception as e:
                print(f"\nDEBUG: An error occurred during streaming: {e}")
                # Return empty string in case of error
                return "" # This line was added
