import gradio as gr
import openai
from pathlib import Path

from modules.classes import Chat


def main():
    with open('api_key.txt', 'r') as keyfile:
        openai.api_key = keyfile.read().replace('\n', '')

    cust_msg = 'Jesteś uprzejmym i pomocnym asystentem. Twoim celem jest analizowanie przesłanego tekstu.' \
               'Odpowiadasz na pytania tylko na podstawie przesłanego tekstu'
    emb_path = Path('baza_small') / 'embeds.pickle'
    chat_obj = Chat(
        model='gpt-3.5-turbo-16k',
        temperature=0.5,
        sys_msg=cust_msg,
        emb_path=emb_path
    )

    def respond(message, history):
        chat_obj.find_embedding(message)
        history.append([message, None])
        reply = chat_obj.get_reply_stream(message)

        for _ in reply:
            try:
                history[-1][1] = next(reply)
            except StopIteration:
                pass
            yield '', history

    with gr.Blocks() as app:
        chatbot = gr.Chatbot(height=900)
        msg = gr.Textbox()
        clr = gr.ClearButton([msg, chatbot])
        msg.submit(respond, [msg, chatbot], [msg, chatbot])

    app.queue()
    app.launch()


if __name__ == '__main__':
    main()
