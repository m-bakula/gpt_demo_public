import openai
import tiktoken
import pandas as pd
from pathlib import Path
from openai.embeddings_utils import get_embedding, cosine_similarity


class Chat:
    def __init__(
            self,
            model: str = 'gpt-3.5-turbo-16k',
            temperature: float = 0.5,
            sys_msg: str = 'Jesteś pomocnym i uprzejmym asystentem',
            emb_model: str = 'text-embedding-ada-002',
            emb_path: Path | None = None
    ) -> None:
        """
        :arg model: nazwa modelu Chat Completion zgodna z OpenAI API
        :arg emb_model: nazwa modelu Embeddings zgodna z OpenAI API
        :arg temperature: 'temperatura'/kreatywnosc modelu. wartosc od 0.0 do 2.0
        :arg sys_msg: wiadomosc systemowa w jezyku naturalnym nadajaca ton i sposob odpowiedzi modelu
        """
        self._mod_name = model
        self._mod_temp = temperature
        self._system_msg = sys_msg

        self._messages: list[dict[str, str]] = [
            {
                'role': 'system',
                'content': sys_msg
            }
        ]
        self.emb_mod_name = emb_model
        self.embeddings: pd.DataFrame | None = pd.DataFrame(
            {
                'filename': list(pd.read_pickle(emb_path).keys()),
                'embedding': list(pd.read_pickle(emb_path).values())
            }
        ) if emb_path is not None else None
        self.used_embeddings: list[float] = []

    def get_reply(
            self,
            text: str
    ) -> str:
        if text:
            self._messages.append(
                {
                    'role': 'user',
                    'content': text
                }
            )

            completion = openai.ChatCompletion.create(
                model=self._mod_name,
                temperature=self._mod_temp,
                messages=self._messages,
            )

            reply = completion.choices[0].message.content
            self._messages.append(
                {
                    'role': 'assistant',
                    'content': reply
                }
            )

            print(self._messages[-2], self._messages[-1])
            return reply

    def get_reply_stream(
            self,
            text: str
    ):
        if text:
            self._messages.append(
                {
                    'role': 'user',
                    'content': text
                }
            )

            completion = openai.ChatCompletion.create(
                model=self._mod_name,
                temperature=self._mod_temp,
                messages=self._messages,
                stream=True
            )

            partial_reply = ''
            for chunk in completion:
                if chunk.choices[0].delta != {}:
                    partial_reply += chunk.choices[0].delta.content
                    yield partial_reply
                else:
                    self._messages.append(
                        {
                            'role': 'assistant',
                            'content': partial_reply
                        }
                    )
                    print(self._messages[-2], self._messages[-1])
                    yield partial_reply

    def upload_content(
            self,
            content: str
    ) -> None:
        if content:
            prefix: str = 'TEKST PONIŻEJ:\n=====\n'
            self._messages.append({'role': 'user', 'content': prefix + content})

    def delete_text_from_content(
            self,
            text_to_delete: str
    ) -> None:
        if text_to_delete:
            for i in range(len(self._messages)):
                if (self._messages[i]['role'] == 'user'
                        and text_to_delete in self._messages[i]['content']):
                    self._messages[i]['content'] = (self._messages[i]['content']
                                                    .replace(text_to_delete, ''))

    def get_text_size(
            self,
            text: str
    ) -> int:
        encoding = tiktoken.encoding_for_model(self._mod_name)
        return len(encoding.encode(text))

    # def upload_embeds(
    #         self,
    #         filename: str,
    #         embedding: list[float]
    # ) -> None:
    #     self.embeddings.loc[len(self.embeddings)] = {'filename': filename, 'embedding': embedding}

    def find_embedding(
            self,
            text: str,
            min_simil: float = 0.83,
    ) -> None:
        """
        Wyszukuje embeddingi najbardziej podobne do zapytania i dodaje je do kontekstu.
        :param text: treść ostatniego zapytania
        :param min_simil: minimalny stopień dopasowania
        :return: None
        """
        if self.embeddings is not None:
            text_embedding = get_embedding(text, engine=self.emb_mod_name)

            df = self.embeddings.copy()
            df['similarity'] = df['embedding'].apply(lambda x: cosine_similarity(x, text_embedding))

            result = df.sort_values('similarity', ascending=False).reset_index()
            print(result[['filename', 'similarity']])

            # usuń z kontekstu
            drop = result.where(result['similarity'] < min_simil).dropna()
            for i in range(len(drop)):
                dropped_copy = drop.copy().reset_index()
                dropped_filename = dropped_copy.loc[i, 'filename']
                with open(dropped_filename, encoding='utf-8') as file:
                    text_to_delete = file.read()
                    self.delete_text_from_content(text_to_delete)

            result.drop(list(drop.index), inplace=True)
            print(result[['filename', 'similarity']])

            # dodaj do kontekstu
            for i in range(len(result)):
                filename = result.loc[i, 'filename']
                embedding = result.loc[i, 'embedding']

                if embedding not in self.used_embeddings:
                    with open(filename, encoding='utf-8') as file:
                        text_from_file = file.read()
                    self.upload_content(text_from_file)
                    self.used_embeddings.append(embedding)
