# # from gpt4all import GPT4All
# # model = GPT4All("orca-mini-3b-gguf2-q4_0.gguf") # downloads / loads a 4.66GB LLM
# # with model.chat_session():
# #     print(model.generate("How can I run LLMs efficiently on my laptop?", max_tokens=1024))
from gpt4all import GPT4All
from googletrans import Translator

plompt = "As you present data on visitors, what trends can you read? It is forbidden to present data using %. Percentage of visitors' preferred genres: Japanese food 10%, Western food 30%, Chinese food 20%, Italian food 10%, Fast food 30%. The percentage of visitors' occupation is 50% working adults, 20% college students, 10% high school students, 10% junior high school students, and 10% other. Gender of visitors: 60% male, 40% female. Nationality of visitors: 80% Japanese, 20% foreign."

translator = Translator()
english_plompt = translator.translate(plompt, src='ja', dest='en')

model = GPT4All("orca-mini-3b-gguf2-q4_0.gguf")
with model.chat_session():
    print(translator.translate(model.generate(plompt, max_tokens=1024), src='en', dest='ja').text)
# from googletrans import Translator

# # 翻訳器のインスタンスを作成
# translator = Translator()

# # 日本語から英語に翻訳
# japanese_text = "こんにちは、元気ですか？"
# translated_to_english = translator.translate(japanese_text, src='ja', dest='en')
# print(f"日本語: {japanese_text}")
# print(f"英語: {translated_to_english.text}")

# # 英語から日本語に翻訳
# english_text = "How are you doing?"
# translated_to_japanese = translator.translate(english_text, src='en', dest='ja')
# print(f"英語: {english_text}")
# print(f"日本語: {translated_to_japanese.text}")
