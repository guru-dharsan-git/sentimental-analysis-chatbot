from googletrans import Translator

translator = Translator()
result = translator.translate('Hello World', src='en', dest='ta')
print(result.text)