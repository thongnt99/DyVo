from flair.data import Sentence
from flair.models import SequenceTagger

# load tagger
tagger = SequenceTagger.load("flair/ner-english-fast")

# make example sentence
text = input("Enter text: ")
sentence = Sentence(text)

# predict NER tags
tagger.predict(sentence)

# print sentence
print(sentence)

# print predicted NER spans
print('The following NER tags are found:')
# iterate over entities and print
for entity in sentence.get_spans('ner'):
    print(entity)

# from flair.models import SequenceTagger
# from flair.data import Sentence
# from flair.nn import Classifier
# import requests

# API_URL = "https://rel.cs.ru.nl/api"
# text_doc = "how many days are you allowed for overnight stays when you are on washington state medicaid"

# tagger = Classifier.load('linker')


# # make example sentence
# sentence = Sentence(text_doc)

# # predict NER tags
# tagger.predict(sentence)

# # print sentence
# print(sentence)

# # print predicted NER spans
# print('The following NER tags are found:')
# # iterate over entities and print
# for label in sentence.get_labels():
#     print(label)
# # Example EL.
# el_result = requests.post(API_URL, json={
#     "text": text_doc,
#     "spans": [(0, 9)]
# }).json()
# print(el_result)

# # # Example ED.
# # ed_result = requests.post(API_URL, json={
# #     "text": text_doc,
# #     "spans": [(41, 16)]
# # }).json()
# # print(ed_result)
