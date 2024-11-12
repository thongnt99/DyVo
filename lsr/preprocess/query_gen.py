import numpy as np
import json
from collections import Counter
import ir_datasets
from tqdm import tqdm
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk import PorterStemmer
from collections import defaultdict
# stopWords = set(stopwords.words("english"))
# ps = PorterStemmer()
dataset = ir_datasets.load("disks45/nocr/trec-robust-2004")
# doc2sentences = {}
# doc2query = defaultdict(list)
# df =
# df = Counter()
# idf = {}
num_sent = 0
qrels = defaultdict(dict)
queries = []


def preprocess(sent):
    sent = sent.strip()
    words = word_tokenize(sent)
    if len(words) < 5:
        return None
    return sent


for doc in tqdm(dataset.docs_iter(), desc="Sentence spliting and word tokenization"):
    text = doc.body.replace("\n", " ")
    sentences = sent_tokenize(text)
    if len(sentences) == 0:
        continue
    if len(sentences[0]) >= 5:
        queries.append((sentences[0], doc.doc_id))
    if len(sentences) > 1:
        sampled_sentence = np.random.choice(sentences[1:])
        if len(sampled_sentence) >= 5:
            queries.append((sampled_sentence, doc.doc_id))
np.random.shuffle(queries)
queries = queries[:10000]
qid2query = {}
qrels = defaultdict(dict)
for qid, qd in enumerate(queries):
    query, doc_id = qd
    qid2query[qid] = query
    qrels[qid][doc_id] = 1
with open("data/robust04/genfree/queries.tsv", "w") as f:
    for qid in qid2query:
        f.write(f"{qid}\t{qid2query[qid]}\n")
json.dump(qrels, open("data/robust04/genfree/qrels.json", "w"))
# tokenized_sentences = [
#     [ps.stem(w.lower()) for w in word_tokenize(sent)] for sent in sentences]
# num_sent += len(tokenized_sentences)
# doc2sentences[doc.doc_id] = list(zip(sentences, tokenized_sentences))
# doc2query[doc.doc_id].append((sentences[0], 1))
# for word_list in tokenized_sentences:
#     df.update(set(word_list))
# if len(doc2query) == 10000:
#     break
# for w in df:
#     w_df = df[w]*1.0/num_sent
#     if w_df <= 0.02 or w_df >= 0.5:
#         idf[w] = 0.0
#     else:
#         idf[w] = 1.0/w_df

# for docid in tqdm(doc2sentences, desc="Scoring sentence selection"):
#     for sent_text, sent_words in doc2sentences[docid]:
#         score = 0
#         tf = Counter(sent_words)
#         for w in tf:
#             score += tf[w]*idf[w]
#         score = score*1.0 / len(sent_words)
#         doc2query[docid].append((sent_text, score))
# json.dump(doc2query, open("data/robust04/genfree/doc2query.json", "w"))
# selected_queries = []
# for docid in doc2query:
#     sent2score = dict(doc2query[docid])
#     probs = np.array(list(sent2score.values()))
#     if sum(probs) == 0 or np.any(np.isnan(probs)):
#         continue
#     probs = probs/sum(probs)
#     samples = np.random.choice(
#         list(sent2score.keys()), size=min(5, len(sent2score), (probs > 0).sum()), replace=False, p=probs)
#     selected_queries.extend([[s, sent2score[s]] for s in samples])
# selected_queries = sorted(selected_queries, key=lambda x: x[1], reverse=True)
# json.dump(selected_queries, open(
#     "data/robust04/genfree/sample_queries.json", "w"))
