from sql.SqlTemplate import SqlTemplate
from csharp.CSharpTemplate import parseCSharp
import re
import collections
import random
import json
import sys
import os

PAD = 1
UNK = 2
START = 3
END = 4

def tokenizeNL(nl):
  nl = nl.strip().decode('utf-8').encode('ascii', 'replace')
  return re.findall(r"[\w]+|[^\s\w]", nl)

def tokenizeCode(code, lang):
  code = code.strip().decode('utf-8').encode('ascii', 'replace')
  typedCode = None
  if lang == "sql":
    query = SqlTemplate(code, regex=True)
    typedCode = query.parseSql()
  elif lang == "csharp":
    typedCode = parseCSharp(code)
  elif lang == "python":
    typedCode = q.strip().decode('utf-8').encode('ascii', 'replace').split("\\s")

  tokens = [re.sub( '\s+', ' ', x.strip())  for x in typedCode]
  return tokens

# lang can be csharp or code
def buildVocab(filename, code_unk_threshold, nl_unk_threshold, lang):
  vocab = {
    "nl_to_num": {"UNK": UNK, "CODE_START": START, "CODE_END": END},
    "code_to_num": {"UNK": UNK, "CODE_START": START, "CODE_END": END},
    "num_to_nl": {PAD: "UNK", UNK: "UNK", START: "CODE_START", END: "CODE_END"},
    "num_to_code": {UNK: "UNK", START: "CODE_START", END: "CODE_END"},
  }

  words = collections.Counter()
  tokens = collections.Counter()

  for line in open(filename, "r"):
    qid, rid, nl, code, weight = line.strip().split('\t')
    tokens.update(tokenizeCode(code, lang))
    words.update(tokenizeNL(nl))

  token_count = END + 1
  nl_count = END + 1

  # Do unigram tokens
  for tok in tokens:
    if tokens[tok] > code_unk_threshold:
      vocab["code_to_num"][tok] = token_count
      vocab["num_to_code"][token_count] = tok
      token_count += 1
    else:
      vocab["code_to_num"][tok] = UNK

  for word in words:
    if words[word] > nl_unk_threshold:
      vocab["nl_to_num"][word] = nl_count
      vocab["num_to_nl"][nl_count] = word
      nl_count += 1
    else:
      vocab["nl_to_num"][word] = UNK

  vocab["max_code"] = token_count - 1
  vocab["max_nl"] = nl_count - 1
  vocab["lang"] = lang

  f = open(os.environ["CODENN_WORK"] + '/vocab.' + lang, 'w')
  f.write(json.dumps(vocab))
  f.close()

  return vocab


def get_data(filename, vocab, dont_skip, max_code_length, max_nl_length):

  dataset = []
  skipped = 0
  for line in open(filename, 'r'):

    qid, rid, nl, code, wt = line.strip().split('\t')
    codeToks  = tokenizeCode(code, vocab["lang"])
    nlToks = tokenizeNL(nl)

    datasetEntry = {"id": rid, "code": code, "code_sizes": len(codeToks), "code_num":[], "nl_num":[]}

    for tok in codeToks:
      if tok not in vocab["code_to_num"]:
        vocab["code_to_num"][tok] = UNK
      datasetEntry["code_num"].append(vocab["code_to_num"][tok])

    datasetEntry["nl_num"].append(vocab["nl_to_num"]["CODE_START"])
    for word in nlToks:
      if word not in vocab["nl_to_num"]:
        vocab["nl_to_num"][word] = UNK
      datasetEntry["nl_num"].append(vocab["nl_to_num"][word])

    datasetEntry["nl_num"].append(vocab["nl_to_num"]["CODE_END"])

    if dont_skip or (len(datasetEntry["code_num"]) <= max_code_length and len(datasetEntry["nl_num"]) <= max_nl_length):
      dataset.append(datasetEntry)
    else:
      skipped += 1

  print 'Total size = ' + str(len(dataset))
  print 'Total skipped = ' + str(skipped)

  f = open(os.environ["CODENN_WORK"] + '/' + os.path.basename(filename) + "." + lang, 'w')
  f.write(json.dumps(dataset))
  f.close()


if __name__ == '__main__':

  lang = sys.argv[1]
  max_code_len = int(sys.argv[2])
  max_nl_len = int(sys.argv[3])
  code_unk_threshold = int(sys.argv[4])
  nl_unk_threshold = int(sys.argv[5])


  vocab = buildVocab(os.environ['CODENN_DIR'] + '/data/stackoverflow/' + lang + '/train.txt', code_unk_threshold, nl_unk_threshold, lang)
  get_data(os.environ['CODENN_DIR'] + '/data/stackoverflow/' + lang + '/train.txt', vocab, False, max_code_len, max_nl_len)
  get_data(os.environ['CODENN_DIR'] + '/data/stackoverflow/' + lang + '/valid.txt', vocab, False, max_code_len, max_nl_len)
  get_data(os.environ['CODENN_DIR'] + '/data/stackoverflow/' + lang + '/dev/dev.txt', vocab, True, max_code_len, max_nl_len)
  get_data(os.environ['CODENN_DIR'] + '/data/stackoverflow/' + lang + '/eval/eval.txt', vocab, True, max_code_len, max_nl_len)
