from bs4 import BeautifulSoup
from csharp.title_filtering.SVM import SVM
import ast
import os
import antlr4
import re
import pdb
import pickle

params = {
    "langXmlFile" : "Posts.python.xml",
    "xmlFile" : "Posts.answers.xml",
    "outputFile" : "python_all.txt"}

# # First download and process the stackoverflow files
# os.system('wget https://archive.org/download/stackexchange/stackoverflow.com-Posts.7z')
# os.system('7z x stackoverflow.com-Posts.7z')
# 
# os.system('grep "c#" Posts.xml > ' + params['langXmlFile'])
# os.system('grep "PostTypeId=\"2\"" Posts.xml > ' + params['xmlFile'])




#   Title filtering using SVM
s = SVM()
s.train("../csharp/title_filtering/balanced/pos_train.txt", "../csharp/title_filtering/balanced/neg_train.txt")
s.test("../csharp/title_filtering/balanced/pos_test.txt", "../csharp/title_filtering/balanced/neg_test.txt")


def getParam(row, p):
  try:
    output = ""
    try:
      output = row[p].decode('utf-8').encode('ascii', 'replace')
    except:
      output = row[p].encode('ascii', 'replace')
    return str(output)
  except KeyError:
    return -1000000

# Two pass algorithm
acceptedAnswers = {}


i = 0
# First pass. Get the posts tagged with C#. Filter the input using a grep on c# so that this is faster
f = open(params["langXmlFile"], 'r')
for line in f:
  i += 1
  if i % 100000 == 0:
    print(i)
  y = BeautifulSoup(line).html.body.row
  if getParam(y, "acceptedanswerid") != -1000000 and "python" in getParam(y, "tags"):
      acceptedAnswers[int(getParam(y, "acceptedanswerid"))] = {"id": int(getParam(y, "id")), "title": getParam(y, "title") }

f.close()
print('Done with fetching Code posts\n')

# Pass 2, find the corresponding accepted answer
i = 0
f = open(params["xmlFile"], 'r')
for line in f:
  i += 1
  if i % 100000 == 0:
    print(i)
  id1 = line.find("\"")              # Find the first attribute enclosed in "" It should be the Id
  id2 = line.find("\"", id1 + 1)
  qid = int(line[(id1 + 1):id2])
  if qid in acceptedAnswers:
    y = BeautifulSoup(line).html.body.row
    acceptedAnswers[qid]["code"] = getParam(y, "body")     # Store the body


f.close()

pdb.set_trace()
pick = open('acceptedAnswers.pickle', 'w')
pickle.dump(acceptedAnswers, pick)
pick.close()

pick = open('acceptedAnswers.pickle', 'r')
acceptedAnswers = pickle.load(pick)
pick.close()

f = open(params["outputFile"], 'w')
for rid in acceptedAnswers:
  if "code" in acceptedAnswers[rid]:                                # Post contains an accepted answer
    titleFilter = s.filter(acceptedAnswers[rid]['title'])           # Title is good
    if titleFilter == 0:

      soup = BeautifulSoup(acceptedAnswers[rid]["code"])
      codeTag = soup.find_all('code')
      if len(codeTag) == 1:                                         # Contains exactly one piece of code
        try:
          code = codeTag[0].get_text().strip().decode('utf=8').encode('ascii', 'replace')
        except:
          code = codeTag[0].get_text().strip().encode('ascii', 'replace')

        if (len(code) > 6 and len(code) <= 1000):                   # Code must be at most 1000 chars

          # Filter out these weird code snippets
          if code[0] == "<" or code[0] == "=" or code[0] == "@" or code[0] == "$" or \
            code[0:7].lower() == "select " or code[0:7].lower() == "update " or code[0:6].lower() == "alter " or \
            code[0:2].lower() == "c:" or code[0:4].lower() == "http" or code[0:4].lower() == "hkey" or \
                  re.match(r"^[a-zA-Z0-9_]*$", code) is not None: # last one is single word answers
            pass
          else:
          
            # Now also make sure it passes the lexer
            try:
              ast.parse(code)
              code = code.replace('\n', '\\n').replace('\t', '')
              try:
                f.write('\t'.join([str(rid), str(acceptedAnswers[rid]['id']), acceptedAnswers[rid]['title'], code, "0"]) + '\n')
              except:
                print("error")
            except:
              pass


f.close()





# Create training and validation and test sets
os.system('shuf python_all.txt > python_shuffled.txt')
numLines = sum(1 for line in open('python_all.txt'))
trainLines = int(0.8 * numLines)
validLines = int(0.1 * numLines)
testLines = numLines - trainLines - validLines
os.system('head -n ' + str(trainLines) + ' python_shuffled.txt > train.txt')
os.system('tail -n +' + str(trainLines + 1)  + ' python_shuffled.txt | head -n ' + str(validLines)  + ' > valid.txt')
os.system('tail -n +' + str(trainLines + validLines + 1)  + ' python_shuffled.txt  > test.txt')


# Title Labeling

# This is the way I did it. Then I removed the database, so this is deprecated.

# # Get titles for manual labeling
# 
# sqlite3 Posts.sqlite3 "select a.title from Posts a, Posts b where  a.tags like '%c#%' and  a.accepted_answer_id is not null and a.accepted_answer_id = b.id and b.body like '%<code>%' and b.body not like '%<code>%<code>%' order by random() limit 1000; " > titles_1000.txt
# 
# sqlite3 Posts.sqlite3 "select a.title from Posts a, Posts b where  a.tags like '%c#%' and  a.accepted_answer_id is not null and a.accepted_answer_id = b.id and b.body like '%<code>%' and b.body not like '%<code>%<code>%'; " > titles_all.txt
# 
# 
# grep "^g " titles_1000.txt > neg.txt
# grep "^n " titles_1000.txt > pos.txt
# 
# # post
# sed -e "s/^g //" neg.txt | head -n 283 > neg_train.txt
# sed -e "s/^g //" neg.txt | tail -n 283 > neg_test.txt
# 
# sed -e "s/^n //" pos.txt | head -n 116 > pos_train.txt
# sed -e "s/^n //" pos.txt | tail -n 116 > pos_test.txt

