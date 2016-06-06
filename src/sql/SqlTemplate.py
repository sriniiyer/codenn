import sqlparse
from sql.ParseTypes import *
import pdb
import re
from regexp_tokenizer import tokenizeRegex

class SqlTemplate:

  @staticmethod
  def sanitizeSql(sql):
    s = sql.strip().lower()

    # Add a semi colon at the end
    if not s[-1] == ";":
      s += ';'
    s = re.sub(r'\(', r' ( ', s)
    s = re.sub(r'\)', r' ) ', s)

    # Assume that its a select query and rename keywords
    words = ['index', 'table', 'day', 'year', 'user', 'text']
    for word in words:
      s = re.sub(r'([^\w])' + word + '$', r'\1' + word + '1', s)
      s = re.sub(r'([^\w])' + word + r'([^\w])', r'\1' + word + '1' + r'\2', s)

    # Get rid of hash
    s = s.replace('#', '')
    return s

  def parseStrings(self, tok):
    if isinstance(tok, sqlparse.sql.TokenList):
      for c in tok.tokens:
        self.parseStrings(c)
    elif tok.ptype == STRING:
      if self.regex:
        tok.value = ' '.join(tokenizeRegex(tok.value))
      else:
        tok.value = "CODE_STRING"

  def renameIdentifiers(self, tok):
    if isinstance(tok, sqlparse.sql.TokenList):
      for c in tok.tokens:
        self.renameIdentifiers(c)
    elif tok.ptype == COLUMN:
      if str(tok) not in self.idMap["COLUMN"]:
        colname = "col" + str(self.idCount["COLUMN"])
        self.idMap["COLUMN"][str(tok)] = colname
        self.idMapInv[colname] = str(tok)
        self.idCount["COLUMN"] += 1
      tok.value = self.idMap["COLUMN"][str(tok)]
    elif tok.ptype == TABLE:
      if str(tok) not in self.idMap["TABLE"]:
        tabname = "tab" + str(self.idCount["TABLE"])
        self.idMap["TABLE"][str(tok)] = tabname
        self.idMapInv[tabname] = str(tok)
        self.idCount["TABLE"] += 1
      tok.value = self.idMap["TABLE"][str(tok)]
    elif tok.ptype == FLOAT:
      tok.value = "CODE_FLOAT"
    elif tok.ptype == INTEGER:
      tok.value = "CODE_INTEGER"
    elif tok.ptype == HEX:
      tok.value = "CODE_HEX"

  def __hash__(self):
    return hash(tuple([str(x) for x in self.tokensWithBlanks])) 

  def __init__(self, sql, regex=False, rename=True):  

    self.sql = SqlTemplate.sanitizeSql(sql)

    self.idMap = {"COLUMN" : {}, "TABLE" : {}}
    self.idMapInv = {}
    self.idCount = {"COLUMN" : 0, "TABLE" : 0}
    self.regex = regex

    # parse tree
    self.parseTreeSentinel = False
    self.tableStack = []

    self.parse = sqlparse.parse(self.sql)
    # only single line sql parses. No multiple statements
    self.parse = [self.parse[0]]

    # Type the parse tree
    self.removeWhitespaces(self.parse[0])
    self.identifyLiterals(self.parse[0])
    self.parse[0].ptype = SUBQUERY
    self.identifySubQueries(self.parse[0])
    self.identifyFunctions(self.parse[0])
    self.identifyTables(self.parse[0])

    self.parseStrings(self.parse[0])
    if rename:
      self.renameIdentifiers(self.parse[0])

    self.tokens = SqlTemplate.getTokens(self.parse)

  @staticmethod
  def getTokens(parse):
    flatParse = []
    for expr in parse:
      for token in expr.flatten():
        if token.ptype == STRING:
          flatParse.extend(str(token).split(' '))
        else:
          flatParse.append(str(token))
    return flatParse

  def removeWhitespaces(self, tok):
    if isinstance(tok, sqlparse.sql.TokenList):
      tmpChildren = []
      for c in tok.tokens:
        if not c.is_whitespace():
          tmpChildren.append(c)

      tok.tokens = tmpChildren
      for c in tok.tokens:
        self.removeWhitespaces(c)

  # Mark subquery nodes
  def identifySubQueries(self, tokenList):
    isSubQuery = False

    for tok in tokenList.tokens:
      if isinstance(tok, sqlparse.sql.TokenList):
        subQuery = self.identifySubQueries(tok)  
        if (subQuery and isinstance(tok, sqlparse.sql.Parenthesis)):
          tok.ptype = SUBQUERY
      elif str(tok) == "select":
        isSubQuery = True

    return isSubQuery

  # Run this first
  def identifyLiterals(self, tokenList):
    blankTokens = [sqlparse.tokens.Name,
                   sqlparse.tokens.Name.Placeholder]

    blankTokenTypes = [sqlparse.sql.Identifier]

    for tok in tokenList.tokens:
      if isinstance(tok, sqlparse.sql.TokenList):
        tok.ptype = INTERNAL
        self.identifyLiterals(tok)
      elif (tok.ttype == sqlparse.tokens.Keyword or str(tok) == "select"):
        tok.ptype = KEYWORD
      elif (tok.ttype == sqlparse.tokens.Number.Integer or tok.ttype == sqlparse.tokens.Literal.Number.Integer):
        tok.ptype = INTEGER
      elif (tok.ttype == sqlparse.tokens.Number.Hexadecimal or tok.ttype == sqlparse.tokens.Literal.Number.Hexadecimal):
        tok.ptype = HEX
      elif (tok.ttype == sqlparse.tokens.Number.Float or tok.ttype == sqlparse.tokens.Literal.Number.Float):
        tok.ptype = FLOAT
      elif (tok.ttype == sqlparse.tokens.String.Symbol or tok.ttype == sqlparse.tokens.String.Single or tok.ttype == sqlparse.tokens.Literal.String.Single or tok.ttype == sqlparse.tokens.Literal.String.Symbol):
        tok.ptype = STRING
      elif (tok.ttype == sqlparse.tokens.Wildcard):
        tok.ptype = WILDCARD
      elif (tok.ttype in blankTokens or isinstance(tok, blankTokenTypes[0])):
        tok.ptype = COLUMN
  
  def identifyFunctions(self, tokenList):
    for tok in tokenList.tokens:
      if (isinstance(tok, sqlparse.sql.Function)):
        self.parseTreeSentinel = True
      elif (isinstance(tok, sqlparse.sql.Parenthesis)):
        self.parseTreeSentinel = False
      if self.parseTreeSentinel:
        tok.ptype = FUNCTION
      if isinstance(tok, sqlparse.sql.TokenList):
        self.identifyFunctions(tok)

  def identifyTables(self, tokenList):

    if tokenList.ptype == SUBQUERY:
      self.tableStack.append(False)

    for i in xrange(len(tokenList.tokens)):
      prevtok = tokenList.tokens[i - 1] # Possible bug but unlikely 
      tok = tokenList.tokens[i]

      if (str(tok) == "." and tok.ttype == sqlparse.tokens.Punctuation and prevtok.ptype == COLUMN):
        prevtok.ptype = TABLE

      elif (str(tok) == "from" and tok.ttype == sqlparse.tokens.Keyword):
        self.tableStack[-1] = True

      elif ((str(tok) == "where" or str(tok) == "on" or str(tok) == "group" or str(tok) == "order" or str(tok) == "union") and tok.ttype == sqlparse.tokens.Keyword):
        self.tableStack[-1] = False

      if isinstance(tok, sqlparse.sql.TokenList):
        self.identifyTables(tok)  

      elif (tok.ptype == COLUMN):
        if self.tableStack[-1]:
          tok.ptype = TABLE

    if tokenList.ptype == SUBQUERY:
      self.tableStack.pop()

  def __str__(self):
    return ' '.join([str(tok) for tok in self.tokens])

  def parseSql(self):
    return [str(tok) for tok in self.tokens]
