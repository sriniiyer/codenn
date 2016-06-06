
import antlr4
from csharp.CSharp4Lexer import CSharp4Lexer
import re

def parseCSharp(code):
  code = code.replace('\\n', '\n')
  parsedVersion = []
  stream = antlr4.InputStream(code)
  lexer = CSharp4Lexer(stream)
  toks = antlr4.CommonTokenStream(lexer)
  toks.fetch(500)

  identifiers = {}
  identCount = 0
  for token in toks.tokens:
    if token.type == 109:
      parsedVersion += ["CODE_INTEGER"]
    elif token.type == 111:
      parsedVersion += ["CODE_REAL"]
    elif token.type == 112:
      parsedVersion += ["CODE_CHAR"]
    elif token.type == 113:
      parsedVersion += ["CODE_STRING"]
    elif token.type == 9 or token.type == 7 or token.type == 6: # whitespace and comments and newline
      pass
    else:
      parsedVersion += [str(token.text)]

  return parsedVersion


if __name__ == '__main__':
  print parseCSharp("public Boolean SomeValue {     get { return someValue; }     set { someValue = value; } }")
  print parseCSharp("Console.WriteLine('cat'); int mouse = 5; int cat = 0.4; int cow = 'c'; int moo = \"mouse\"; ")
  print parseCSharp("int i = 4;  // i is assigned the literal value of '4' \n int j = i   // j is assigned the value of i.  Since i is a variable,               //it can change and is not a 'literal'")
  try:
    print parseCSharp('string `fixed = Regex.Replace(input, "\s*()","$1");');
  except:
    print "Error"

