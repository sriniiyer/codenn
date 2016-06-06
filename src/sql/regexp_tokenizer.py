import re

scanner=re.Scanner([
  (r"\[[^\]]*\]",       lambda scanner,token: token),
  (r"\+",      lambda scanner,token:"R_PLUS"),
  (r"\*",        lambda scanner,token:"R_KLEENE"),
  (r"%",        lambda scanner,token:"R_WILD"),
  (r"\^",        lambda scanner,token:"R_START"),
  (r"\$",        lambda scanner,token:"R_END"),
  (r"\?",        lambda scanner,token:"R_QUESTION"),
    (r"[\.~``;_a-zA-Z0-9\s=:\{\}\-\\]+", lambda scanner,token:"R_FREE"),
    (r'.', lambda scanner, token: None),
])

def tokenizeRegex(s):
  results, remainder=scanner.scan(s)
  return results

if __name__ == '__main__':
  print tokenizeRegex("^discount[^(]*\\([0-9]+\\%\\)$")
  print tokenizeRegex("'helloworld'")
