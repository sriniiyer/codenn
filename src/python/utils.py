

def stripComments(line):
  if '#' in line:
    return line[:line.index('#')].strip()
  else:
    return line

def addBraces(code):
  # First normalize i.e. replace all tabs by 2 spaces
  code = code.replace('\t', '  ')
  code = code.replace('\\n', '\n')

  # Then split into lines
  lines = code.split('\n')

  # append blank line in the end
  lines.append('')

  # Then find initial spacing
  lines[0] = stripComments(lines[0])
  p_space = len(lines[0]) - len(lines[0].lstrip())

  # Now iterate and add braces
  level = 0
  stack = [0]

  for i in range(0, len(lines)):
    lines[i] = stripComments(lines[i])
    spaces = len(lines[i]) - len(lines[i].lstrip()) 
    if spaces > stack[-1]:
      lines[i] = '{ ' + lines[i]
      stack.append(spaces)
    elif spaces < stack[-1]:
      while (spaces < stack[-1]):
        lines[i] = '} ' + lines[i]
        stack.pop()


  return ' '.join(lines)


if __name__ == '__main__':
  print addBraces('def mouse():\n\tprint("cat")\n\tif (True):\n\t\tprint("mouse")\n')
  print '----------------'
  print addBraces('def mouse():\n  print("cat")\n  if (True):\n    print("mouse")\n')
  print '----------------'
  print addBraces('def mouse():\n    print("cat")\n    if (True):\n        print("mouse")\n')
  print '----------------'
  print addBraces('def a():\n  if True:\n    if True:\n      cat\n  def b():\n    if False:\n      print "e"\n  return 34')












