#!/usr/bin/env python

import urllib.request, urllib.parse, urllib.error
import urllib.request, urllib.error, urllib.parse

REMOTE_API = 'http://sqlformat.appspot.com/format/'

payload = (
    ('data', 'select * from foo join bar on val1 = val2 where id = 123;'),
    ('format', 'text'),
    ('keyword_case', 'upper'),
    ('reindent', True),
    ('n_indents', 2),
    )


response = urllib.request.urlopen(REMOTE_API,
                           urllib.parse.urlencode(payload))
print(response.read())

