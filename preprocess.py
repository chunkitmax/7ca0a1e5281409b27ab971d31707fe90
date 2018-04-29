'''
preprocess.py
data preprocessing
'''

import re


def clean_str(string):
  '''
  Remove noise from input string
  '''
  # Remove newline characters
  string = re.sub(r'\n|\r', ' ', string)
  # Remove footnotes
  string = re.sub(r'\[Footnote [0-9]+\:.+\]', ' ', string)
  string = re.sub(r'\^[0-9]+', ' ', string)
  # Limit character set
  string = re.sub(r'[^A-Za-z0-9,!?\(\)\.\'\`\"\-]', ' ', string)
  # Let <num> symbolizes numbers
  string = re.sub(r'[0-9]+', ' <num> ', string)
  # Add space around quotes
  string = re.sub(r'( (\'|\") ?)|( ?(\'|\") )', r' \1 ', string)
  # Separate short forms
  string = re.sub(r'(\'s|\'ve|n\'t|\'re|\'d|\'ll|\.|,|!|\?|\(|\))',
                  r' \1 ', string)
  # Add space around '--'
  string = re.sub(r'\-\-', ' -- ', string)
  # Insert newline characters
  string = re.sub(r'(,|\.|!|\?|;) ([^\'\"])', r'\1\n\2', string)
  # Remove consecutive space
  string = re.sub(r'\s{2,}', ' ', string)
  # Lower case
  return string.strip()
