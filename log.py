'''
log.py
Manager log
'''

class Logger:
  '''
  Verbose level:
  0 - Only log important message
  1 - All debug information
  '''
  def __init__(self, verbose_level=1):
    self.verbose_level = verbose_level
    self.end_str = ['\033[K', '\033[K\n']
  def d(self, msg, new_line=False, reset_cursor=True):
    '''
    Print debug message
    '''
    prefix = ''
    if reset_cursor:
      prefix = '\r'
    if self.verbose_level > 0:
      print(prefix+str(msg), end=self.end_str[int(new_line)])
  def i(self, msg, new_line=False, reset_cursor=True):
    '''
    Print important message
    '''
    prefix = ''
    if reset_cursor:
      prefix = '\r'
    print(prefix+str(msg), end=self.end_str[int(new_line)])
