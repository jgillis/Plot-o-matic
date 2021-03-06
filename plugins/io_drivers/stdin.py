from io_driver import IODriver
from enthought.traits.api import Str, Float
from enthought.traits.ui.api import View, Item
import sys

class StdinDriver(IODriver):
  """
      Simple driver for taking input from stdin
  """
  name = Str('Stdin Driver')
  view = View(
    title='Stdin input driver'
  )
  
  def receive(self):    
    return sys.stdin.readline() 
