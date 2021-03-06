from enthought.traits.api import Str
from enthought.traits.ui.api import View, Item
from data_decoder import DataDecoder
import re

class RegexDecoder(DataDecoder):
  """
      Decodes arbitrary text using regex.
  """
  name = Str('Regex Decoder')
  view = View(
    Item(name = 'regex', label='Regex'),
    Item(label= "Each subgroup in the regex is \nassigned to a variable \nin the list in order."),
    Item(name = 'variable_names', label='Group names'),
    Item(label= "(use '_' to ignore a subgroup)"),
    title='Regex decoder'
  )
  regex = Str
  variable_names = Str
  
  def decode(self, data):
    """
        Decode CSV input data then assign variables based on a CSV format list
        list of names, using an '_' to ignore a field.
    """
    try:
      re_result = re.search(self.regex, data)
    except:
      re_result = None
    
    if re_result:
      re_groups = re_result.groups()
      var_names = self.variable_names.split(',')
      
      if len(re_groups) == len(var_names):
        data_dict = {}
        for n, var in enumerate(var_names):
          if var != '_':
            try:
              data_dict[var] = float(re_groups[n])
            except:
              data_dict[var] = re_groups[n]
        return data_dict
      
    return None
    