from enthought.traits.api import HasTraits, Instance, Str
from variables import Variables

class DataDecoder(HasTraits):
  """
      Decodes the input stream into a dictionary of name, value pairs which will
      become the variables we can plot.
  """
  name = 'Decoder'
  variables = Instance(Variables) 
  
  def decode(self, data):
    """
        This function gets called when some new data is received from
        the input, here you should decode it and return a dict containing
        variable names and values for the data. Return None if no new data
        is decoded.
    """
    return None
    
  def _receive_callback(self, data):
    new_vars = self.decode(data)
    if new_vars:
      self.variables.update_variables(new_vars)