from enthought.traits.api import HasTraits, Int, Float, Dict, List, Property, Enum, Color, Instance, Str, Any, on_trait_change, Event, Button, TraitType, DelegatesTo, Bool
from enthought.traits.ui.api import View, Item, ValueEditor, TabularEditor, HSplit, TextEditor
from enthought.traits.trait_base import Undefined
from enthought.traits.ui.tabular_adapter import TabularAdapter
import time

import math, numpy
import cPickle as pickle

expression_context = {}
expression_context.update(numpy.__dict__)


def update_context(context):
  expression_context.update(context)

class VariableTableAdapter(TabularAdapter):
  columns = [('Variable name', 0), ('Value', 1)]

class Variables(HasTraits):
  vars_pool = {}
  vars_list = List()
  vars_table_list = List()  # a list version of vars_pool maintained for the TabularEditor
  vars_table_list_update_time = Float(0)

  sample_number = Int(0)
  sample_count = Int(0)
  max_samples = Int(20000)

  start_time = time.time()
  
  add_var_event = Event()

  expressions = List()

  clear_button = Button('Clear')
  view = View(
           HSplit(
             Item(name = 'clear_button', show_label = False),
             Item(name = 'max_samples', label = 'Max samples'),
             Item(name = 'sample_count', label = 'Samples')
           ),
           Item(
             name = 'vars_table_list',
             editor = TabularEditor(
               adapter = VariableTableAdapter(),
               editable = False,
               dclicked = "add_var_event"
             ),
             resizable = True,
             show_label = False
           ),
           title = 'Variable view',
           resizable = True,
           width = .7,
           height = .2
         )
  
  def new_expression(self, expr):
    new_expression = Expression(self, expr)
    self.expressions.append(new_expression)
    return new_expression
    
  def update_variables(self, data_dict):
    """
        Receive a dict of variables from a decoder and integrate them
        into our global variable pool.
    """
    self.sample_number += 1
    
    # We update into a new dict rather than vars_pool due to pythons pass by reference
    # behaviour, we need a fresh object to put on our array
    new_vars_pool = {}
    new_vars_pool.update(self.vars_pool)
    new_vars_pool.update(data_dict)
    new_vars_pool.update({'sample_num': self.sample_number, 'system_time': time.time(), 'time': time.time() - self.start_time})
    if '' in new_vars_pool: 
      del new_vars_pool[''] # weed out undesirables

    self.vars_list.append(new_vars_pool)
    self.update_vars_list()

  def update_vars_list(self): 
    self.vars_pool = self.vars_list[-1]

    if time.time() - self.vars_table_list_update_time > 0.2:
      self.vars_table_list_update_time = time.time()
      self.update_vars_table()

    self.sample_count = len(self.vars_list)
    if self.sample_count > self.max_samples:
      self.vars_list = self.vars_list[-self.max_samples:]
      self.sample_count = self.max_samples
      
  @on_trait_change('clear_button')
  def clear(self):
    """ Clear all recorded data. """
    self.sample_number = 0
    self.vars_list = [{}]
    self.update_vars_list()
    self.update_vars_table()
    self.start_time = time.time()

    for expression in self.expressions:
      expression.clear_cache()

  def save_data_set(self, filename):
    fp = open(filename, 'wb')
    pickle.dump(self.vars_list, fp, True)
    fp.close() 

  def open_data_set(self, filename):
    fp = open(filename, 'rb')
    self.vars_list = pickle.load(fp)
    fp.close() 
    
    self.update_vars_list()
    self.update_vars_table()
    self.sample_number = self.sample_count
    # spoof start time so that we start where we left off
    self.start_time = time.time() - self.vars_list[-1]['time']

  def update_vars_table(self):
    vars_list_unsorted = [(name, repr(val)) for (name, val) in list(self.vars_pool.iteritems())]
    self.vars_table_list = sorted(vars_list_unsorted, key=(lambda x: x[0].lower()))
    
  def _eval_expr(self, expr, vars_pool=None):
    """
        Returns the value of a python expression evaluated with 
        the variables in the pool in scope. Used internally by
        Expression. Users should use Expression instead as it
        has caching etc.
    """
    if vars_pool == None:
      vars_pool = self.vars_pool

    try:
      data = eval(expr, expression_context, vars_pool)
    except:
      data = None
    return data

  def bound_array(self, first, last):
    if first < 0:
      first += self.sample_number
      if first < 0:
        first = 0
    if last and last < 0:
      last += self.sample_number
    if last == None:
      last = self.sample_number

    return (first, last)

  def _get_array(self, expr, first=0, last=None):
    """
        Returns an array of tuples containing the all the values of an
        the supplied expression and the sample numbers and times corresponding to
        these values. Used internally by Expression, users should use Expression
        directly as it has caching etc.
    """

    first, last = self.bound_array(first, last)
    if expr in self.vars_pool:
      data = [vs.get(expr) for vs in self.vars_list[first:last]]
    else:
      data = [self._eval_expr(expr, vs) for vs in self.vars_list[first:last]]
    data = [d for d in data if d is not None]
    
    #try:
    #  data = [try: eval(expr, expression_context, vs); except: pass; for vs in self.vars_list[first:last]]
    #except Exception as e:
    #  print e
    #  data = []

    data_array = numpy.array(data)
    return data_array

class Expression(HasTraits):
  _vars = Instance(Variables)
  _expr = Str('')
  _data_array_cache = numpy.array([])
  _data_array_cache_index = Int(0)

  view = View(Item('_expr', show_label = False, editor=TextEditor(enter_set=True, auto_set=False)))

  def __init__(self, variables, expr, **kwargs):
    HasTraits.__init__(self, **kwargs)
    self._vars = variables
    self.set_expr(expr)

  def set_expr(self, expr):
    if self._expr != expr:
      self._expr = expr

  def __expr_changed(self):
    self.clear_cache()

  def clear_cache(self):
    self._data_array_cache = numpy.array([])
    self._data_array_cache_index = 0

  def get_curr_value(self):
    return self._vars._eval_expr(self._expr)
    
  def get_historic_value(self,lag):
    if lag==0:
        return self.get_curr_value()
    if len(self._vars.vars_list) > lag:
        vars_pool = self._vars.vars_list[-lag-1]
        return self._vars._eval_expr(self._expr,vars_pool)
    return None

  def get_array(self, first=0, last=None):
    first, last = self._vars.bound_array(first, last)
    
    if last > self._data_array_cache_index:
      #print "Cache miss of", (last - self._data_array_cache_index)
      if self._data_array_cache.shape[0]==0:
        self._data_array_cache =self._vars._get_array(self._expr, self._data_array_cache_index, last)
      else: 
        self._data_array_cache = numpy.vstack((self._data_array_cache, self._vars._get_array(self._expr, self._data_array_cache_index, last)))
      self._data_array_cache_index = last

    return self._data_array_cache[first:last,...]
    
# http://code.enthought.com/projects/files/ETS31_API/enthought.traits.trait_types.Instance.html
# set_value is the magic word

"""

TExpression

class Foo(HasExpressionTraits)
	bar=TExpression(Float)
	baz=Float
	
f=Foo()

f.bar = 'time'     or  f.bar = variables.new_expression('time')
print f.bar -> returns the time

f.bar = 3
print f.bar -> returns 3, quickly

When editing, we see the expression's string value

class Bar(HasExpressionTraits)
	foo = Instance(Foo)
	baz= DelegatesTo("foo")
	
b=Bar()

b.baz = 'time'
f.baz will now contain the time

How TEpression works

The main idea is to have an object that stores your data (Expression, value, ...) called TExpressionWrapper.
When getting, setting or editing the TExpression, we should have the possibility to add a layer of functionality.
That's why we have a TExpressionInstance traittype that overwrites default set, get, validate methods
The trick here is in set. You cannot possible do:
    'object.__dict__[name] =value'  as this will kill the get/set functionality
    'setattr(object,name,value)' as this will cause an infinite recursion
    
Add this point, the user should be able to write:
TExpressionInstance(TExpressionWrapper,mytrait)

However, we introduce a layer of syntax sugar so you can write TExpression(mytrait) instead

Lastly, to receives updates from plot-o-matic, the user has to update all expression traits.
We introduce a class HasExpressionTraits which takes care of this when called with HasExpressionTraits.update(self)

"""


TraitsCache ="_traits_cache_"
class TExpressionInstance(Instance):
	def __init__(self,myclass, mytrait):
		Instance.__init__(self,myclass)
		self.mytrait=mytrait
		self.default_value=mytrait.default_value
				
	def set(self,object,name,value):
		W= None
		if (object.__dict__.has_key(TraitsCache + name)):
			W= object.__dict__[TraitsCache + name]
		else :
			W=TExpressionWrapper()
			self.set_value(object,name,W)
			W.initialize(object,self.mytrait,name)
		W.set(value)
		object.trait_property_changed( name, value, value )
		return value
		
	validate=set
		
	def get(self,object,name):
		if (object.__dict__.has_key(TraitsCache + name)):
			return object.__dict__[TraitsCache + name].get()
			
	def get_editor(self,trait = None):
		return TextEditor()
	

def TExpression(mytrait):
	return TExpressionInstance(TExpressionWrapper,mytrait)
		
class HasExpressionTraits(HasTraits):
	lag=Int(0)
	global TExpression
	def update(self):
		for k,v in self.traits().items():
			if isinstance(v.handler,TExpressionInstance) :
				if (self.__dict__.has_key(TraitsCache + k)):
					self.__dict__[TraitsCache + k].update()
				else:
					setattr(self,k,v.handler.default_value)
					
	def getExpression(self,name):
		if (self.__dict__.has_key(TraitsCache + name)):
			return self.__dict__[TraitsCache + name].expression
		else:
			return None
	
class TExpressionWrapper(HasExpressionTraits):
	parent = Instance(HasTraits)
	lag = DelegatesTo('parent')
	variables = DelegatesTo('parent')
	expression = Instance(Expression)
	_expr = DelegatesTo('expression')
	is_initialized = Bool(False)
	myTrait = Any
	is_pure = Bool(True)
	value = Any
	name =Str

	view = View(Item('name', show_label = False, editor=TextEditor(enter_set=True, auto_set=True)))
	
	
	def  handle_delegates(self):
		if self.value is Undefined or self.value is None:
			return
		self.parent.trait_property_changed( self.name, self.value, self.value )
		if (isinstance(self.mytrait, DelegatesTo)):
			delegate=getattr(self.parent,self.mytrait.delegate)
			myprefix=self.mytrait.prefix
			if myprefix is '':
				myprefix = self.name
			setattr(delegate,myprefix,self.value)
	
	def initialize(self,parent,trait,name):
		self.parent = parent
		self.mytrait   = trait
		self.is_initialized   = True
		self.name = name
		
	def set(self,value):
		if (self.is_expr_string(value)):
			self.expression=self.variables.new_expression(value)
			self.is_pure = False
		elif (isinstance(value,Expression)):
			self.expression = value
			self.is_pure = False
		else:
			self.value = value
			self.is_pure = True
		self.handle_delegates()
			
	def get(self):
		if (self.is_pure):
			return self.value
		else:
			return self.expression.get_historic_value(self.lag)
			
	def is_expr_string(self,input):
		if isinstance(input,str) or isinstance(input,unicode):
			return True
		return False
		
	def update(self):
		if (self.is_pure):
			return
		self.value = self.get()
		if self.value is Undefined or self.value is None:
			return
		self.handle_delegates()

