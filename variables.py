from enthought.traits.api import HasTraits, Int, Float, Dict, List, Property, Enum, Color, Instance, Str, Any, on_trait_change, Event, Button, TraitType, DelegatesTo
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

		
def TExpression(mytrait):
	if isinstance(mytrait,DelegatesTo):
		return TExpressionTraitDelegatesTo(mytrait.delegate,mytrait.prefix)
	else:
		return TExpressionTrait(mytrait)

#Possibly we need this to do cloning
class ExpressionTraitListener(HasTraits):
	def Echanged(self, new):
		setattr(self,self.name,new)
		
	def changed(self, new):
		print self.name + "=" ,  new
		self.object.changed(self.name,new)
		
	def __init__(self,object,name):
		self.name = name
		self.object=object
		setattr(self,'_E_' + name + '_changed',self.Echanged)
		print "registering " +'_E_' + name + '_changed'
		print "registering " +'_' + name + '_changed'
		setattr(self,'_' +  name + '_changed',self.changed)

		
class TExpressionTrait(TraitType):
	def validate(self,object,name,value):
		return value
		
	def __init__(self,mytrait):
		self.mytrait=mytrait
		if hasattr(mytrait,'default_value'):
			self.default_value=mytrait.default_value
		TraitType.__init__(self)
	
	def get_editor(self,object):
		return TextEditor()

class TExpressionTraitDelegatesTo(TraitType):
	def __init__(self,delegate,prefix):
		self.delegate=delegate
		self.prefix=prefix
		TraitType.__init__(self)

	def validate(self,object,name,value):
		return value
		
	def get_editor(self,object):
		return TextEditor()
			
class HasExpressionTraits(HasTraits):
	lag=Int(0)
	global TExpression
	def update(self):
		if not(hasattr(self,'_expressionDict')):
			self._expressionDict=dict()
		for k,v in self.traits().items():
			if isinstance(v.handler,TExpressionTrait) or isinstance(v.handler,TExpressionTraitDelegatesTo) :
				if not(self._expressionDict.has_key(k)):
					self._expressionDict[k]=dict()
				if isinstance(v.handler,TExpressionTrait) :
					self.updateExpressionTrait(k,v.handler)
				if isinstance(v.handler,TExpressionTraitDelegatesTo) :
					self.updateExpressionTraitDelegatesTo(k,v.handler)
		
	def updateExpressionTrait(self,name,handler):
		#print "updating updateExpressionTrait"
		if not(self._expressionDict[name].has_key('hasmeta')):
			self._expressionDict[name]['hasmeta']=True
			try:
				self.add_class_trait('E_'+name,handler.mytrait)

				#print "added meta trait"
			except Exception as e:
				#print e
				pass
			#try:
			#	#en=ExpressionTraitListener(self,name))
			#	#self.add_trait_listener(ExpressionTraitListener(self,name))
			#	#self._on_trait_change(lambda s,n: self.Echanged(name,n), 'E_'+name)
			#	#self._on_trait_change(lambda s,n: self.changed(name,n),name)
			#	#print "Added listener for "+ 'E_'+name
			#except Exception as e:
			#	#print e
			#	pass
				
		self.updateExpressionTraitAll(name,handler)
		
	def updateExpressionTraitDelegatesTo(self,name,handler):
		#print "updating updateExpressionTraitDelegatesTo"
		if not(self._expressionDict[name].has_key('hasmeta')):
			self._expressionDict[name]['hasmeta']=True
			try:
				myprefix=handler.prefix
				if myprefix is '':
					myprefix = name
				self.add_class_trait('E_'+name,DelegatesTo(handler.delegate,prefix=myprefix))
				#print "added meta trait delegation " + handler.delegate + " -> " + myprefix
			except Exception as e:
				#print e
				pass
			#try:
			#	#self.add_trait_listener(ExpressionTraitListener(self,name))
			#	#self._on_trait_change(lambda s,n: self.Echanged(name,n), 'E_'+name)
			#	#self._on_trait_change(lambda s,n: self.changed(name,n),name)
			#	print "Added listener for " + 'E_'+name
			#except Exception as e:
			#	#print e
			#	pass
			
		self.updateExpressionTraitAll(name,handler)
	
	def updateExpressionTraitAll(self,name,handler):
		input = getattr(self,name)
		#print name + " input: ", input
		#if self._expressionDict[name].has_key('Ecache'):
		#	if not(self._expressionDict[name]['Ecache'] != getattr(self,'E_'+name)):
		#		print "Forcing a refresh"
		#		self._expressionDict[name]['Ecache']=getattr(self,'E_'+name)
		#		if self._expressionDict[name].has_key('expression'):
		#			del self._expressionDict[name]['expression']
		#		setattr(self,name,getattr(self,'E_'+name))
		#		print "attr " + name + " set to", getattr(self,'E_'+name)
		#		return
		flag=False
		if isinstance(input,str) or isinstance(input,unicode):
			flag=True
			if self._expressionDict[name].has_key('expression'):
				#todo
				self._expressionDict[name]['expression'].set_expr(input)
			else :
				self._expressionDict[name]['expression']=self.variables.new_expression(input)
		elif isinstance(input,Expression):
			flag=True
			self._expressionDict[name]['expression']=input
		else :
			output=input
		
		if flag:
			setattr(self,'Ex_'+name,self._expressionDict[name]['expression'])
			output=self._expressionDict[name]['expression'].get_historic_value(self.lag)
			


		if not(output is None or output is Undefined):
			try:
				setattr(self,'E_'+name,output)
				#self._expressionDict[name]['Ecache']=output
				#print "Set output to ", output
			except Exception as e:
				print e
		
		#if output is None or output == output==Undefined:
		#	print "hy"
		#	setattr(self,name,handler.default_value)
		#	setattr(self,'E_'+name,handler.default_value)
			
	def changed(self,name,value):
		pass
		#print "We registered a change: " , name , " - " , value
		
	def Echanged(self,name,value):
		#print "We registered a change: " , name , " - " , value
		setattr(self,name,value)
		
