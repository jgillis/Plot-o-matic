from variables import Expression, Variables, HasExpressionTraits

from enthought.traits.api import HasTraits, Str, Regex, Either,This, List, Instance, DelegatesTo, Any, on_trait_change, Float, Range, TraitType
from enthought.traits.ui.api import TreeEditor, TreeNode, View, Item, VSplit, \
  HGroup, Handler, Group, Include, ValueEditor, HSplit, ListEditor, InstanceEditor


from numpy import eye,matrix

#numpy.matrix

class TransformationMatrix(TraitType):
	default_value=eye(4)
	def validate(self,object,name,value):
		try:
			matrix(value)
			return matrix(value)
		except:
			print e
			return None

	def get_editor(self,object):
		return TextEditor()

class Frame(HasTraits):
  parent=This
  T = Instance(Expression)
  name= Str("")
  variables = DelegatesTo('parent')
  world=DelegatesTo('parent')
  
  traits_view = View(
    Item(name = 'name'),
    Item(name = 'parent' , label='Base', editor = InstanceEditor(label="Frame")),
    Item(name = 'T', label = 'Matrix4x4', style = 'custom'),
    title = 'Frame properties'
  )
  
  def evalT(self,lag=0):
    A = self.T.get_historic_value(lag)
    B = self.parent.evalT(lag)
    if A!=None and B!=None:
      return B*A
    else:
      return None

  def __init__(self, parent, T,name=""):
    self.name=name
    self.parent=parent
    if isinstance(T,Expression):
      self.T=T
    else :
      self.T=self.variables.new_expression(T)

class WorldFrame(Frame):
  #Nothing to be seen here
  e=eye(4)
  variables = Instance(Variables)
  world=This
  def evalT(self,lag=0):
    return self.e

  def __init__(self,variables):
    self.variables=variables
    self.name="WorldFrame"
    self.world=self
    #self.parent=None
    #self.T=T

  traits_view = View(
	Item(label="The world is immutable"),
    title = 'WorldFrame'
  )

from numpy import matrix, sin, cos

class FrameHelperFunctions:
  e1=matrix([1,0,0])
  e2=matrix([0,1,0])
  e3=matrix([0,0,1])
  
  def TRx(a):
    return  matrix([[1,0,0,0],[0,cos(a),-sin(a),0],[0,sin(a),cos(a),0],[0,0,0,1]])

  def TRy(a):
    return  matrix([[cos(a),0,sin(a),0],[0,1,0,0],[-sin(a),0,cos(a),0],[0,0,0,1]])

  def TRz(a):
    return  matrix([[cos(a),-sin(a),0,0],[sin(a),cos(a),0,0],[0,0,1,0],[0,0,0,1]])

  def tr(x,y,z):
    return  matrix([[1,0,0,x],[0,1,0,y],[0,0,1,z],[0,0,0,1]])

  def origin() :
    return tr(0,0,0)
    
  def sc(s) :
    return matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1.0/s]])

  def quat(a,b,c,d):
    return matrix([[	a*a+b*b-c*c-d*d,	2*b*c-2*a*d,	2*b*d+2*a*c,	0],
                [	2*b*c+2*a*d,		a*a-b*b+c*c-d*d,2*c*d-2*a*b,	0],
                [	2*b*d-2*a*c,		2*c*d+2*a*b,	a*a-b*b-c*c+d*d,0],
                [	0,			0,		0,		1]])

  def align(x,y,z):
      n=norm(x,y,z)
      v1=matrix([x,y,z])
      if x/n < 0.5:
         v2=cross(v1,e1)
         v2=v2/norm(v2)
         v3=cross(v1,v2)
      else:
         v2=cross(v1,e2)
         v2=v2/norm(v2)
         v3=cross(v1,v2)
      return vstack((v1,v2,v3)).T

  def norm(x,y=None,z=None):
      return norms(x,y,z)

  def norms(x,y=None,z=None):
      if y==None:
         return x[0]**2+y[1]**2+z[2]**2
      else:
         return x**2+y**2+z**2
    
from variables import update_context

update_context(FrameHelperFunctions.__dict__)



