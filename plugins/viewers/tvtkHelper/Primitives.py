from variables import Expression, Variables, HasExpressionTraits, TExpression

from enthought.traits.api import HasTraits, Str, Regex, Either,This, List, Instance, PrototypedFrom,DelegatesTo, Any, on_trait_change, Float, Range, Int, Tuple, Undefined, TraitType, Color
from enthought.traits.ui.api import TreeEditor, TreeNode, View, Item, VSplit, \
  HGroup, Handler, Group, Include, ValueEditor, HSplit, ListEditor, InstanceEditor

from enthought.tvtk.api import tvtk
from plugins.viewers.tools3D.Frame import *

import numpy
from numpy import array, ndarray, linspace, zeros
# actor inherits from Prop3D

import numpy as np

import colorsys

from vtk.util import colors

#class CColor(Color):
#	default_value=(1,1,1)
	
#	def __init__(self,*args,**kwargs):
#		Color.__init__(self,*args,**kwargs)
	
#	def validate(self,object,name,value):
#		if isinstance(value,str) or isinstance(value,unicode):
#			return getattr(colors,value)
#		else:
#			return Color.validate(self,object,name,value)



class NumpyArray(TraitType):
	def validate(self,object,name,value):
		return array(value)

#note: to change color, pull slider all the way accross the range

class Primitive(HasExpressionTraits):
  parent=Instance(Frame)
  T = TExpression(TransformationMatrix)
  polyDataMapper = Instance(tvtk.PolyDataMapper)
  actor = Instance(tvtk.Prop)
  #actor = Instance(tvtk.Actor)
  TM = Instance(numpy.matrix)
  variables=DelegatesTo('parent')
  properties=PrototypedFrom('actor', 'property')
  lag=Int(0)
  
  #This should also add delegated trait objects.
  def handle_arguments(self,*args,**kwargs): 
    HasTraits.__init__(self)		#magic by fnoble
    for a in args:
      if isinstance(a,Frame):
        self.parent=a
      if isinstance(a,str) or isinstance(a,unicode) or isinstance(a,Expression) or isinstance(a,numpy.matrix):
        self.T=a
    for k,v in kwargs.items():
      if k == 'frame':
        self.parent=v
      elif len(self.trait_get(k))>0:
         #self.trait_set({k:v})
         self.__setattr__(k,v)
      elif len(self.actor.trait_get(k))>0:
         self.actor.__setattr__(k,v)
      elif len(self.properties.trait_get(k))>0:
         self.properties.__setattr__(k,v)
      elif len(self.source.trait_get(k))>0:
         self.source.__setattr__(k,v)
      else :
         print "unknown argument", k , v

    if not(self.parent):
      self.parent = WorldFrame()
         
  def __init__(self,**kwargs):
    self.tm=tvtk.Matrix4x4()
    pass
    
  def update(self):
      HasExpressionTraits.update(self)
      TMt=None
      if hasattr(self,'T'):
        if self.E_T !=None :
          p = self.parent.evalT(self.lag)
          if p!=None:
            TMt=np.matrix(p*self.E_T)
          else:
             return
      else:
        p=self.parent.evalT(self.lag)
        if p!=None:
          TMt=np.matrix(p)
        else:
           return
      if TMt is None:
         return
      self.tm.deep_copy(array(TMt).ravel())
      self.actor.poke_matrix(self.tm)
      print "GRRR", self.TM,  TMt
      if not(self.TM is Undefined or self.TM is None):
         if (self.TM!=TMt).any():
            print "updated cache"
            self.TM=TMt
      else:
         self.TM=TMt

  def add_to_scene(self,sc):
       sc.add_actors(self.actor)
       self.scene=sc
       
  def remove_from_scene(self):
       self.scene.remove_actors(self.actor)

  def setall(self,attr,value):
       setattr(self,attr,value)
       
class Cone(Primitive):
  source = Instance(tvtk.ConeSource)
  height= TExpression(DelegatesTo('source'))
  radius= TExpression(DelegatesTo('source'))
  resolution= TExpression(DelegatesTo('source'))
  traits_view = View(
    Item(name = 'parent', label='Frame'),
    Item(name = 'T', label = 'Matrix4x4', style = 'custom'),
    Item(name = 'height'),
    Item(name = 'radius'),
    Item(name = 'resolution'),
    Item(name = 'properties', editor=InstanceEditor(), label = 'Render properties'),
    title = 'Cone properties'
  )
  def __init__(self,*args,**kwargs):
    Primitive.__init__(self,**kwargs)
    self.source = tvtk.ConeSource()
    self.polyDataMapper = tvtk.PolyDataMapper()
    self.polyDataMapper.input=self.source.output
    self.actor = tvtk.Actor(mapper=self.polyDataMapper)
    self.handle_arguments(*args,**kwargs)
    
class Box(Primitive):
  source = Instance(tvtk.CubeSource)
  x_length=TExpression(DelegatesTo('source'))
  y_length=TExpression(DelegatesTo('source'))
  z_length=TExpression(DelegatesTo('source'))

  traits_view = View(
    Item(name = 'parent', label='Frame'),
    Item(name = 'T', label = 'Matrix4x4', style = 'custom'),
    Item(name = 'x_length'),
    Item(name = 'y_length'),
    Item(name = 'z_length'),
    Item(name = 'properties',editor=InstanceEditor(), label = 'Render properties'),
    title = 'Box properties'
  )
  def __init__(self,*args,**kwargs):
    Primitive.__init__(self,**kwargs)
    self.source = tvtk.CubeSource()
    self.polyDataMapper = tvtk.PolyDataMapper()
    self.polyDataMapper.input=self.source.output
    
    self.actor = tvtk.Actor(mapper=self.polyDataMapper)
    self.handle_arguments(*args,**kwargs)
    
    
class Axes(Primitive):
  source = Instance(tvtk.Axes)
  tube = Instance(tvtk.TubeFilter)
  
  scale_factor=DelegatesTo('tube')
  radius=TExpression(DelegatesTo('tube'))
  sides=PrototypedFrom('tube','number_of_sides')
  
  traits_view = View(
    Item(name = 'parent', label='Frame'),
    Item(name = 'T', label = 'Matrix4x4', style = 'custom'),
    Item(name = 'properties',editor=InstanceEditor(), label = 'Render properties'),
    title = 'Axes properties'
  )
  def __init__(self,*args,**kwargs):
    Primitive.__init__(self,*kwargs)
    self.source = tvtk.Axes(symmetric=1)
    self.tube = tvtk.TubeFilter(vary_radius='vary_radius_off',input=self.source.output)
    self.mapper = tvtk.PolyDataMapper(input=self.tube.output)
    self.actor = tvtk.Actor(mapper=self.mapper)
    self.handle_arguments(*args,**kwargs)
    

class Cylinder(Primitive):
  source = Instance(tvtk.CylinderSource)
  height= TExpression(DelegatesTo('source'))
  radius= TExpression(DelegatesTo('source'))
  resolution= TExpression(DelegatesTo('source'))
  traits_view = View(
    Item(name = 'parent', label='Frame'),
    Item(name = 'T', label = 'Matrix4x4', style = 'custom'),
    Item(name = 'height'),
    Item(name = 'radius'),
    Item(name = 'resolution'),
    Item(name = 'properties', editor=InstanceEditor(), label = 'Render properties'),
    title = 'Cylinder properties'
  )
  def __init__(self,*args,**kwargs):
    Primitive.__init__(self,*kwargs)
    self.source = tvtk.CylinderSource()
    self.mapper = tvtk.PolyDataMapper(input=self.source.output)
    self.actor = tvtk.Actor(mapper=self.mapper)
    self.handle_arguments(*args,**kwargs)

class Sphere(Primitive):
  source=Instance(tvtk.SphereSource)
  radius=TExpression(DelegatesTo('source'))
  theta_resolution=DelegatesTo('source')
  phi_resolution=DelegatesTo('source')
  traits_view = View(
    Item(name = 'parent', label='Frame'),
    Item(name = 'T', label = 'Matrix4x4', style = 'custom'),
    Item(name = 'radius'),
    Item(name = 'theta_resolution'),
    Item(name = 'phi_resolution'),
    Item(name = 'properties', editor=InstanceEditor(), label = 'Render properties'),
    title = 'Sphere properties'
  )
  def __init__(self,*args,**kwargs):
    Primitive.__init__(self,**kwargs)
    self.source = tvtk.SphereSource()
    self.mapper = tvtk.PolyDataMapper(input=self.source.output)
    self.actor = tvtk.Actor(mapper=self.mapper)
    self.handle_arguments(*args,**kwargs)
    
class Point(Sphere):
	def __init__(self,*args,**kwargs):
		Sphere.__init__(self,**kwargs)
		self.radius=0.05
		
class Arrow(Primitive):
   source=Instance(tvtk.ArrowSource)
   tip_resolution = DelegatesTo("source")
   point1=TExpression(NumpyArray)
   point2=TExpression(NumpyArray)
   traits_view = View(
    Item(name = 'parent', label='Frame'),
    Item(name = 'from',style = 'custom'),
    Item(name = 'to',style = 'custom'),
    Item(name = 'tip_resolution'),
    Item(name = 'source', editor=InstanceEditor(), label = 'Geometric properties'),
    Item(name = 'properties', editor=InstanceEditor(), label = 'Render properties'),
    title = 'Arrow properties'
   )
   def __init__(self,*args,**kwargs):
    Primitive.__init__(self,**kwargs)
    self.source = tvtk.ArrowSource()
    self.mapper = tvtk.PolyDataMapper(input=self.source.output)
    self.actor = tvtk.Actor(mapper=self.mapper)
    self.handle_arguments(*args,**kwargs)
    
    

class Plane(Primitive):
   source=Instance(tvtk.PlaneSource)
   traits_view = View(
    Item(name = 'parent', label='Frame'),
    Item(name = 'T', label = 'Matrix4x4', style = 'custom'),
    Item(name = 'properties', editor=InstanceEditor(), label = 'Render properties'),
    Item(name = 'source', editor=InstanceEditor(), label = 'Geometric properties'),
    title = 'Plane properties'
   )
   def __init__(self,*args,**kwargs):
    Primitive.__init__(self,**kwargs)
    self.source = tvtk.PlaneSource()
    self.mapper = tvtk.PolyDataMapper(input=self.source.output)
    self.actor = tvtk.Actor(mapper=self.mapper)
    self.handle_arguments(*args,**kwargs)
    
class Line(Primitive):
   source=Instance(tvtk.LineSource)
   point1=TExpression(DelegatesTo('source'))
   point2=TExpression(DelegatesTo('source'))
   traits_view = View(
    Item(name = 'parent', label='Frame'),
    Item(name = 'T', label = 'Matrix4x4', style = 'custom'),
    Item(name = 'point1', style = 'custom'),
    Item(name = 'point2', style = 'custom'),
    Item(name = 'source', editor=InstanceEditor(), label = 'Source properties'),
    Item(name = 'properties', editor=InstanceEditor(), label = 'Render properties'),
    title = 'Line properties'
   )
   def __init__(self,*args,**kwargs):
    Primitive.__init__(self,**kwargs)
    self.source = tvtk.LineSource()
    self.mapper = tvtk.PolyDataMapper(input=self.source.output)
    self.actor = tvtk.Actor(mapper=self.mapper)
    self.handle_arguments(*args,**kwargs)
    
class ProjectedPoint(Line):
  point=TExpression(Either(List,Tuple))
  point1=None
  point2=None
  traits_view = View(
    Item(name = 'parent', label='Frame'),
    Item(name = 'T', label = 'Matrix4x4', style = 'custom'),
    Item(name = 'point', style = 'custom'),
    Item(name = 'source', editor=InstanceEditor(), label = 'Source properties'),
    Item(name = 'properties', editor=InstanceEditor(), label = 'Render properties'),
    title = 'ProjectionLine properties'
   )
  def __init__(self,*args,**kwargs):
    if kwargs.has_key('point'):
      self.point=point
      del kwargs['point']
      Line.__init__(self,**kwargs)
      
  def __init__(self,*args,**kwargs):
    Line.__init__(self,*args,**kwargs)
    
  def _E_point_changed(self,new):
    if self.E_point is Undefined:
      pass
    self.source.point2=[self.E_point[0],self.E_point[1],0]
    self.source.point1=self.E_point



# The following code is mainly from 
# http://www.enthought.com/~rkern/cgi-bin/hgwebdir.cgi/colormap_explorer/file/a8aef0e90790/colormap_explorer/colormaps.py
class PolyLine(Primitive):
   source=Instance(tvtk.PolyData)
   points=Instance(numpy.ndarray)
   traits_view = View(
    Item(name = 'parent', label='Frame'),
    Item(name = 'properties', editor=InstanceEditor(), label = 'Render properties'),
    title = 'Line properties'
   )
   def __init__(self,*args,**kwargs):
    Primitive.__init__(self,**kwargs)
    self.source = tvtk.PolyData()
    self.mapper = tvtk.PolyDataMapper(input=self.source)
    self.actor = tvtk.Actor(mapper=self.mapper)
    self.handle_arguments(*args,**kwargs)
    #kwargs.get('foo', 12)  fnoble cleverity

   def _points_changed(self,old, new):
    npoints = len(self.points)
    if npoints<2 :
      return
    lines = np.zeros((npoints-1, 2), dtype=int)
    lines[:,0] = np.arange(0, npoints-1)
    lines[:,1] = np.arange(1, npoints)
    self.source.points=self.points
    self.source.lines=lines
    
#tvtk.DataSetMapper allows fancy things with colors, but not with opacity
#actor.set_lut(mm.scalar_lut_manager.lut)

class FadePolyLine(Primitive):
   source=Instance(tvtk.PolyData)
   points=Instance(numpy.ndarray)
   mapper=Instance(tvtk.DataSetMapper)
   actor=Instance(tvtk.Actor)
   color=Color

   traits_view = View(
    Item(name = 'parent', label='Frame'),
    Item(name = 'color'),
    Item(name = 'properties', editor=InstanceEditor(), label = 'Render properties'),
    title = 'Line properties'
   )
   def __init__(self,*args,**kwargs):
    Primitive.__init__(self,**kwargs)
    self.source = tvtk.PolyData()
    self.mapper = tvtk.DataSetMapper(input=self.source)
    self.actor = tvtk.Actor(mapper=self.mapper)
    self.lut=self.mapper.lookup_table
    self.lut.alpha_range=(0,1)
    self.handle_arguments(*args,**kwargs)
    
   def _color_changed(self,color):
      hsv=colorsys.rgb_to_hsv(color[0]/256.0,color[1]/256.0,color[2]/256.0)
      print color , " -> " , hsv
      self.lut.hue_range=(hsv[0],hsv[0])
      self.lut.saturation_range=(hsv[1],hsv[1])
      self.lut.value_range=(hsv[2],hsv[2])
      
   def _points_changed(self,old, new):
    npoints = len(self.points)
    if npoints<2 :
      return
    lines = np.zeros((npoints-1, 2), dtype=int)
    lines[:,0] = np.arange(0, npoints-1)
    lines[:,1] = np.arange(1, npoints)
    self.source.points=self.points
    self.source.lines=lines
    self.source.point_data.scalars=np.linspace(0,1,npoints)

class Circle(PolyLine):
   radius=TExpression(Float)
   resolution=Int(100)
   points=Instance(numpy.ndarray)
   traits_view = View(
    Item(name = 'parent', label='Frame'),
    Item(name = 'T', label = 'Matrix4x4', style = 'custom'),
    Item(name = 'radius', style = 'custom'),
    Item(name = 'resolution'),
    Item(name = 'properties', editor=InstanceEditor(), label = 'Render properties'),
    title = 'Circle properties'
   )
   def __init__(self,*args,**kwargs):
     PolyLine.__init__(self,*args,**kwargs)
     
   def _E_radius_changed(self):
     self.calc()
     
   def _resolution_changed(self):
     self.calc()
     
   def calc(self):
     t=linspace(0,6.29,self.resolution)
     if not(self.E_radius is Undefined) :
     	self.points = array([self.E_radius*sin(t),self.E_radius*cos(t),zeros(t.shape)]).T
class Trace(PolyLine):
   point=TExpression(NumpyArray)
   points=Instance(numpy.ndarray)
   length = Int(0)
   
   traits_view = View(
    Item(name = 'length', label='Frame'),
    Item(name = 'point', style = 'custom'),
    Item(name = 'properties', editor=InstanceEditor(), label = 'Render properties'),
    title = 'Line properties'
   )
   def __init__(self,*args,**kwargs):
     PolyLine.__init__(self,*args,**kwargs)
     
   def _E_point_changed(self,new):
     if hasattr(self,'Ex_point'):
       expression=self.Ex_point
       #fix this
       #history=expression.get_array(first=-3*self.length)
       #print history, history.shape
       #print history.reshape((3,history.shape[0]/3))

class ProjectedPolyLine(Primitive):
	from numpy import *
	watch=Instance(PolyLine)
	watchpoints=DelegatesTo('watch',prefix='points')
	watchTM=DelegatesTo('watch',prefix='TM')
	pd=Instance(tvtk.PolyData)
	polypoints=DelegatesTo('pd',prefix='points')
	polys=DelegatesTo('pd')
	
	traits_view = View(
	 Item(name = 'watch', editor=InstanceEditor()),
	 Item(name = 'properties', editor=InstanceEditor(), label = 'Render properties'),
	 title = 'ProjectedPolyLine properties'
	)
	def __init__(self,*args,**kwargs):
		Primitive.__init__(self,**kwargs)
		self.pd=tvtk.PolyData()
		self.mapper = tvtk.PolyDataMapper(input=self.pd)
		self.actor = tvtk.Actor(mapper=self.mapper)
		self.handle_arguments(*args,**kwargs)
		
	def _watchpoints_changed(self,new):
		self.calc()

	def _watchTM_changed(self,new):
		self.calc()

	def calc(self):
		project=lambda x: (x[0],x[1],0)
		q=self.watchpoints
		if not(q is Undefined or q is None):
			if not(self.watch.TM is None):
				q=array((self.watch.TM*np.hstack((q,np.ones((q.shape[0],1)))).T).T)
				n=q[:,3]
				q=q[:,0:3]/(np.vstack((n,n,n)).T+0.0)
			self.polypoints=np.hstack((q,map(project,q))).reshape((q.shape[0]*2,3))
			self.polys=array([[i*2,2*i+1,2*i+3,2*i+2] for i in range(q.shape[0]-1)])


     

class Text(Primitive):
  text=DelegatesTo('source')
  traits_view = View(
    Item(name = 'parent', label='Frame'),
    Item(name = 'T', label = 'Matrix4x4', style = 'custom'),
    Item(name = 'text'),
    Item(name = 'properties', editor=InstanceEditor(), label = 'Render properties'),
    title = 'Text properties'
   )
  def __init__(self,*args,**kwargs):
    Primitive.__init__(self,**kwargs)
    self.source = tvtk.VectorText()
    self.mapper = tvtk.PolyDataMapper(input=self.source.get_output())
    self.actor = tvtk.Actor(mapper=self.mapper)
    self.handle_arguments(*args,**kwargs)
    #kwargs.get('foo', 12)  fnoble cleverity

class Image(Primitive):
    source=Instance(tvtk.ImageReader)
    file_name=DelegatesTo('source')
    traits_view = View(
    Item(name = 'parent', label='Frame'),
    Item(name = 'T', label = 'Matrix4x4', style = 'custom'),
    Item(name = 'file_name'),
    Item(name = 'source', editor=InstanceEditor()),
    Item(name = 'actor', editor=InstanceEditor()),
    Item(name = 'properties', editor=InstanceEditor(), label = 'Render properties'),
    title = 'Image properties'
    )
    def __init__(self,*args,**kwargs):
        Primitive.__init__(self,*kwargs)
        self.source=tvtk.ImageReader(file_name="woodpecker.bmp") # im.ouput
        self.source.set_data_scalar_type_to_unsigned_char()
        #self.mapper = tvtk.ImageMapper(input=self.source.output)
        #self.actor = tvtk.Actor2D(mapper=self.mapper)
        self.actor=tvtk.ImageActor(input=self.source.output)
        self.handle_arguments(*args,**kwargs)
    
    
    
#http://mayavi2.sourcearchive.com/documentation/3.3.0-2/actors_8py-source.html
#source = tvtk.ArrowSource(tip_resolution=resolution,
#                              shaft_resolution=resolution)

# LineSource, PlaneSource

class PrimitiveCollection(HasTraits):
  #primitives=List(Either(Primitive,This))
  primitives=List(Instance(HasTraits))
  T=TExpression(TransformationMatrix)
  frame=Instance(Frame)
  
  traits_view = View(
    Item(name = 'parent', label='Frame'),
    Item(name = 'T', label = 'Matrix4x4', style = 'custom'),
    Item(name = 'primitives', editor=ListEditor(),style='custom'),
    title = 'Collection properties'
   )
   
  def getPrimitives(self):
    return self.primitives
    
  def __init__(self,frame,T=None):
    if not(T is None):
      self.T=T
    self.frame=frame
      
  def add(self, arg):
    if isinstance(arg,list):
      map(self.add,arg)
    if isinstance(arg,Primitive):
      self.primitives.append(arg)
    if isinstance(arg,PrimitiveCollection):
      #self.add(arg.getPrimitives()) 
      self.primitives.append(arg)

  def update(self):
       map(lambda x: x.update(),self.primitives)
      
  def add_to_scene(self,sc):
       self.scene=sc
       map(lambda x: x.add_to_scene(sc),self.primitives)
       
  def remove_from_scene(self):
       map(lambda x: x.remove_from_scene(),self.primitives)
       self.primitives=[]
       
  def setall(self,attr,value):
       map(lambda x: setattr(x,attr,value),self.primitives)


class Strobe(PrimitiveCollection):
	length=Int(10)
	step=Int(1)
	template=Either(Primitive,PrimitiveCollection)
	
	traits_view = View(
	    Item(name = 'length'),
	    Item(name = 'step'),
	    Item(name = 'template', editor=InstanceEditor(),style='custom'),
	    title = 'Strobe properties'
	   )
	   
	def __init__(self,template):
		self.template=template
	
	def _length_changed(self):
		self.setup()
		
	def _step_changed(self):
		self.setup()
		
	def setup(self):
		if hasattr(self,'scene'):
			self.remove_from_scene()
			for lag in range(0,length*step,step):
				cl=Lag(self.template,lag)
				self.add(cl)
			self.add_to_scene(self.scene)
	
def Lag(shape,lag):
	cl=shape.clone_traits()
	cl.setall('lag',lag)
	return cl
		
	