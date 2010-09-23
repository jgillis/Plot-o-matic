from variables import Expression, Variables, HasExpressionTraits, TExpression

from enthought.traits.api import HasTraits, Str, Regex, Either,This, List, Instance, PrototypedFrom,DelegatesTo, Any, on_trait_change, Float, Range, Int, Tuple, Undefined, TraitType, Color
from enthought.traits.ui.api import TreeEditor, TreeNode, View, Item, VSplit, \
  HGroup, Handler, Group, Include, ValueEditor, HSplit, ListEditor, InstanceEditor, ColorEditor
  
from enthought.tvtk.api import tvtk
from plugins.viewers.tools3D.Frame import *

from numpy import array, ndarray, linspace, zeros, eye, matrix, zeros, ones, arange, linspace, hstack, vstack, ndarray
# actor inherits from Prop3D

import colorsys
from vtk.util import colors

from enthought.enable.colors import ColorTrait
from enthought.mayavi.sources.image_reader import ImageReader


class NumpyArray(TraitType):
	"""
	Trait type for numpy.arrays
	"""
	def validate(self,object,name,value):
		return array(value)

class VisualObject(HasExpressionTraits):
	""" 
	A baseclass for Primitives and PrimitiveCollection
	"""
	frame=Instance(Frame)
	T = TExpression(TransformationMatrix)
	variables=DelegatesTo('frame')
	lag=Int(0)
	e=eye(4)
	def __init__(self,*args,**kwargs):
		self.inits=(args,kwargs)

	def clone(self):
		return self.__class__(*self.inits[0],**self.inits[1])


class Primitive(VisualObject):
  """
  A primitive object is the most basic TVTK drawable object
  
  Each primitive takes a parent of type Frame and possible a transformation matrix T
  
  Common properties of Primitives that can be set by passing keyword arguments to the constructor:
  * T: a 4x4 homogeneous transformation matrix
  * color : tuple (1,1,1) for white, or string keyword
  * opacity : float [0,1]
  * 
  
  """
  polyDataMapper = Instance(tvtk.PolyDataMapper)
  actor = Instance(tvtk.Prop)
  TM = Instance(matrix)
  properties=PrototypedFrom('actor', 'property')

  
  
  #This should also add delegated trait objects.
  def handle_arguments(self,*args,**kwargs):
    VisualObject.__init__(self,*args,**kwargs)
    """
    Do smart handling of keyword arguments
    
    Keywords are matched against members of the following objects:
       * The Primitive itself
       * The Primitive's actor
       * The Primitive's properties - rendering options
             Typical: opacity [0,1], color "red" or  (1,0,0)
       * The Primitive's source
    """
    HasTraits.__init__(self)		#magic by fnoble
    for a in args:
      if isinstance(a,Frame):
        self.frame=a
      if isinstance(a,str) or isinstance(a,unicode) or isinstance(a,Expression) or isinstance(a,matrix):
        self.T=a
    for k,v in kwargs.items():
      if k == 'frame':
        self.frame=v
      elif len(self.trait_get(k))>0:
         #self.trait_set({k:v})
         setattr(self,k,v)
      elif len(self.actor.trait_get(k))>0:
         setattr(self.actor,k,v)
      elif len(self.properties.trait_get(k))>0:
         if k=="color" and isinstance(v,str):
           self.properties.color=getattr(colors,v)
         else:
           setattr(self.properties,k,v)
      elif len(self.source.trait_get(k))>0:
         setattr(self.source,k,v)
      else :
         print "unknown argument", k , v

    if not(self.frame):
      raise Exception('All primitives must have a frame', 'All primitives must have a frame')
         
  def __init__(self,**kwargs):
    self.tm=tvtk.Matrix4x4()
    pass
    
  def update(self,pre=None,post=None):
      """
      This method is called when plot-o-matic receives new data
      """
      if pre is None:
        pre=self.e
      if post is None:
        post=self.e
      HasExpressionTraits.update(self)
      TMt=None
      if hasattr(self,'T'):
        if self.T !=None :
          p = self.frame.evalT(self.lag)
          if p!=None:
            TMt=matrix(p*self.T)
          else:
             return
      else:
        p=self.frame.evalT(self.lag)
        if p!=None:
          TMt=matrix(p)
        else:
           return
      if TMt is None:
         return
      TMt=pre*TMt*post
      self.tm.deep_copy(array(TMt).ravel())
      self.actor.poke_matrix(self.tm)
      if not(self.TM is Undefined or self.TM is None):
         if (self.TM!=TMt).any():
            #updating cache
            self.TM=TMt
      else:
         self.TM=TMt

  def add_to_scene(self,sc):
       """
       Add Primitive to a tvtk.scene
       """
       sc.add_actors(self.actor)
       self.scene=sc
       
  def remove_from_scene(self):
       """
       Remove primitive from atvtk.scene
       """      
       self.scene.remove_actors(self.actor)

  def setall(self,attr,value):
       """
       Used by PrimitiveCollection to set a property to a whole tree of VisualObjects
       """
       setattr(self,attr,value)
       

       
class Cone(Primitive):
  """
  Cone object
  
  Example usage:
  
  worldframe=WorldFrame()
  Cone(worldframe,radius='time',height=2,resolution=100)
  
  Obligatory parameters:
  * frame:  frame of reference
  
  Mayor geometric parameters:
  
  
  Important non-trivial parameters:
  * resolution is the number of polygons that is used to approximate the curved surface.
  
  """
  source = Instance(tvtk.ConeSource)
  height= TExpression(DelegatesTo('source'))
  radius= TExpression(DelegatesTo('source'))
  resolution= TExpression(DelegatesTo('source'))
  traits_view = View(
    Item(name = 'frame', label='Frame'),
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
  """
  py:function:: Box(frame[, T=numpy.matrix, x_length=1, y_length=1, z_length=1, **common])
  Box object
  
  Example usage:
  
  worldframe=WorldFrame()
  Box(worldframe,x_length='time',y_length=2)
  
 Look at the documentation of Primitive to see the other 'common' keyword options
  
  """
  source = Instance(tvtk.CubeSource)
  x_length=TExpression(DelegatesTo('source'))
  y_length=TExpression(DelegatesTo('source'))
  z_length=TExpression(DelegatesTo('source'))

  traits_view = View(
    Item(name = 'frame', label='Frame'),
    Item(name = 'T', label = 'Matrix4x4', style = 'custom'),
    Item(name = 'x_length', style = 'custom'),
    Item(name = 'y_length', style = 'custom'),
    Item(name = 'z_length', style = 'custom'),
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
  """
  py:function:: Axes(frame[, T=numpy.matrix])
  Plot a coordinate Axes
  
  Important non-trivial parameters:
  * scale_factor
  
  
  """
  source = Instance(tvtk.Axes)
  tube = Instance(tvtk.TubeFilter)
  
  scale_factor=DelegatesTo('tube')
  radius=TExpression(DelegatesTo('tube'))
  sides=PrototypedFrom('tube','number_of_sides')
  
  traits_view = View(
    Item(name = 'frame', label='Frame'),
    Item(name = 'T', label = 'Matrix4x4', style = 'custom'),
    Item(name = 'properties',editor=InstanceEditor(), label = 'Render properties'),
    title = 'Axes properties'
  )
  def __init__(self,*args,**kwargs):
    Primitive.__init__(self,**kwargs)
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
    Item(name = 'frame', label='Frame'),
    Item(name = 'T', label = 'Matrix4x4', style = 'custom'),
    Item(name = 'height'),
    Item(name = 'radius'),
    Item(name = 'resolution'),
    Item(name = 'properties', editor=InstanceEditor(), label = 'Render properties'),
    title = 'Cylinder properties'
  )
  def __init__(self,*args,**kwargs):
    Primitive.__init__(self,**kwargs)
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
    Item(name = 'frame', label='Frame'),
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
  """
  py:function:: Arrow(frame, axis=[1,0,0], **common)
  
  Plots an arrow in the 3D view
  
  Arguments:
   'axis' determines the direction in which the arrow is pointing
   'tip_resolution'  defines the resolution of the tip
   
  Look at the documentation of Primitive to see the other 'common' keyword options
  """
   source=Instance(tvtk.ArrowSource)
   tip_resolution = DelegatesTo("source")
   axis=TExpression(NumpyArray)
   traits_view = View(
    Item(name = 'frame', label='Frame'),
    Item(name = 'axis',style = 'custom'),
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
    Item(name = 'frame', label='Frame'),
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
    Item(name = 'frame', label='Frame'),
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
  """
  py:function:: ProjectedPoint(frame, point=[0,0,0], **common)
  Arguments:
   'point' is a list or tuple
   
  Look at the documentation of Primitive to see the other 'common' keyword options
  """
  point=TExpression(Either(List,Tuple))
  point1=None
  point2=None
  traits_view = View(
    Item(name = 'frame', label='Frame'),
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
    
  def _point_changed(self,new):
    if self.point is Undefined:
      pass
    self.source.point2=[self.point[0],self.point[1],0]
    self.source.point1=self.point



# The following code is mainly from 
# http://www.enthought.com/~rkern/cgi-bin/hgwebdir.cgi/colormap_explorer/file/a8aef0e90790/colormap_explorer/colormaps.py
class PolyLine(Primitive):
   """
   py:function::  PolyLine(frame, points=array, **common)
   
   Plots a line specified by a numpy array
   
   Look at the documentation of Primitive to see the other 'common' keyword options
   
   """
   
   source=Instance(tvtk.PolyData)
   points=Instance(ndarray)
   traits_view = View(
    Item(name = 'frame', label='Frame'),
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
    lines = zeros((npoints-1, 2), dtype=int)
    lines[:,0] = arange(0, npoints-1)
    lines[:,1] = arange(1, npoints)
    self.source.points=self.points
    self.source.lines=lines
    
#tvtk.DataSetMapper allows fancy things with colors, but not with opacity
#actor.set_lut(mm.scalar_lut_manager.lut)

class FadePolyLine(PolyLine):
   mapper=Instance(tvtk.DataSetMapper)
   actor=Instance(tvtk.Actor)
   color=ColorTrait((1,1,1))

   traits_view = View(
    Item(name = 'frame', label='Frame'),
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
    self._color_changed(self.color)
    
   def _color_changed(self,color):
      hsv=colorsys.rgb_to_hsv(color[0],color[1],color[2])
      self.lut.hue_range=(hsv[0],hsv[0])
      self.lut.saturation_range=(hsv[1],hsv[1])
      self.lut.value_range=(hsv[2],hsv[2])
      
   def _points_changed(self,old, new):
    PolyLine._points_changed(self,old, new)
    self.source.point_data.scalars=linspace(0,1,self.points.shape[0])

class Circle(PolyLine):
  """
  py:function:: Circle(frame, radius=1, resolution=100, **common)
 
  Plots a circle in the 3D view
   
  Look at the documentation of Primitive to see the other 'common' keyword options
  
  """
   radius=TExpression(Float)
   resolution=Int(100)
   points=Instance(ndarray)
   traits_view = View(
    Item(name = 'frame', label='Frame'),
    Item(name = 'T', label = 'Matrix4x4', style = 'custom'),
    Item(name = 'radius', style = 'custom'),
    Item(name = 'resolution'),
    Item(name = 'properties', editor=InstanceEditor(), label = 'Render properties'),
    title = 'Circle properties'
   )
   def __init__(self,*args,**kwargs):
     PolyLine.__init__(self,*args,**kwargs)
     
   def _radius_changed(self):
     self.calc()
     
   def _resolution_changed(self):
     self.calc()
     
   def calc(self):
     t=linspace(0,6.29,self.resolution)
     if not(self.radius is Undefined or self.radius is None) :
     	self.points = array([self.radius*sin(t),self.radius*cos(t),zeros(t.shape)]).T

class Trace(FadePolyLine):
  """
  py:function:: Trace(frame, point='', length=0, **common)
  This object will plot a trace, a line connecting historical points and fading.
  
  Example usage that will plot a helix trace
  
  worldframe=WorldFrame()
  Trace(worldframe,point='[sin(time),cos(time),time]')
  
  Special arguments:
  * length: length of trace in timesteps. Special value 0 to indicate the maximal length

  
  Look at the documentation of Primitive to see the other 'common' keyword options
  
  """
   point=TExpression(NumpyArray)
   length = Int(0)
   
   traits_view = View(
    Item(name = 'frame', label='Frame'),
    Item(name = 'length'),
    Item(name = 'point', style = 'custom'),
    Item(name = 'color'),
    Item(name = 'properties', editor=InstanceEditor(), label = 'Render properties'),
    title = 'Line properties'
   )
   def __init__(self,*args,**kwargs):
     FadePolyLine.__init__(self,*args,**kwargs)
     
   def _point_changed(self,new):
     expression=self.getExpression('point')
     if expression==None:
       return
     self.points=expression.get_array(first=-self.length)

class ProjectedPolyLine(Primitive):
	"""
	py:function:: ProjectedPolyLine(frame, watch=object, **common)
	Arguments:
	  'object' should be a PolyLine instance

	Plot a projection to the ground plane of a given PolyLine.
	
	Example usage:
	
	ProjectedPolyLine(world,Circle(world,T='TRx(0.5)'))
	
	Plots a tilted circle and vertical lines reaching to the ground on each circle discretisation point.
	
	Look at the documentation of Primitive to see the other 'common' keyword options
	"""
	watch=Instance(PolyLine)
	watchpoints=DelegatesTo('watch',prefix='points')
	watchTM=DelegatesTo('watch',prefix='TM')
	pd=Instance(tvtk.PolyData)
	polypoints=DelegatesTo('pd',prefix='points')
	lines=DelegatesTo('pd')
	
	color=DelegatesTo('properties')
	
	traits_view = View(
	 Item(name = 'watch', editor=InstanceEditor()),
	 Item(name = 'color'),
	 Item(name = 'properties', editor=InstanceEditor(), label = 'Render properties'),
	 title = 'ProjectedPolyLine properties'
	)
	def __init__(self,*args,**kwargs):
		Primitive.__init__(self,**kwargs)
		self.pd = tvtk.PolyData()
		self.mapper = tvtk.PolyDataMapper(input=self.pd)
		self.actor = tvtk.Actor(mapper=self.mapper)
		self.handle_arguments(*args,**kwargs)
		self.properties.representation='wireframe'
		
	def _watchpoints_changed(self,new):
		self.calc()

	def _watchTM_changed(self,new):
		self.calc()

	def calc(self):
		project=lambda x: (x[0],x[1],0)
		q=self.watchpoints
		if not(q is Undefined or q is None):
			if not(self.watch.TM is None):
				q=array((self.watch.TM*hstack((q,ones((q.shape[0],1)))).T).T)
				n=q[:,3]
				q=q[:,0:3]/(vstack((n,n,n)).T+0.0)
			self.polypoints=hstack((q,map(project,q))).reshape((q.shape[0]*2,3))
			#self.polys=array([[i*2,2*i+1,2*i+3,2*i+2] for i in range(q.shape[0]-1)])
			self.lines=array([[i*2,2*i+1] for i in range(q.shape[0]-1)])


     

class Text(Primitive):
  """
  py:function:: Text(frame, text='', **common)
  Plot text in the 3D window
  
  Look at @tofillin@ for overlaying 2D text

  
  Look at the documentation of Primitive to see the other 'common' keyword options
  """
  text=DelegatesTo('source')
  traits_view = View(
    Item(name = 'frame', label='Frame'),
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
    """
    py:function:: Image(frame, file='', **common)
    Plot an image in 3D space.
    Use T keyword to set the scale
    
    Example usage:
    Image(worldframe, T='sc(5)', file='woodpecker.bmp')
    
    Plots a bitmap scaled up 5 times.
    
    Look at the documentation of Primitive to see the other 'common' keyword options
    """
    source=Instance(ImageReader)
    file=DelegatesTo('source',prefix='base_file_name')
    traits_view = View(
    Item(name = 'frame', label='Frame'),
    Item(name = 'T', label = 'Matrix4x4', style = 'custom'),
    Item(name = 'file'),
    Item(name = 'source', editor=InstanceEditor()),
    Item(name = 'actor', editor=InstanceEditor()),
    title = 'Image properties'
    )

    def __init__(self,*args,**kwargs):
        Primitive.__init__(self,**kwargs)
        self.source=ImageReader(base_file_name=kwargs['file']) # im.ouput
        self.actor=tvtk.ImageActor(input=self.source.reader.output)
        self.handle_arguments(*args,**kwargs)
    
class ImageHeightMap(Primitive):
    """
    py:function:: ImageHeightMap(frame, file='', ?)

    This class is not yet implemented
    """
    source=Instance(ImageReader)
    file=DelegatesTo('source',prefix='base_file_name')
    warper= Instance( tvtk.WarpScalar)
    traits_view = View(
    Item(name = 'frame', label='Frame'),
    Item(name = 'T', label = 'Matrix4x4', style = 'custom'),
    Item(name = 'file'),
    Item(name = 'warper', editor=InstanceEditor()),
    Item(name = 'source', editor=InstanceEditor()),
    Item(name = 'actor', editor=InstanceEditor()),
    title = 'Image properties'
    )

    def __init__(self,*args,**kwargs):
        Primitive.__init__(self,**kwargs)
        self.source=ImageReader(base_file_name=kwargs['file']) # im.ouput
        self.geom=tvtk.ImageDataGeometryFilter(input=self.source.reader.get_output())
        print self.geom.get_output().number_of_cells
        print self.geom.get_output().number_of_lines
        print self.geom.get_output().number_of_pieces
        print self.geom.get_output().number_of_points
        print self.geom.get_output().number_of_polys
        self.warper = tvtk.WarpScalar(input=self.geom.get_output())
        self.mapper = tvtk.PolyDataMapper(input=self.warper.output)
        self.actor = tvtk.Actor(mapper=self.mapper)
        self.handle_arguments(*args,**kwargs)
    
#http://mayavi2.sourcearchive.com/documentation/3.3.0-2/actors_8py-source.html
#source = tvtk.ArrowSource(tip_resolution=resolution,
#                              shaft_resolution=resolution)

# LineSource, PlaneSource

class PrimitiveCollection(VisualObject):
  primitives=List(Instance(VisualObject))
  
  traits_view = View(
    Item(name = 'frame', label='Frame'),
    Item(name = 'T', label = 'Matrix4x4', style = 'custom'),
    Item(name = 'primitives', editor=ListEditor(),style='custom'),
    title = 'Collection properties'
   )
   
  def getPrimitives(self):
    return self.primitives
    
  def __init__(self,frame,T=eye,**kwargs):
    self.frame=frame
    VisualObject.__init__(self,frame,T,**kwargs)
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

  def update(self,pre=None,post=None):
      HasExpressionTraits.update(self)
      if pre is None:
        pre=self.e
      if post is None:
        post=self.e
      if self.T is None:
        map(lambda x: x.update(pre=pre,post=post),self.primitives)
      else:
        map(lambda x: x.update(pre=pre*self.T,post=post),self.primitives)

  def add_to_scene(self,sc):
       self.scene=sc
       map(lambda x: x.add_to_scene(sc),self.primitives)
       
  def remove_from_scene(self):
       map(lambda x: x.remove_from_scene(),self.primitives)
       self.primitives=[]
       
  def setall(self,attr,value):
       setattr(self,attr,value)
       map(lambda x: x.setall(attr,value),self.primitives)


class Strobe(PrimitiveCollection):
	"""
	py:function:: Strobe(object, [length=10, step=1])
	Arguments:
	  'object' is a Primitive or PrimitiveCollection instance
	  'length' is the number of strobed copies
	  'step' allows to interlace time history; only add a strobed copy every other 'step' frame
	  
	Example usage:
	
	Strobe(Sphere(world,T='tr(time,0,0)*sc(time)'), length=20)
	
	Will displays a trace of 20 spheres moving and growing in size
	
	"""
	length=Int(10)
	step=Int(1)
	template=Instance(VisualObject)
	variables=DelegatesTo('template')
	
	traits_view = View(
	    Item(name = 'length'),
	    Item(name = 'step'),
	    Item(name = 'template', editor=InstanceEditor(),style='custom'),
	    title = 'Strobe properties'
	   )
	   
	def __init__(self,template):
		print template
		self.template=template
	
	def _length_changed(self):
		self.setup()
		
	def _step_changed(self):
		self.setup()
		
	def setup(self):
		if hasattr(self,'scene'):
			self.remove_from_scene()
			for lag in range(0,self.length*self.step,self.step):
				print self.template
				cl=Lag(self.template,lag)
				self.add(cl)
				cl.add_to_scene(self.scene)
			
	def add_to_scene(self,sc):
		self.scene=sc
		self.setup()
	
def Lag(shape,lag):
	"""
	py:function:: Lag(visualObject, lag)
	
	Modifier that adds a delayed clone of a visualObject to the scene.
	
	Arguments:
	  'object' is a Primitive or PrimitiveCollection instance
	  'lag' is specified in timesteps
	
	Example usage:
	a=Box(world,T='tr(time,0,0)')
	
	self.add(a)
	self.add(Lag(a,10))
	
	Plots two moving boxes. One is lagging behind the other by 10 timesteps.

	Look at @tofillin@ for overlaying 2D text

	"""
	cl=shape.clone()
	cl.setall('lag',lag)
	return cl
		
	
