from enthought.tvtk.pyface.scene_model import SceneModel
from enthought.tvtk.pyface.scene_editor import SceneEditor
from enthought.tvtk.pyface import actors
from enthought.tvtk.api import tvtk

from enthought.traits.api import HasTraits, Str, Regex, List, Instance, DelegatesTo, Any, on_trait_change, Float, Range
from enthought.traits.ui.api import TreeEditor, TreeNode, View, Item, VSplit, \
  HGroup, Handler, Group, Include, ValueEditor, HSplit, ListEditor, InstanceEditor

from enthought.pyface.api import GUI

import numpy
from viewers import Viewer
from variables import Variables, Expression
from plugins.viewers.tools3D.Frame import *
from plugins.viewers.tvtkHelper.Primitives import *

class TVTKViewer(Viewer):
  name = Str('TVTK Viewer')
  primitives = List(VisualObject)
  scene = Instance(SceneModel, ())
  
  view = View(
      Item(
        name = 'scene',
        height = 400,
        show_label = False,
        editor = SceneEditor()
      )
  )

  traits_view = View(
    Item(name = 'name'),
    Item(name = 'refresh_rate'),
    Item(name = 'primitives', editor=ListEditor(),style='custom'),
    title = 'Viewer'
  )

  def __init__(self,vs,config):
    Viewer.__init__(self)
    world=WorldFrame(vs)
    self.config=config(world)

  def start(self):
    self.primitives=self.config.getPrimitives()
    map(lambda x: x.add_to_scene(self.scene),self.primitives)
    
  def stop(self):
    pass

  def show(self):
    pass

  def hide(self):
    pass

  def update(self):
    for prim in self.primitives:
      prim.update()
    GUI.invoke_later(self.scene.render)

