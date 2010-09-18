from variables import Variables, Expression

from plugins.viewers.tvtkHelper.toolbox import *

from vtk.util import colors

from numpy.random import rand

class Arnold1(PrimitiveCollection):
  def __init__(self,frame,T=None,**kwargs):
    PrimitiveCollection.__init__(self,frame,T)
    self.primitives=[
       Box(self.frame,T='tr(-0.8,0,0)',x_length=3.2,y_length=0.20,z_length=0.01,**kwargs),
       Box(self.frame,x_length=0.40,y_length=4,z_length=0.04,**kwargs),
       Box(self.frame,T='tr(-2.4,0,0)',x_length=0.40,y_length=1,z_length=0.04,**kwargs),
       Box(self.frame,T='tr(-2.4,0,-0.20)',x_length=0.40,y_length=0.04,z_length=0.40,**kwargs),
    ]

class Logo(PrimitiveCollection):
  def __init__(self,frame,T=None,**kwargs):
    PrimitiveCollection.__init__(self,frame,T)
    self.primitives=[
       Text(self.frame,text='Joby')
    ]
    
class Logo(PrimitiveCollection):
  def __init__(self,frame,T=None,**kwargs):
    PrimitiveCollection.__init__(self,frame,T)
    self.primitives=[
       Text(self.frame,text='Joby')
    ]
    
class TVTKconfig(PrimitiveCollection):
  def __init__(self,w):

    PrimitiveCollection.__init__(self,w)
    
    self.add(Box(w,x_length='time',y_length=2))
    f=Frame(w,T='tr(3*sin(time),0,0)*tr(0,0,1)')
    self.add(Box(f))
    self.add(Box(f,T='tr(0,0,1)',color=colors.red))
    g=Frame(f,T='TRz(time)')
    self.add(Box(g,T='tr(2,0,0)',color=colors.blue))
    
    self.add(Arnold1(w,T='tr(0,0,time)',color=colors.red))
    #self.add(Lag(Arnold1(w,T='tr(0,0,time)',color=colors.red),10))

    self.add(Arrow(g,color=colors.red))
    self.add(Circle(g,T='tr(0,0,10)',radius=4))
    self.add(Trace(w,point='[time,sin(time),3]'))
    
    #self.add(FadePolyLine(w,points='[sin(time),cos(time),0]'))
    #c=Circle(w,T='tr(0,0,10)',radius=4)
    #self.add(c)
    #self.add(ProjectedPolyLine(w,watch=c,representation='wireframe'))
    
    #self.add(Line(w,color=colors.red,point1='(1,2,3)'))
    #self.add(ProjectionLine(w,color=colors.red,point=(1,2,3)))
   # self.add(Arnold1(w,color=colors.red))
    #self.add(Trace(w,point='[time,sin(time),3]'))
    #self.add(Circle(w,radius='time'))
    
    #self.add(Box(w,T='tr(10*cos(time),2,0)',x_length=2))
    #self.add(Box(w,T='tr(10*sin(time),2,0)',x_length=2))
    #b=Box(w,T='tr(10*cos(time),0,0)',x_length=2)
    #self.add(b)
    #self.add(Lag(b,5))
    


class TVTKconfig2(PrimitiveCollection):
  def __init__(self,variables):
    self.variables=variables
    w=WorldFrame(variables)

    self.add(Text(w,text='Plot-o-matic goes TVTK!'))
    ned=Frame(w,'TRx(pi)',name="North East Down");
    diskframe=Frame(ned,'tr(AP_DISK_r_n2d_n_x,AP_DISK_r_n2d_n_y,AP_DISK_r_n2d_n_z)*quat(AP_DISK_q_n2d_q0,AP_DISK_q_n2d_q1,AP_DISK_q_n2d_q2,AP_DISK_q_n2d_q3)',name="diskframe")
    orientation=Frame(ned,'tr(20,20,0)*quat(AP_EST2USER_0_q_n2b_q0,AP_EST2USER_0_q_n2b_q1,AP_EST2USER_0_q_n2b_q2,AP_EST2USER_0_q_n2b_q3)')
    airframe=Frame(ned,'tr(AP_EST2USER_0_r_n2b_n_x,AP_EST2USER_0_r_n2b_n_y,AP_EST2USER_0_r_n2b_n_z)*quat(AP_EST2USER_0_q_n2b_q0,AP_EST2USER_0_q_n2b_q1,AP_EST2USER_0_q_n2b_q2,AP_EST2USER_0_q_n2b_q3)')

    ax=Frame(ned,'sc(50)')
    self.add(Arrow(ax,color=colors.red))
    self.add(Text(ax,T='tr(1,0,0)*sc(0.1)',text='N / X'))
    self.add(Arrow(ax,T='TRz(pi/2)',color=colors.green))
    self.add(Text(ax,T='tr(0,1,0)*sc(0.1)',text='E / Y'))
    self.add(Arrow(ax,T='TRy(-pi/2)',color=colors.blue))
    self.add(Text(ax,T='tr(0,0,1)*sc(0.1)',text='D / Z'))

    self.add(Plane(ned,T='sc(400)',representation='wireframe',color=colors.grey,x_resolution=40,y_resolution=40))

    
    self.add(Arnold1(orientation,T='sc(5)',color=colors.blue))
    self.add(Text(orientation,text='Reference only'))

    self.add(Arnold1(airframe,color=colors.red))

    self.add(Arnold1(w,color=colors.red))
    self.add(Logo(ned))
    
    self.add(Circle(diskframe,radius=variables.new_expression('AP_DISK_radius')))
