import vtk
from vtk import *
#%%

red = [100,50, 0]
colors = vtk.vtkUnsignedCharArray()
colors.SetNumberOfComponents(3)
colors.SetName("Colors")
#colors.SetNumberOfTuples(96)
for i in range(9):
    colors.InsertNextTuple(red)

ren = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)

# create a renderwindowinteractor
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)
appendPos = vtk.vtkAppendPolyData()

sphere = vtk.vtkSphereSource()
sphere.SetCenter(0,0,0)
sphere.SetRadius(1)

sp = sphere.GetOutput()
sphere.Update()
sp.GetCellData().SetScalars(colors)
#%%
appendPos.AddInputData(sp)

mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(appendPos.GetOutputPort())

actor = vtk.vtkActor()
actor.SetMapper(mapper)

mapper.SetColorModeToDirectScalars()

ren.AddActor(actor)

iren.Initialize()
renWin.Render()
iren.Start()
