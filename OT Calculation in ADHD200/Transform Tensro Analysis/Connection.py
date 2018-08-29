#%%
import vtk
import numpy as np
import colorsys

def CalTrans(start, end):
    startPoint = start
    endPoint = end

    # Compute a basis
    normalizedX = [0] * 3
    normalizedY = [0] * 3
    normalizedZ = [0] * 3

    # The X axis is a vector from start to end
    vtk.vtkMath.Subtract(endPoint, startPoint, normalizedX)
    length = vtk.vtkMath.Norm(normalizedX)
    vtk.vtkMath.Normalize(normalizedX)
    arbitrary = [0, 0, 1]
    '''
    # The Z axis is an arbitrary vector cross X
    arbitrary = [0] * 3
    for i in range(0, 3):
        rng.Next()
        arbitrary[i] = rng.GetRangeValue(-10, 10)
    '''
    vtk.vtkMath.Cross(normalizedX, arbitrary, normalizedZ)
    vtk.vtkMath.Normalize(normalizedZ)

    # The Y axis is Z cross X
    vtk.vtkMath.Cross(normalizedZ, normalizedX, normalizedY)
    matrix = vtk.vtkMatrix4x4()

    # Create the direction cosine matrix
    matrix.Identity()
    for i in range(0, 3):
        matrix.SetElement(i, 0, normalizedX[i])
        matrix.SetElement(i, 1, normalizedY[i])
        matrix.SetElement(i, 2, normalizedZ[i])

    return matrix, length

def SetArrow(p, matrix, length, d):
    arrowSource = vtk.vtkArrowSource()
    arrowSource.SetShaftRadius(0.5)
    arrowSource.SetTipRadius(2)
    arrow = arrowSource.GetOutput()
    arrowSource.Update()
    #n = arrow.GetNumberOfCells()
    #print(n)
    r,g,b = colorsys.hsv_to_rgb(0, 1, d*4)
    red = [r, g, b]
    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(3)
    colors.SetName("Colors")
    # colors.SetNumberOfTuples(96)
    # print("ok")

    for i in range(15):
        colors.InsertNextTuple(red)

    arrowSource.Update()
    arrow.GetCellData().SetScalars(colors)

    transform = vtk.vtkTransform()
    transform.Translate(p)
    transform.Concatenate(matrix)
    transform.Scale(length, 1, 1)

    transformPD = vtk.vtkTransformPolyDataFilter()
    transformPD.SetTransform(transform)
    #transformPD.SetInputConnection(arrowSource.GetOutputPort())
    transformPD.SetInputData(arrow)

    #arrow = transformPD.GetOutput()
    #n = arrow.GetNumberOfCells()
    #print(n)



    appendArrow.AddInputConnection(transformPD.GetOutputPort())
    #mapper.AddInputConnection(arrowSource.GetOutputPort())
    #actor.SetUserMatrix(transform.GetMatrix())
    #actor.SetMapper(mapper)


def SetPoint(pos,r,g,blue):
    sphereStartSource = vtk.vtkSphereSource()
    sphereStartSource.SetCenter(pos)
    sphereStartSource.SetRadius(1)

    point = sphereStartSource.GetOutput()
    sphereStartSource.Update()
    color = [r, g, blue]
    colorsP = vtk.vtkUnsignedCharArray()
    colorsP.SetNumberOfComponents(3)
    colorsP.SetName("ColorsofPoints")
    # colors.SetNumberOfTuples(96)
    # print("ok")
    for k in range(96):
        colorsP.InsertNextTuple(color)

    sphereStartSource.Update()
    point.GetCellData().SetScalars(colorsP)
    appendPos.AddInputConnection(sphereStartSource.GetOutputPort())
    #sphereStartMapper = vtk.vtkPolyDataMapper()
    #sphereStartMapper.SetInputConnection(sphereStartSource.GetOutputPort())
    #sphereStart = vtk.vtkActor()
    #sphereStart.SetMapper(sphereStartMapper)
    #sphereStart.GetProperty().SetColor(colors.GetColor3d("Yellow"))
    #renderer.AddActor(sphereStart)


def drawArrow(pos1, pos2, d):
    m, l = CalTrans(pos1, pos2)
    SetArrow(pos1, m, l, d)


def pickpart(X, dlevel, ulevel):
    ran = ulevel - dlevel
    para = 255/ran
    for i in range(b):
        for j in range(i + 1, b):
            d = X[i, j]
            if ulevel > abs(d) > dlevel:
                pos1 = MNIspos[i]
                pos2 = MNIspos[j]
                # print("ok")
                if d > 0:
                    d = (d - dlevel) * para
                    start = pos1
                    end = pos2
                    if d < 60:
                        d = 60
                else:
                    d = -d
                    d = (d - dlevel) * para
                    start = pos2
                    end = pos1
                    if d < 60:
                        d = 60
                drawArrow(start, end, d)

#%%


posdirc = "C:\\Users\Yin\Desktop\ADHD200\Pos\MNIspos.npy"
MNIspos = np.load(posdirc)
colordirc = "C:\\Users\Yin\Desktop\ADHD200\colorMap.npy"
colormap = np.load(colordirc)
#Mdic = "C:\\Users\Yin\Desktop\ADHD200\SumMapOfGroup\ADHDUp15\HeatADHDUp15.npy"
Mdic = "C:\\Users\Yin\Desktop\ADHD200\SumMapOfGroup\\ADHDDown15\HeatADHDDown15.npy"
#Mdic = "C:\\Users\Yin\Desktop\ADHD200\SumMapOfGroup\\NormalUp15\HeatNormalUp15mid300.npy"
#Mdic = "C:\\Users\Yin\Desktop\ADHD200\SumMapOfGroup\\NormalDown15\HeatNormalDown15.npy"
X = np.load(Mdic)
b = X.shape[0]
#%%
colors = vtk.vtkNamedColors()
appendArrow = vtk.vtkAppendPolyData()
appendPos = vtk.vtkAppendPolyData()

for i in range(MNIspos.shape[0]):
    pos = MNIspos[i]
    r,g,blue = colormap[i]
    SetPoint(pos,r,g,blue)

pickpart(X, 70, 75)

ArrowMapper = vtk.vtkPolyDataMapper()
ArrowMapper.SetColorModeToDirectScalars()
ArrowMapper.SetInputConnection(appendArrow.GetOutputPort())
ArrowActor = vtk.vtkActor()
ArrowActor.SetMapper(ArrowMapper)
ArrowActor.GetProperty().SetOpacity(0.5)

PosMapper = vtk.vtkPolyDataMapper()
PosMapper.SetColorModeToDirectScalars()
PosMapper.SetInputConnection(appendPos.GetOutputPort())
PosActor = vtk.vtkActor()
PosActor.SetMapper(PosMapper)
#PosActor.GetProperty().SetColor(colors.GetColor3d("White"))

renderer = vtk.vtkRenderer()
renderer.AddActor(ArrowActor)
renderer.AddActor(PosActor)


renderWindow = vtk.vtkRenderWindow()
renderWindow.SetWindowName("Oriented Arrow")
renderWindow.AddRenderer(renderer)
renderWindowInteractor = vtk.vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)


# Render and interact
renderWindow.Render()
renderWindowInteractor.Start()