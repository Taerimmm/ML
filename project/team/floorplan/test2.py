from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot

# Create a new plot
figure = pyplot.figure()
axes = mplot3d.Axes3D(figure)

# Load the STL files and add the vectors to the plot
your_mesh = mesh.Mesh.from_file('C:/3d/FloorplanToBlender3d/target/floorplan.stl')
# axes.add_collection3d(mplot3d?.art3d.Poly3DCollection(your_mesh.vectors))

# Auto scale to the mesh size.
your_mesh.points[:,2]=0
your_mesh.points[:,5]=0
your_mesh.points[:,8]=0
print(your_mesh.points)
print(your_mesh.points.shape)
axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors))
scale = your_mesh.points.flatten()
axes.auto_scale_xyz(scale/2, scale/2, scale/2)
# Show the plot to the screen
pyplot.show()