import gmsh

filename = "T-Bracket3.STEP"

gmsh.initialize()

# Set mesh size (controls STL resolution)
gmsh.option.setString("Geometry.OCCTargetUnit", "M")
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.001)
gmsh.option.setNumber("Mesh.Binary", 1)

gmsh.model.add("model")

# Import STEP
gmsh.model.occ.importShapes(f"meshes/test/step/{filename}")
gmsh.model.occ.synchronize()

# Generate mesh and export
gmsh.model.mesh.generate(2)  # 2D surface mesh
gmsh.write(f"meshes/test/stl/{filename.replace('.STEP', '.stl')}")

gmsh.finalize()
