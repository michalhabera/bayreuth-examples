# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Neo-Hooke compressible solid material
#
# Neo-Hooke compressible material is a hyperelastic material with strain energy density (for $d=3$) given as
# $$
# W(I_1, J) = C_1 (I_1 - 3 - 2 \log J) + D_1 (J - 1)^2
# $$
# where
# $$
# \begin{aligned}
# J &= \det \bm{F},\\
# I_1 &= \text{Tr} (\bm F^T \bm F),\\
# \bm F(\bm u) &= \bm I + \nabla \bm u.
# \end{aligned}
# $$
#
# Material parameters are chosen as $C_1 = \mu / 2, \, D_1 = \lambda / 2$ for consistency with linear Hooke elasticity, $\mu$ is the second Lamé parameter (shear modulus) and $\lambda$ is first Lamé parameter.
#
# We can formulate the following (unconstrained) optimization problem: find $\bm u \in V(\Omega) := [H^1_0(\Omega)]^3$ for which
# $$
# \bm u = \argmin\limits_{\bm u \in V(\Omega)} \left\{ \int_\Omega W(\bm u) \, \mathrm dx - \int_{\Gamma_N} \bm t \cdot \bm u \, \mathrm ds \right\}.
# $$
#

# +
import gmsh

if not gmsh.is_initialized():
    gmsh.initialize()
gmsh.clear()
gmsh.open("model.step")

mesh_size = 3

gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
gmsh.model.geo.synchronize()
# gmsh.model.mesh.setSize([(0, 5), (0, 6)], 0.5)
gmsh.model.addPhysicalGroup(3, [1], 1)
gmsh.model.addPhysicalGroup(2, [1], 2)
gmsh.model.addPhysicalGroup(2, [4], 3)
gmsh.model.mesh.generate()
gmsh.finalize()


# +
import dolfinx
from mpi4py import MPI
mesh, cell_tags, facet_tags = dolfinx.io.gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, rank=0, gdim=3)
fixed_facets = facet_tags.indices[facet_tags.values == 2]
loaded_facets = facet_tags.indices[facet_tags.values == 3]

print(f"Number of fixed facets: {len(fixed_facets)}")
print(f"Number of loaded facets: {len(loaded_facets)}")

# +
import pyvista as pv
import os
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
pv.set_jupyter_backend("trame")
pv.start_xvfb()

pv.global_theme.window_size = [600, 500]
pv.global_theme.cmap = "viridis"

# +
cells, types, x = dolfinx.plot.vtk_mesh(mesh)
grid = pv.UnstructuredGrid(cells, types, x)

plotter = pv.Plotter()
# plotter.add_mesh(grid, show_edges=True, ambient=0.2, specular=0.2)
plotter.add_mesh_clip_plane(grid)
plotter.camera_position = "xy"
plotter.camera.azimuth = 30
plotter.camera.elevation = 15
plotter.enable_parallel_projection()
plotter.show_axes()
plotter.show()
# -

# The choice of discrete space for the displacement we make is $[P_2]^3 \subset [H^1(\Omega)]^3$, i.e. a vector-valued degree 2 Lagrange space, see [defelement.org](https://defelement.org/elements/examples/tetrahedron-vector-lagrange-2.html),
#
# <div align="center"><img src="images/vp2.png" alt="lagrange-p1" width="600"/></div>

V = dolfinx.fem.functionspace(mesh, ("P", 2, (3, )))
print(f"Number of degrees of freedom: {V.dofmap.index_map.size_global} x {V.dofmap.index_map_bs}")

# +
import ufl
import numpy as np

nu = 0.49
E = 0.01  # GPa

mu = E / (2 * (1 + nu))
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

C1 = mu / 2
D1 = lmbda / 2


def W(I1, J):
    return C1*(I1 - 3 - 2*ufl.ln(J)) + D1*(J - 1)**2


u0 = dolfinx.fem.Function(V)

F = ufl.Identity(3) + ufl.grad(u0)
I1 = ufl.tr(F.T * F)
J = ufl.det(F)

x = ufl.SpatialCoordinate(mesh)
t = dolfinx.fem.Constant(mesh, np.array([0.0, 0.0, 0.0]) * 1e-6)

ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)
W_total = W(I1, J) * ufl.dx(metadata={"quadrature_degree": 4}) - ufl.inner(t, u0) * ds(3)

# +
boundary_dofs = dolfinx.fem.locate_dofs_topological(V, entity_dim=2, entities=fixed_facets)
bc = dolfinx.fem.dirichletbc(dolfinx.fem.Constant(mesh, [0.0, 0.0, 0.0]), dofs=boundary_dofs, V=V)

loaded_dofs = dolfinx.fem.locate_dofs_topological(V, entity_dim=2, entities=loaded_facets)
bc1 = dolfinx.fem.dirichletbc(dolfinx.fem.Constant(mesh, [0.0, 0.0, 0.0]), dofs=loaded_dofs, V=V)

# +
import dolfinx.nls.petsc
from petsc4py import PETSc

R = ufl.derivative(W_total, u0)
problem = dolfinx.fem.petsc.NonlinearProblem(R, u0, bcs=[bc, bc1])
solver = dolfinx.nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)

ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "preonly"
opts[f"{option_prefix}pc_type"] = "cholesky"
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
ksp.setFromOptions()

dolfinx.log.set_log_level(dolfinx.log.LogLevel.WARNING)
for i in range(20):
    n, converged = solver.solve(u0)
    print(f"Number of Newton iterations: {n}")
    bc1.g.value[2] += 1

# +
U = dolfinx.fem.functionspace(mesh, ("P", 1, (3, )))
u_viz = dolfinx.fem.Function(U)
u_viz.interpolate(u0)

grid.point_data["u"] = u_viz.x.array.reshape(-1, 3)
grid.set_active_scalars("u")
grid_warped = grid.warp_by_vector("u")

plotter = pv.Plotter()
plotter.add_mesh(grid_warped, show_edges=True, ambient=0.2, specular=0.2, n_colors=10)
plotter.camera_position = "xy"
plotter.enable_parallel_projection()
plotter.show_axes()
plotter.show()
