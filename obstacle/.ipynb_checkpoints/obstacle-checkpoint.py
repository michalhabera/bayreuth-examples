# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: dolfinx-env
#     language: python
#     name: python3
# ---

# # Obstacle problem
#
# The classical version of the obstacle problem is to find a minimizer of the Dirichlet (membrane) energy
# under constraint, that the deflection $u$ must be above a certain prescribed function, i.e. $u >= \varphi$.
#
# <div align="center"><img src="images/obstacle.png" alt="Obstacle drawing" width="400"/></div>
#
# (Image by Xavier Ros-Oton, taken from ["Obstacle problems and free boundaries:
# An overview"](https://www.ub.edu/pde/xros/articles/SeMA-article-2017.pdf).)
#
# Classical obstacle problem could be formulated as a constrained minimization problem: find $u \in K$ for which
# $$
# \begin{aligned}
# u &= \argmin\limits_{u \in K} \left\{ \int_\Omega |\nabla u|^2 \, \mathrm dx \right\},\\
# K &= \left\{ u \in H^1_0(\Omega), \, u \ge \varphi \right\},
# \end{aligned}
# $$
# where $\varphi$ is a given smooth obstacle.
#
#

# +
import gmsh

if not gmsh.is_initialized():
    gmsh.initialize()
gmsh.clear()
gmsh.open("model.step")

mesh_size = 0.5

gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
gmsh.model.geo.synchronize()
gmsh.model.addPhysicalGroup(2, [1], tag=1)
gmsh.model.addPhysicalGroup(1, [1, 2, 3, 4, 5, 6, 7, 8], tag=2)

gmsh.model.mesh.generate()
gmsh.finalize()

# +
import dolfinx
from mpi4py import MPI
mesh, cell_tags, facet_tags = dolfinx.io.gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, rank=0, gdim=2)
fixed_facets = facet_tags.indices[facet_tags.values == 2]

print(f"Number of fixed facets: {len(fixed_facets)}")

# +
import pyvista as pv
import os
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
pv.set_jupyter_backend("trame")
pv.start_xvfb()

pv.global_theme.window_size = [600, 500]
pv.global_theme.cmap = "viridis"
pv.global_theme.lighting = False


# +
cells, types, x = dolfinx.plot.vtk_mesh(mesh)
grid = pv.UnstructuredGrid(cells, types, x)

plotter = pv.Plotter()
plotter.add_mesh(grid, show_edges=True)
plotter.camera_position = "xy"
plotter.enable_parallel_projection()
plotter.show_axes()
plotter.show_grid()
plotter.show()
# -

V = dolfinx.fem.functionspace(mesh, ("P", 1))
print(f"Number of degrees of freedom: {V.dofmap.index_map.size_global}")

# +

import numpy as np

def obstacle(x, shift=5):
    return 10 * np.exp(-((x[0] - 7)**2 + (x[1] - 7)**2) / 20) - 10 + shift

phi = dolfinx.fem.Function(V)
phi.interpolate(obstacle)

grid.point_data["phi"] = phi.x.array
grid_warped = grid.warp_by_scalar("phi", factor=1.0)

plotter = pv.Plotter()
plotter.add_mesh(grid, opacity=0.8, color="white", show_edges=True)

plotter.add_mesh(grid_warped, show_edges=True)
plotter.camera_position = "xy"
plotter.camera.elevation = -70
plotter.enable_parallel_projection()
plotter.show_axes()
plotter.show()

# +
import ufl

u = dolfinx.fem.Function(V)
W = ufl.inner(ufl.grad(u), ufl.grad(u)) * ufl.dx
# -

boundary_dofs = dolfinx.fem.locate_dofs_topological(V, entity_dim=1, entities=fixed_facets)
bc = dolfinx.fem.dirichletbc(dolfinx.fem.Constant(mesh, 0.0), dofs=boundary_dofs, V=V)

# ### Interfacing with external solvers
#
# Let us consider a general optimization problem in $\mathbb R^n$: find $\bm x \in \mathbb R^n$,
# $$
# \begin{aligned}
# \bm x = \argmin f(\bm x),\\
# \bm g(\bm x) = 0,\\
# \bm h(\bm x) \le 0,\\
# \bm l \le \bm x \le \bm u,
# \end{aligned}
# $$
#
# with $f: \mathbb R^n \rightarrow \mathbb R$, $g: \mathbb R^n \rightarrow \mathbb R^k$, $h: \mathbb R^n \rightarrow \mathbb R^m$, $\bm l, \bm u \in \mathbb R^n$, i.e. there are $k$ equality, $m$ inequality and $n$ bounds constraints.
#
# In order to interface with existing optimization solvers (PETSc TAO, SciPy, NLOpt) we need to provide:
#
# 1. callback for evaluation of objective, $f(\bm x) \in \mathbb R$,
# 2. callback for evaluation of Jacobian, $\bm J = \partial f(\bm x) / \partial \bm x \in \mathbb R^{n}$,
# 3. callback for evaluation of Hessian, $\bm H = \partial \bm J(\bm x) / \partial \bm x \in \mathbb R^{n \times n}$,
# 4. callback for bounds constraints,
# 5. callback for evaluation of equality constraints and their Jacobian,
# 6. callback for evaluation of inequality constraints and their Jacobian.
#
# Specific interfaces and data structures differ across external libraries, but we explore the use of PETSc TAO for the obstacle problem. TAO supports evaluation of objective and gradient in a single routine (performance advantage).
#
# We're going to use the automatic differentiation capabilities of UFL to automatically compute the continuous versions of Jacobian and Hessian. This is often referred to as "optimize-then-discretize"
# approach and is crucial for achieving discretization-independent methods.
#
# At this point, we need to operate with `dolfinx.fem.Form` objects on a lower level. First, we explicitly invoke the compilation (translation of the UFL form into C language and compilation of that into a binary extension). This step allows us to control the compilation process, including the compiler and compilation flags.

# +

W_compiled = dolfinx.fem.form(W, jit_options={"cffi_verbose": True,
                                              "cffi_extra_compile_args": ["-O3"]})

J = ufl.derivative(W, u)
J_compiled = dolfinx.fem.form(J)

H = ufl.derivative(J, u)
H_compiled = dolfinx.fem.form(H)

# +


from petsc4py import PETSc


def evaluate_objective_and_gradient(tao, x, grad):
    with x.localForm() as x_loc:
        u.x.array[:] = x_loc.array[:]
    u.x.scatter_forward()

    value = MPI.COMM_WORLD.allreduce(
        dolfinx.fem.assemble_scalar(W_compiled)
    )

    b = dolfinx.fem.petsc.assemble_vector(J_compiled)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.petsc.set_bc(b, [bc], x, -1.0)

    with b.localForm() as b_loc, grad.localForm() as grad_loc:
        grad_loc.array[:] = b_loc.array[:]

    return value


def evaluate_hessian(tao, x, hess, hess_pc):
    with x.localForm() as x_loc:
        u.x.array[:] = x_loc.array[:]
    u.x.scatter_forward()
    hess.zeroEntries()
    dolfinx.fem.petsc.assemble_matrix(hess, H_compiled, [bc])
    hess.assemble()



# -

# We can create a TAO solver object, set its type and attach the callbacks.
#
# The bounds-constrained solvers available in TAO are Bounded Newton-Krylov methods BNLS/BNTR/BNTL.
#
# BNLS is a Bounded Newton Line Search method that we'll use in this example. It is a second-order line search method with additional safe guarding and stabilization for indefinite Hessians. For step $k$, it seeks the descent direction as a solution to
# $$
# (\bm H_k + \rho_k \bm I) \bm p_k = - \bm J_k.
# $$
# Line search is by default the More-Thuente line search. See [PETSc TAO manual](https://petsc.org/release/manual/manual.pdf#page=190) for more details.
#
# Bounds constraints are implemented using active-set approach with the active-set estimation based on Bertsekas.

# +
import dolfinx.fem.petsc

grad0 = u.x.petsc_vec.duplicate()
grad0.zeroEntries()

hess0 = dolfinx.fem.petsc.assemble_matrix(H_compiled, [])

opts_tao = PETSc.Options("tao_")
opts_tao["type"] = "bnls"
opts_tao["max_it"] = 100
opts_tao["monitor"] = ""

tao = PETSc.TAO().create()
solution = dolfinx.fem.Function(V)
tao.setSolution(solution.x.petsc_vec)
tao.setObjectiveGradient(evaluate_objective_and_gradient, grad0)
tao.setHessian(evaluate_hessian, hess0)

upper_bound = dolfinx.fem.Function(V)
upper_bound.x.array[:] = PETSc.INFINITY
tao.setVariableBounds(phi.x.petsc_vec, upper_bound.x.petsc_vec)
tao.setFromOptions()
tao.solve()

# +
grid = pv.UnstructuredGrid(cells, types, x)
grid.point_data["u"] = solution.x.array
grid.point_data["phi"] = phi.x.array

grid_u = grid.warp_by_scalar("u", factor=1.0)
grid_phi = grid.warp_by_scalar("phi", factor=1.0)

plotter = pv.Plotter()
plotter.add_mesh(grid_phi, color="white", show_edges=True)
plotter.add_mesh(grid_u, show_edges=False, opacity=1.0)
plotter.camera_position = "xy"
# plotter.camera.elevation = -70
plotter.enable_parallel_projection()
plotter.show_axes()
plotter.show()
# -

# ### Post-processing
#
# Dolfinx supports interpolation of arbitrary UFL expressions into a function space. This allows evaluation and visualization of various post-processing quantities. In the obstacle problem we'd like to inspect the contact area, i.e. the region where the bounds contraint is active,
# $$
# A = \int_\Omega I(u = \varphi) \, \mathrm dx,
# $$
# where $I(u = \varphi)$ denotes the indicator function of the set where $u = \varphi$.
#
#

# +
contact = ufl.conditional(solution < (phi + 1e-8), 1, 0)

expr = dolfinx.fem.Expression(contact, V.element.interpolation_points())
contact_fn = dolfinx.fem.Function(V)
contact_fn.interpolate(expr)

A = dolfinx.fem.assemble_scalar(dolfinx.fem.form(contact * ufl.dx))
print(f"Contact area: {A:.6g} mm^2")

grid = pv.UnstructuredGrid(cells, types, x)
grid.point_data["contact"] = contact_fn.x.array

plotter = pv.Plotter()
plotter.add_mesh(grid, show_edges=False)
plotter.camera_position = "xy"
plotter.enable_parallel_projection()
plotter.show_axes()
plotter.show()

