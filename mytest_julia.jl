using Convex, SCS
using LinearAlgebra

w = [1, 2, 3, 4]
rho = Semidefinite(2)

objective = dot(w, vec(rho))
constraints = [tr(rho) == 1, rho in :SDP]

problem = maximize(objective, constraints)

solve!(problem, SCS.Optimizer)

println(evaluate(rho))
println(eigvals(evaluate(rho)))

println(problem.optval)
