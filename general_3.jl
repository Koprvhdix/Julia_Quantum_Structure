using Convex, SCS
using LinearAlgebra

# “Find the value of rho that has the maximum distance from I among all possible values of rho.
function find_rho()
  rho = Semidefinite(16)
  constraints = [tr(rho) == 1, rho in :SDP]
  objective = norm(vec(rho) - vec(I(16)))

  problem = maximize(objective, constraints)

  solve!(problem, SCS.Optimizer)

  println(evaluate(rho))
  println(problem.optval)
end

find_rho()


