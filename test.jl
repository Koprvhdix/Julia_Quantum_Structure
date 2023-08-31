using Convex, SCS

using LinearAlgebra

rho = [0.   0.   0.   0.   0.   0.   0.   0.  ;
      0.   1/3  1/3  0.   1/3  0.   0.   0.  ;
      0.   1/3  1/3  0.   1/3  0.   0.   0.  ;
      0.   0.   0.   0.   0.   0.   0.   0.  ;
      0.   1/3  1/3  0.   1/3  0.   0.   0.  ;
      0.   0.   0.   0.   0.   0.   0.   0.  ;
      0.   0.   0.   0.   0.   0.   0.   0.  ;
      0.   0.   0.   0.   0.   0.   0.   0.  ]

exchange = [1. 0. 0. 0. 0. 0. 0. 0. ;
      0. 0. 1. 0. 0. 0. 0. 0. ;
      0. 1. 0. 0. 0. 0. 0. 0. ;
      0. 0. 0. 1. 0. 0. 0. 0. ;
      0. 0. 0. 0. 1. 0. 0. 0. ;
      0. 0. 0. 0. 0. 0. 1. 0. ;
      0. 0. 0. 0. 0. 1. 0. 0. ;
      0. 0. 0. 0. 0. 0. 0. 1. ]

rho = 0.4 * rho + (0.4 / 8) * I(8)

X = rand(4, 4); A = X * X'; rho_a = A / tr(A)
X = rand(4, 4); B = X * X'; rho_b = B / tr(B)
X = rand(4, 4); C = X * X'; rho_c = C / tr(C)

X_list = [ [rho_a, rho_b, rho_c] ]

for i in 1:4
  X = rand(4, 4); A = X * X'; rho_a = A / tr(A)
  X = rand(4, 4); B = X * X'; rho_b = B / tr(B)
  X = rand(4, 4); C = X * X'; rho_c = C / tr(C)
  push!(X_list, [rho_a, rho_b, rho_c])
end

rho_1 = Semidefinite(2)
rho_2 = Semidefinite(2)
rho_3 = Semidefinite(2)

rho_next = kron(rho_1, X_list[1][1])
rho_next += kron(rho_2, X_list[1][2])
rho_next += kron(rho_3, X_list[1][3])

rho_list = [ [rho_1, rho_2, rho_3] ]

for index in 2:length(X_list)
  rho_1 = Semidefinite(2)
  rho_2 = Semidefinite(2)
  rho_3 = Semidefinite(2)

  rho_next += kron(rho_1, X_list[index][1])
  rho_next += kron(rho_2, X_list[index][2])
  rho_next += kron(rho_3, X_list[index][3])
  push!(rho_list, [rho_1, rho_2, rho_3])
end

objective = -tr(rho_next)
constraints = [isposdef(rho - rho_next)]
problem = minimize(objective, constraints)
solve!(problem, SCS.Optimizer)
evaluate(rho_list)
