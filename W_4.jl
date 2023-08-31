using Convex, SCS

using LinearAlgebra

rho = [0.0  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  ;
 0.0  0.25 0.25 0.   0.25 0.   0.   0.   0.25 0.   0.   0.   0.   0.   0.   0.  ;
 0.0  0.25 0.25 0.   0.25 0.   0.   0.   0.25 0.   0.   0.   0.   0.   0.   0.  ;
 0.0  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  ;
 0.0  0.25 0.25 0.   0.25 0.   0.   0.   0.25 0.   0.   0.   0.   0.   0.   0.  ;
 0.0  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  ;
 0.0  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  ;
 0.0  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  ;
 0.0  0.25 0.25 0.   0.25 0.   0.   0.   0.25 0.   0.   0.   0.   0.   0.   0.  ;
 0.0  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  ;
 0.0  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  ;
 0.0  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  ;
 0.0  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  ;
 0.0  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  ;
 0.0  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  ;
 0.0  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  ]

rho = 0.3 * rho + (0.7 / 16) * I(16)

exchange_1 = [1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ;
 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ;
 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ;
 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ;
 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ;
 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ;
 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ;
 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ;
 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ;
 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 ;
 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 ;
 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 ;
 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 ;
 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 ;
 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 ;
 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 ]

exchange_2 = [1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ;
 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ;
 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ;
 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ;
 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ;
 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ;
 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ;
 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ;
 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ;
 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 ;
 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 ;
 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 ;
 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 ;
 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 ;
 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 ;
 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 ]

function f_1(X_list)
  rho_1 = Semidefinite(4)
  rho_2 = Semidefinite(4)
  rho_3 = Semidefinite(4)

  rho_next = kron(rho_1, X_list[1][1])
  rho_next += (exchange_1 * kron(rho_2, X_list[1][2]) * exchange_1)
  rho_next += (exchange_2 * kron(rho_3, X_list[1][3]) * exchange_2)

  rho_list = [ [rho_1, rho_2, rho_3 ] ]

  for index in 2:length(X_list)
    rho_1 = Semidefinite(4)
    rho_2 = Semidefinite(4)
    rho_3 = Semidefinite(4)

    rho_next += kron(rho_1, X_list[index][1])
    rho_next += (exchange_1 * kron(rho_2, X_list[index][2]) * exchange_1)
    rho_next += (exchange_2 * kron(rho_3, X_list[index][3]) * exchange_2)

    push!(rho_list, [rho_1, rho_2, rho_3 ] )
  end

  objective = -tr(rho_next)
  constraints = [isposdef(rho - rho_next)]

  for item in rho_list
    push!(constraints, isposdef(item[1]))
    push!(constraints, isposdef(item[2]))
    push!(constraints, isposdef(item[3]))
  end

  problem = minimize(objective, constraints)
  solve!(problem, SCS.Optimizer)
  println(problem.optval)

  next_X = [ [item[1].value, item[2].value, item[3].value] for item in evaluate(rho_list) ]
  # println(next_X)
  return next_X
end

function f_2(X_list)
  rho_1 = Semidefinite(4)
  rho_2 = Semidefinite(4)
  rho_3 = Semidefinite(4)

  rho_next = kron(X_list[1][1], rho_1)
  rho_next += (exchange_1 * kron(X_list[1][2], rho_2) * exchange_1)
  rho_next += (exchange_2 * kron(X_list[1][3], rho_3) * exchange_2)

  rho_list = [ [rho_1, rho_2, rho_3 ] ]

  for index in 2:length(X_list)
    rho_1 = Semidefinite(4)
    rho_2 = Semidefinite(4)
    rho_3 = Semidefinite(4)

    rho_next += kron(X_list[index][1], rho_1)
    rho_next += (exchange_1 * kron(X_list[index][2], rho_2) * exchange_1)
    rho_next += (exchange_2 * kron(X_list[index][3], rho_3) * exchange_2)

    push!(rho_list, [rho_1, rho_2, rho_3 ] )
  end

  objective = -tr(rho_next)
  constraints = [isposdef(rho - rho_next)]

  for item in rho_list
    push!(constraints, isposdef(item[1]))
    push!(constraints, isposdef(item[2]))
    push!(constraints, isposdef(item[3]))
  end

  problem = minimize(objective, constraints)
  solve!(problem, SCS.Optimizer)
  println(problem.optval)

  next_X = [ [item[1].value, item[2].value, item[3].value] for item in evaluate(rho_list) ]
  # println(next_X)
  return next_X
end

X = rand(4, 4); A = X * X'; rho_a = A / tr(A)
X = rand(4, 4); B = X * X'; rho_b = B / tr(B)
X = rand(4, 4); C = X * X'; rho_c = C / tr(C)

first_list = [ [rho_a, rho_b, rho_c] ]

for i in 1:999
  X = rand(4, 4); A = X * X'; rho_a = A / tr(A)
  X = rand(4, 4); B = X * X'; rho_b = B / tr(B)
  X = rand(4, 4); C = X * X'; rho_c = C / tr(C)
  push!(first_list, [rho_a, rho_b, rho_c])
end

first_list_list = [first_list]

for i in 1:100
  second_list = f_1(first_list_list[i])
  next_first_list = f_2(second_list)
  push!(first_list_list, next_first_list)
end
