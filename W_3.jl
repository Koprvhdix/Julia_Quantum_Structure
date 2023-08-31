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

rho = 0.45 * rho + (0.55 / 8) * I(8)

function f_1(X_list)
  rho_1 = Semidefinite(2)
  rho_2 = Semidefinite(2)
  rho_3 = Semidefinite(2)

  rho_next = kron(rho_1, X_list[1][1])
  rho_next += (exchange * kron(X_list[1][2], rho_2) * exchange)
  rho_next += kron(X_list[1][3], rho_3)

  rho_list = [ [rho_1, rho_2, rho_3 ] ]

  for index in 2:length(X_list)
    rho_1 = Semidefinite(2)
    rho_2 = Semidefinite(2)
    rho_3 = Semidefinite(2)

    rho_next += kron(rho_1, X_list[index][1])
    rho_next +=(exchange * kron(X_list[index][2], rho_2) * exchange)
    rho_next += kron(X_list[index][3], rho_3)

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
  rho_next += (exchange * kron(rho_2, X_list[1][2]) * exchange)
  rho_next += kron(rho_3, X_list[1][3])

  rho_list = [ [rho_1, rho_2, rho_3 ] ]

  for index in 2:length(X_list)
    rho_1 = Semidefinite(4)
    rho_2 = Semidefinite(4)
    rho_3 = Semidefinite(4)

    rho_next += kron(X_list[index][1], rho_1)
    rho_next += (exchange * kron(rho_2, X_list[index][2]) * exchange)
    rho_next += kron(rho_3, X_list[index][3])

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
  # println(evaluate(rho_list))

  next_X = [ [item[1].value, item[2].value, item[3].value] for item in evaluate(rho_list) ]
  # println(next_X)
  return next_X
end

X = rand(4, 4); A = X * X'; rho_a = A / tr(A)
X = rand(4, 4); B = X * X'; rho_b = B / tr(B)
X = rand(4, 4); C = X * X'; rho_c = C / tr(C)

X_list = [ [rho_a, rho_b, rho_c] ]

for i in 1:299
  X = rand(4, 4); A = X * X'; rho_a = A / tr(A)
  X = rand(4, 4); B = X * X'; rho_b = B / tr(B)
  X = rand(4, 4); C = X * X'; rho_c = C / tr(C)
  push!(X_list, [rho_a, rho_b, rho_c])
end

# for i in 1:10
#   next_X = f_1(X_list)
#   X_list = f_2(next_X)
# end
next_X = f_1(X_list)
X_2 = f_2(next_X)
next_2 = f_1(X_2)
X_3 = f_2(next_2)
next_3 = f_1(X_3)
X_4 = f_2(next_3)
next_4 = f_1(X_4)
X_5 = f_2(next_4)
next_5 = f_1(X_5)
X_6 = f_2(next_5)
