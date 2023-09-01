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

  constraints = [rho_1 in :SDP, rho_2 in :SDP, rho_3 in :SDP, rho_1[1, 1] > 0, rho_1[2, 2] > 0, rho_2[1, 1] > 0, rho_2[2, 2] > 0,  rho_3[1, 1] > 0, rho_3[2, 2] > 0 ]

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

    push!(constraints, rho_1 in :SDP)
    push!(constraints, rho_1[1, 1] > 0, rho_1[2, 2] > 0)
    push!(constraints, rho_2 in :SDP)
    push!(constraints, rho_2[1, 1] > 0, rho_2[2, 2] > 0)
    push!(constraints, rho_3 in :SDP)
    push!(constraints, rho_3[1, 1] > 0, rho_3[2, 2] > 0)

    push!(rho_list, [rho_1, rho_2, rho_3 ] )
  end

  objective = -tr(rho_next)
  push!(constraints, (rho - rho_next) in :SDP)

  problem = minimize(objective, constraints)
  solve!(problem, SCS.Optimizer)
  # println(problem.optval)

  next_X = [ [item[1].value, item[2].value, item[3].value] for item in evaluate(rho_list) ]
  return next_X, problem.optval
end

function f_2(X_list)
  rho_1 = Semidefinite(4)
  rho_2 = Semidefinite(4)
  rho_3 = Semidefinite(4)

  constraints = [rho_1 in :SDP, rho_2 in :SDP, rho_3 in :SDP, rho_1[1, 1] > 0, rho_1[2, 2] > 0, rho_2[1, 1] > 0, rho_2[2, 2] > 0,  rho_3[1, 1] > 0, rho_3[2, 2] > 0 ]

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

    push!(constraints, rho_1 in :SDP)
    push!(constraints, rho_1[1, 1] > 0, rho_1[2, 2] > 0)
    push!(constraints, rho_2 in :SDP)
    push!(constraints, rho_2[1, 1] > 0, rho_2[2, 2] > 0)
    push!(constraints, rho_3 in :SDP)
    push!(constraints, rho_3[1, 1] > 0, rho_3[2, 2] > 0)

    push!(rho_list, [rho_1, rho_2, rho_3 ] )
  end

  objective = -tr(rho_next)
  push!(constraints, (rho - rho_next) in :SDP)

  problem = minimize(objective, constraints)
  solve!(problem, SCS.Optimizer)

  next_X = [ [item[1].value, item[2].value, item[3].value] for item in evaluate(rho_list) ]
  # println(next_X)
  return next_X, problem.optval
end

X = rand(4, 4); A = X * X'; rho_a = A / tr(A)
X = rand(4, 4); B = X * X'; rho_b = B / tr(B)
X = rand(4, 4); C = X * X'; rho_c = C / tr(C)

first_list = [ [rho_a, rho_b, rho_c] ]

for i in 1:299
  X = rand(4, 4); A = X * X'; rho_a = A / tr(A)
  X = rand(4, 4); B = X * X'; rho_b = B / tr(B)
  X = rand(4, 4); C = X * X'; rho_c = C / tr(C)
  push!(first_list, [rho_a, rho_b, rho_c])
end

first_list_list = [first_list]
second_list_list = []
optival_list = [0.0]

for i in 1:100
  second_list, optval = f_1(first_list_list[i])
  push!(second_list_list, second_list)

  if optval > optival_list[end] * 0.95
    println("Second List")
    println(optval, optival_list[end])
    println(second_list_list[end - 1])
    println(second_list_list[end])
    break
  end
  push!(optival_list, optval)

  next_first_list, optval = f_2(second_list)
  push!(first_list_list, next_first_list)

  if optval > optival_list[end] * 0.95
    println("First List")
    println(optval, optival_list[end])
    println(first_list_list[end - 1])
    println(first_list_list[end])
    break
  end
  push!(optival_list, optval)
end
