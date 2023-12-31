using Convex, SCS

using LinearAlgebra

p = Variable()

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

rho = 0.4 * rho + (0.6 / 8) * I(8)

function f_1(X_list)
  rho_1 = Semidefinite(2)
  rho_2 = Semidefinite(2)
  rho_3 = Semidefinite(2)

  constraints = [rho_1 in :SDP, rho_2 in :SDP, rho_3 in :SDP]

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

    constraints += [rho_1 in :SDP, rho_2 in :SDP, rho_3 in :SDP]

    push!(rho_list, [rho_1, rho_2, rho_3 ] )
  end

  # objective = -tr(rho_next)

  constraints += [rho == rho_next, p < 1]

  problem = minimize(p, constraints)
  solve!(problem, SCS.Optimizer)
  # println(problem.optval)

  next_X = [ [item[1].value, item[2].value, item[3].value] for item in evaluate(rho_list) ]
  return next_X, problem.optval
end

function f_2(X_list)
  rho_1 = Semidefinite(4)
  rho_2 = Semidefinite(4)
  rho_3 = Semidefinite(4)

  constraints = [rho_1 in :SDP, rho_2 in :SDP, rho_3 in :SDP]
  
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

    constraints += [rho_1 in :SDP, rho_2 in :SDP, rho_3 in :SDP]

    push!(rho_list, [rho_1, rho_2, rho_3 ] )
  end

  constraints += [rho == rho_next, p < 1]

  problem = minimize(p, constraints)
  # problem = minimize(objective, constraints)
  solve!(problem, SCS.Optimizer)

  next_X = [ [item[1].value, item[2].value, item[3].value] for item in evaluate(rho_list) ]

  # rho_result = kron(X_list[1][1], next_X[1][1])
  # rho_next += (exchange * kron(next_X[1][2], X_list[1][2]) * exchange)
  # rho_next += kron(next_X[1][3], X_list[1][3])
  # for index in 2:length(X_list)
  #   rho_result += kron(X_list[index][1], next_X[index][1])
  #   rho_result += (exchange * kron(next_X[index][2], X_list[index][2]) * exchange)
  #   rho_result += kron(next_X[index][3], X_list[index][3])
  # end

  # println(rho_result)
  # println(eigvals(rho - rho_result))

  return next_X, problem.optval
end

X = rand(4, 4); A = X * X'; rho_a = A / tr(A)
X = rand(4, 4); B = X * X'; rho_b = B / tr(B)
X = rand(4, 4); C = X * X'; rho_c = C / tr(C)

first_list = [ [rho_a, rho_b, rho_c] ]

println("Start")

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

  # if optval > optival_list[end] * 0.95
  #   println("Second List")
  #   println(optval, optival_list[end])
  #   println(second_list_list[end - 1])
  #   println(second_list_list[end])
  #   break
  # end
  # push!(optival_list, optval)

  next_first_list, optval = f_2(second_list)
  push!(first_list_list, next_first_list)

#   if optval > optival_list[end] * 0.95
#     println("First List")
#     println(optval, optival_list[end])
#     println(first_list_list[end - 1])
#     println(first_list_list[end])
#     break
#   end
#   push!(optival_list, optval)
end


# rho_0_1 = Semidefinite(4)
# rho_0_2 = Semidefinite(4)
# rho_0_3 = Semidefinite(4)
# X = rand(4, 4); A = X * X'; rho_0_a = A / tr(A)
# X = rand(4, 4); B = X * X'; rho_0_b = B / tr(B)
# X = rand(4, 4); C = X * X'; rho_0_c = C / tr(C)

# println(rho_0_a)
# println(rho_0_b)

# rho_next = kron(rho_0_1, rho_0_a) + kron(rho_0_2, rho_0_b)

# constraints = [ rho_0_1 ⪰ 0, rho_0_2 ⪰ 0, rho_next == rho_p]

# problem = minimize(p, constraints)

# solve!(problem, SCS.Optimizer)

# println(eigvals(evaluate(rho_0_1)), eigvals(evaluate(rho_0_2)))
# println(eigvals(evaluate(rho_next)))
