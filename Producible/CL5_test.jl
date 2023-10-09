include("BaseModule.jl")

using Convex, SCS, MosekTools
using LinearAlgebra
using Random, RandomMatrices

function swap_chars(s, i, j)
  lst = collect(s)
  lst[i], lst[j] = lst[j], lst[i]
  return join(lst)
end

function nlize(rho)
  evs = eigvals(rho)
  revs = [real(it) for it in evs]
  ievs = [imag(it) for it in evs]
  if ievs'*ievs/(revs'*revs) > 1e-3 || minimum(revs) < 0
    println(revs)
  end
end

N = 5

function get_exchange_matrix(N, num1, num2)
  the_matrix = zeros(Int64, 2 ^ N, 2 ^ N)
  for number in 0:(2 ^ N - 1)
    number_str = lpad(string(number, base=2), N, '0')
    number_str = swap_chars(number_str, num1, num2)
    number_23 = parse(Int64, number_str, base = 2)
    the_matrix[number + 1, number_23 + 1] = 1
  end
  return(the_matrix)
end

exchange_23 = get_exchange_matrix(N, 2, 3)
exchange_24 = get_exchange_matrix(N, 2, 3)
exchange_25 = get_exchange_matrix(N, 2, 3)
exchange_34 = get_exchange_matrix(N, 3, 4)
exchange_35 = get_exchange_matrix(N, 3, 5)
exchange_45 = get_exchange_matrix(N, 4, 5)
exchange_13 = get_exchange_matrix(N, 1, 3)

Cl5_matrix = zeros(32, 32)

indices = [1, 16, 20, 29]
for index in indices
    for index2 in indices
        Cl5_matrix[index, index2] = 0.25
    end
end

struct MultiState
    mat::Matrix{Complex{Float64}}
    dims::Array{Int,1}
end;
    
rho = MultiState(Cl5_matrix, [2, 2, 2, 2, 2])
  
function randState(dim, trace)
    V=randn(Complex{Float64},dim);
    V/=norm(V);
    V = V*conj(transpose(V))
    return V * trace
end

function part_4_rho_next(X_list1, X_list2, nps, train_part)
    if train_part == 1
        rho_1s = [HermitianSemidefinite(4) for i in 1:nps]
        rho_2s = [HermitianSemidefinite(4) for i in 1:nps]
        rho_3s = [HermitianSemidefinite(4) for i in 1:nps]
        rho_4s = [HermitianSemidefinite(4) for i in 1:nps]
        rho_5s = [HermitianSemidefinite(4) for i in 1:nps]
        rho_6s = [HermitianSemidefinite(4) for i in 1:nps]
        rho_7s = [HermitianSemidefinite(4) for i in 1:nps]
        rho_8s = [HermitianSemidefinite(4) for i in 1:nps]
        rho_9s = [HermitianSemidefinite(4) for i in 1:nps]
        rho_10s = [HermitianSemidefinite(4) for i in 1:nps]

        rho_1_1s = [HermitianSemidefinite(2) for i in 1:nps]
        rho_1_2s = [HermitianSemidefinite(2) for i in 1:nps]
        rho_1_3s = [HermitianSemidefinite(2) for i in 1:nps]
        rho_1_4s = [HermitianSemidefinite(2) for i in 1:nps]
        rho_1_5s = [HermitianSemidefinite(2) for i in 1:nps]
        rho_1_6s = [HermitianSemidefinite(2) for i in 1:nps]
        rho_1_7s = [HermitianSemidefinite(2) for i in 1:nps]
        rho_1_8s = [HermitianSemidefinite(2) for i in 1:nps]
        rho_1_9s = [HermitianSemidefinite(2) for i in 1:nps]
        rho_1_10s = [HermitianSemidefinite(2) for i in 1:nps]
    
        rho_next = sum(kron(kron(kron(rho_1s[i], rho_1_1s[i]), X_list1[1][i]), X_list2[1][i]) for i in 1:nps) + 
        sum((exchange_23 * kron(kron(kron(rho_2s[i], rho_1_2s[i]), X_list1[2][i]), X_list2[2][i]) * exchange_23) for i in 1:nps) + 
        sum((exchange_24 * kron(kron(kron(rho_3s[i], rho_1_3s[i]), X_list1[3][i]), X_list2[3][i]) * exchange_24) for i in 1:nps) +
        sum((exchange_25 * kron(kron(kron(rho_4s[i], rho_1_4s[i]), X_list1[4][i]), X_list2[4][i]) * exchange_25) for i in 1:nps) +
        sum(kron(kron(kron(rho_1_5s[i], rho_5s[i]), X_lis12[5][i]), X_list2[5][i]) for i in 1:nps) +
        sum((exchange_34 * kron(kron(kron(rho_1_6s[i], rho_6s[i]), X_list1[6][i]), X_list2[6][i]) * exchange_34) for i in 1:nps) +
        sum((exchange_35 * kron(kron(kron(rho_1_7s[i], rho_7s[i]), X_list1[7][i]), X_list2[7][i]) * exchange_35) for i in 1:nps) +
        sum(kron(kron(kron(rho_1_8s[i], X_list1[8][i]), rho_8s[i]), X_list2[8][i]) for i in 1:nps) +
        sum((exchange_45 * kron(kron(kron(rho_1_9s[i], X_list1[9][i]), rho_9s[i]), X_list2[9][i]) * exchange_45) for i in 1:nps) +
        sum(kron(kron(kron(rho_1_10s[i], X_list1[10][i]), X_list2[10][i]), rho_10s[i]) for i in 1:nps) 
    
        return rho_next, [rho_1s, rho_2s, rho_3s, rho_4s, rho_5s, rho_6s, rho_7s, rho_8s, rho_9s, rho_10s], [rho_1_1s , rho_1_2s , rho_1_3s , rho_1_4s , rho_1_5s , rho_1_6s , rho_1_7s , rho_1_8s , rho_1_9s , rho_1_10s]
    else
        rho_1s = [HermitianSemidefinite(2) for i in 1:nps]
        rho_2s = [HermitianSemidefinite(2) for i in 1:nps]
        rho_3s = [HermitianSemidefinite(2) for i in 1:nps]
        rho_4s = [HermitianSemidefinite(2) for i in 1:nps]
        rho_5s = [HermitianSemidefinite(2) for i in 1:nps]
        rho_6s = [HermitianSemidefinite(2) for i in 1:nps]
        rho_7s = [HermitianSemidefinite(2) for i in 1:nps]
        rho_8s = [HermitianSemidefinite(2) for i in 1:nps]
        rho_9s = [HermitianSemidefinite(2) for i in 1:nps]
        rho_10s = [HermitianSemidefinite(2) for i in 1:nps]

        rho_1_1s = [HermitianSemidefinite(2) for i in 1:nps]
        rho_1_2s = [HermitianSemidefinite(2) for i in 1:nps]
        rho_1_3s = [HermitianSemidefinite(2) for i in 1:nps]
        rho_1_4s = [HermitianSemidefinite(2) for i in 1:nps]
        rho_1_5s = [HermitianSemidefinite(2) for i in 1:nps]
        rho_1_6s = [HermitianSemidefinite(2) for i in 1:nps]
        rho_1_7s = [HermitianSemidefinite(2) for i in 1:nps]
        rho_1_8s = [HermitianSemidefinite(2) for i in 1:nps]
        rho_1_9s = [HermitianSemidefinite(2) for i in 1:nps]
        rho_1_10s = [HermitianSemidefinite(2) for i in 1:nps]
    
        rho_next = sum(kron(kron(kron(X_list1[1][i], X_list2[1][i]), rho_1s[i]), rho_1_1s[i]) for i in 1:nps) + 
        sum((exchange_23 * kron(kron(kron(X_list1[2][i], X_list2[2][i]), rho_2s[i]), rho_1_2s[i]) * exchange_23) for i in 1:nps) + 
        sum((exchange_24 * kron(kron(kron(X_list1[3][i], X_list2[3][i]), rho_3s[i]), rho_1_3s[i]) * exchange_24) for i in 1:nps) +
        sum((exchange_25 * kron(kron(kron(X_list1[4][i], X_list2[4][i]), rho_4s[i]), rho_1_4s[i]) * exchange_25) for i in 1:nps) +
        sum(kron(kron(kron(X_list2[5][i], X_list1[5][i]), rho_5s[i]), rho_1_5s[i]) for i in 1:nps) +
        sum((exchange_34 * kron(kron(kron(X_list2[6][i], X_list1[6][i]), rho_6s[i]), rho_1_6s[i]) * exchange_34) for i in 1:nps) +
        sum((exchange_35 * kron(kron(kron(X_list2[7][i], X_list1[7][i]), rho_7s[i]), rho_1_7s[i]) * exchange_35) for i in 1:nps) +
        sum(kron(kron(kron(X_list2[8][i], rho_8s[i]), X_list1[8][i]), rho_1_8s[i]) for i in 1:nps) +
        sum((exchange_45 * kron(kron(kron(X_list2[9][i], rho_9s[i]), X_list1[9][i]), rho_1_9s[i]) * exchange_45) for i in 1:nps) +
        sum(kron(kron(kron(X_list2[10][i], rho_10s[i]), rho_1_10s[i]), X_list1[10][i]) for i in 1:nps) 
    
        return rho_next, [rho_1s, rho_2s, rho_3s, rho_4s, rho_5s, rho_6s, rho_7s, rho_8s, rho_9s, rho_10s], [rho_1_1s , rho_1_2s , rho_1_3s , rho_1_4s , rho_1_5s , rho_1_6s , rho_1_7s , rho_1_8s , rho_1_9s , rho_1_10s]
    end
end

function prod_3_rho_next(part_2_qubit, part_3_qubit, nps, train_part)
  if train_part == 1
    rho_1s = [HermitianSemidefinite(8) for i in 1:nps]
    rho_2s = [HermitianSemidefinite(8) for i in 1:nps]
    rho_3s = [HermitianSemidefinite(8) for i in 1:nps]
    rho_4s = [HermitianSemidefinite(8) for i in 1:nps]
    rho_5s = [HermitianSemidefinite(8) for i in 1:nps]
    rho_6s = [HermitianSemidefinite(8) for i in 1:nps]
    rho_7s = [HermitianSemidefinite(8) for i in 1:nps]
    rho_8s = [HermitianSemidefinite(8) for i in 1:nps]
    rho_9s = [HermitianSemidefinite(8) for i in 1:nps]
    rho_10s = [HermitianSemidefinite(8) for i in 1:nps]

    rho_next = sum(kron(rho_1s[i], part_2_qubit[1][i]) for i in 1:nps) + 
    sum((exchange_34 * kron(rho_2s[i], part_2_qubit[2][i]) * exchange_34) for i in 1:nps) + 
    sum((exchange_35 * kron(rho_3s[i], part_2_qubit[3][i]) * exchange_35) for i in 1:nps) + 
    sum((exchange_24 * kron(rho_4s[i], part_2_qubit[4][i]) * exchange_24) for i in 1:nps) + 
    sum((exchange_25 * kron(rho_5s[i], part_2_qubit[5][i]) * exchange_25) for i in 1:nps) + 
    sum((exchange_13 * kron(part_2_qubit[6][i], rho_6s[i]) * exchange_13) for i in 1:nps) + 
    sum(kron(part_2_qubit[7][i], rho_7s[i]) for i in 1:nps) + 
    sum((exchange_23 * kron(part_2_qubit[8][i], rho_8s[i]) * exchange_23) for i in 1:nps) + 
    sum((exchange_24 * kron(part_2_qubit[9][i], rho_9s[i]) * exchange_24) for i in 1:nps) + 
    sum((exchange_25 * kron(part_2_qubit[10][i], rho_10s[i]) * exchange_25) for i in 1:nps)

    return rho_next, [rho_1s, rho_2s, rho_3s, rho_4s, rho_5s, rho_6s, rho_7s, rho_8s, rho_9s, rho_10s]
  else
    rho_1s = [HermitianSemidefinite(4) for i in 1:nps]
    rho_2s = [HermitianSemidefinite(4) for i in 1:nps]
    rho_3s = [HermitianSemidefinite(4) for i in 1:nps]
    rho_4s = [HermitianSemidefinite(4) for i in 1:nps]
    rho_5s = [HermitianSemidefinite(4) for i in 1:nps]
    rho_6s = [HermitianSemidefinite(4) for i in 1:nps]
    rho_7s = [HermitianSemidefinite(4) for i in 1:nps]
    rho_8s = [HermitianSemidefinite(4) for i in 1:nps]
    rho_9s = [HermitianSemidefinite(4) for i in 1:nps]
    rho_10s = [HermitianSemidefinite(4) for i in 1:nps]

    rho_next = sum(kron(part_3_qubit[1][i], rho_1s[i]) for i in 1:nps) + 
    sum((exchange_34 * kron(part_3_qubit[2][i], rho_2s[i]) * exchange_34) for i in 1:nps) + 
    sum((exchange_35 * kron(part_3_qubit[3][i], rho_3s[i]) * exchange_35) for i in 1:nps) + 
    sum((exchange_24 * kron(part_3_qubit[4][i], rho_4s[i]) * exchange_24) for i in 1:nps) + 
    sum((exchange_25 * kron(part_3_qubit[5][i], rho_5s[i]) * exchange_25) for i in 1:nps) + 
    sum((exchange_13 * kron(rho_6s[i], part_3_qubit[6][i]) * exchange_13) for i in 1:nps) + 
    sum(kron(rho_7s[i], part_3_qubit[7][i]) for i in 1:nps) + 
    sum((exchange_23 * kron(rho_8s[i], part_3_qubit[8][i]) * exchange_23) for i in 1:nps) + 
    sum((exchange_24 * kron(rho_9s[i], part_3_qubit[9][i]) * exchange_24) for i in 1:nps) + 
    sum((exchange_25 * kron(rho_10s[i], part_3_qubit[10][i]) * exchange_25) for i in 1:nps)

    return rho_next, [rho_1s, rho_2s, rho_3s, rho_4s, rho_5s, rho_6s, rho_7s, rho_8s, rho_9s, rho_10s]
  end
end

function part_2_rho_next(part_2_qubit, part_3_qubit, part_4_qubit, part_1_qubit, nps, train_part)
  if train_part == 1
    rho_1s = [HermitianSemidefinite(8) for i in 1:nps]
    rho_2s = [HermitianSemidefinite(8) for i in 1:nps]
    rho_3s = [HermitianSemidefinite(8) for i in 1:nps]
    rho_4s = [HermitianSemidefinite(8) for i in 1:nps]
    rho_5s = [HermitianSemidefinite(8) for i in 1:nps]
    rho_6s = [HermitianSemidefinite(8) for i in 1:nps]
    rho_7s = [HermitianSemidefinite(8) for i in 1:nps]
    rho_8s = [HermitianSemidefinite(8) for i in 1:nps]
    rho_9s = [HermitianSemidefinite(8) for i in 1:nps]
    rho_10s = [HermitianSemidefinite(8) for i in 1:nps]

    rho_2_1s = [HermitianSemidefinite(16) for i in 1:nps]
    rho_2_2s = [HermitianSemidefinite(16) for i in 1:nps]
    rho_2_3s = [HermitianSemidefinite(16) for i in 1:nps]
    rho_2_4s = [HermitianSemidefinite(16) for i in 1:nps]
    rho_2_5s = [HermitianSemidefinite(16) for i in 1:nps]

    rho_next = sum(kron(rho_1s[i], part_2_qubit[1][i]) for i in 1:nps) + 
    sum((exchange_34 * kron(rho_2s[i], part_2_qubit[2][i]) * exchange_34) for i in 1:nps) + 
    sum((exchange_35 * kron(rho_3s[i], part_2_qubit[3][i]) * exchange_35) for i in 1:nps) + 
    sum((exchange_24 * kron(rho_4s[i], part_2_qubit[4][i]) * exchange_24) for i in 1:nps) + 
    sum((exchange_25 * kron(rho_5s[i], part_2_qubit[5][i]) * exchange_25) for i in 1:nps) + 
    sum((exchange_13 * kron(part_2_qubit[6][i], rho_6s[i]) * exchange_13) for i in 1:nps) + 
    sum(kron(part_2_qubit[7][i], rho_7s[i]) for i in 1:nps) + 
    sum((exchange_23 * kron(part_2_qubit[8][i], rho_8s[i]) * exchange_23) for i in 1:nps) + 
    sum((exchange_24 * kron(part_2_qubit[9][i], rho_9s[i]) * exchange_24) for i in 1:nps) + 
    sum((exchange_25 * kron(part_2_qubit[10][i], rho_10s[i]) * exchange_25) for i in 1:nps)

    return rho_next, [rho_1s, rho_2s, rho_3s, rho_4s, rho_5s, rho_6s, rho_7s, rho_8s, rho_9s, rho_10s]
  else
    rho_1s = [HermitianSemidefinite(4) for i in 1:nps]
    rho_2s = [HermitianSemidefinite(4) for i in 1:nps]
    rho_3s = [HermitianSemidefinite(4) for i in 1:nps]
    rho_4s = [HermitianSemidefinite(4) for i in 1:nps]
    rho_5s = [HermitianSemidefinite(4) for i in 1:nps]
    rho_6s = [HermitianSemidefinite(4) for i in 1:nps]
    rho_7s = [HermitianSemidefinite(4) for i in 1:nps]
    rho_8s = [HermitianSemidefinite(4) for i in 1:nps]
    rho_9s = [HermitianSemidefinite(4) for i in 1:nps]
    rho_10s = [HermitianSemidefinite(4) for i in 1:nps]

    rho_2_1s = [HermitianSemidefinite(2) for i in 1:nps]
    rho_2_2s = [HermitianSemidefinite(2) for i in 1:nps]
    rho_2_3s = [HermitianSemidefinite(2) for i in 1:nps]
    rho_2_4s = [HermitianSemidefinite(2) for i in 1:nps]
    rho_2_5s = [HermitianSemidefinite(2) for i in 1:nps]

    rho_next = sum(kron(part_3_qubit[1][i], rho_1s[i]) for i in 1:nps) + 
    sum((exchange_34 * kron(part_3_qubit[2][i], rho_2s[i]) * exchange_34) for i in 1:nps) + 
    sum((exchange_35 * kron(part_3_qubit[3][i], rho_3s[i]) * exchange_35) for i in 1:nps) + 
    sum((exchange_24 * kron(part_3_qubit[4][i], rho_4s[i]) * exchange_24) for i in 1:nps) + 
    sum((exchange_25 * kron(part_3_qubit[5][i], rho_5s[i]) * exchange_25) for i in 1:nps) + 
    sum((exchange_13 * kron(rho_6s[i], part_3_qubit[6][i]) * exchange_13) for i in 1:nps) + 
    sum(kron(rho_7s[i], part_3_qubit[7][i]) for i in 1:nps) + 
    sum((exchange_23 * kron(rho_8s[i], part_3_qubit[8][i]) * exchange_23) for i in 1:nps) + 
    sum((exchange_24 * kron(rho_9s[i], part_3_qubit[9][i]) * exchange_24) for i in 1:nps) + 
    sum((exchange_25 * kron(rho_10s[i], part_3_qubit[10][i]) * exchange_25) for i in 1:nps)

    return rho_next, [rho_1s, rho_2s, rho_3s, rho_4s, rho_5s, rho_6s, rho_7s, rho_8s, rho_9s, rho_10s]
  end
end

function train(rho_next)
    t = Variable(1,Positive());
  
    II= zeros(Complex{Float64},prod(rho.dims),prod(rho.dims))+I;
    problem= maximize(t);
    problem.constraints += ((t*rho.mat+(1.0-t)/prod(rho.dims)*II) == rho_next);
  
    solve!(problem, Mosek.Optimizer, silent_solver=true)
    println(problem.optval)
    problem.optval
end
  
function part_4_train()
    # 2|1|1|1
    nps = 20
    for j in 1:20
      X_list_3 = [[randState(2, 1) for index2 in 1:nps] for index1 in 1:10]
      X_list_4 = [[randState(2, 1) for index2 in 1:nps] for index1 in 1:10]
      for i in 1:30
        println("finish init")
        rho_next, rhos_list, rhos_list2 = part_4_rho_next(X_list_3, X_list_4, nps, 1)
        println("start train")
        optval = train(rho_next)
        temp_X_list_1 = [[current_rho.value for current_rho in rhos] for rhos in rhos_list]
        temp_X_list_2 = [[current_rho.value for current_rho in rhos] for rhos in rhos_list2]
        for index1 in 1:6
          for index2 in 1:nps
            nlize(temp_X_list_1[index1][index2])
            nlize(temp_X_list_2[index1][index2])
          end
        end
        X_list_1 = [[temp_X_list_1[index1][index2] * tr(X_list_3[index1][index2]) for index2 in 1:nps] for index1 in 1:10]
        X_list_2 = [[temp_X_list_2[index1][index2] * tr(X_list_4[index1][index2]) for index2 in 1:nps] for index1 in 1:10]
  
        rho_next, rhos_list, rhos_list2 = part_4_rho_next(X_list_1, X_list_2, nps, 2)
        optval = train(rho_next)
        temp_X_list_3 = [[current_rho.value for current_rho in rhos] for rhos in rhos_list]
        temp_X_list_4 = [[current_rho.value for current_rho in rhos] for rhos in rhos_list2]
        for index1 in 1:6
          for index2 in 1:nps
            nlize(temp_X_list_3[index1][index2])
            nlize(temp_X_list_4[index1][index2])
          end
        end
        X_list_3 = [[temp_X_list_3[index1][index2] * tr(X_list_1[index1][index2]) for index2 in 1:nps] for index1 in 1:10]
        X_list_4 = [[temp_X_list_4[index1][index2] * tr(X_list_2[index1][index2]) for index2 in 1:nps] for index1 in 1:10]
      end
      println("one round ", j)
    end
end

# function prod_2_train()
#     # 2|2|1
# end

# function part_3_train()
#     # 2|2|1   3|1|1
# end

# function prod_3_train()
#     # 3|2|
#     nps = 50
#     for j in 1:20
#       X_list_2 = [[randState(4, nps) for index2 in 1:nps] for index1 in 1:10]
#       for i in 1:30
#         println("finish init")
#         rho_next, rhos_list = prod_3_rho_next(X_list_2, nps, 1)
#         println("start train")
#         optval = train(rho_next)
#         temp_X_list_1 = [[current_rho.value for current_rho in rhos] for rhos in rhos_list]
#         for index1 in 1:6
#           for index2 in 1:nps
#             nlize(temp_X_list_1[index1][index2])
#           end
#         end
#         X_list_1 = [[temp_X_list_1[index1][index2] * tr(X_list_2[index1][index2]) for index2 in 1:nps] for index1 in 1:10]

#         rho_next, rhos_list = prod_3_rho_next(X_list_1, nps, 2)
#         println("start train")
#         optval = train(rho_next)
#         temp_X_list_2 = [[current_rho.value for current_rho in rhos] for rhos in rhos_list]
#         for index1 in 1:6
#           for index2 in 1:nps
#             nlize(temp_X_list_2[index1][index2])
#           end
#         end
#         X_list_2 = [[temp_X_list_2[index1][index2] * tr(X_list_1[index1][index2]) for index2 in 1:nps] for index1 in 1:10]
#       end
      
#     end
# end

function main_train()
  nps = 50
  # 3|2   4|1
  # random init
  part_3_qubit = [[randState(8, nps) for index2 in 1:nps] for index1 in 1:10]
  part_2_qubit = [[randState(4, nps) for index2 in 1:nps] for index1 in 1:10]

  part_4_qubit = [[randState(16, nps) for index2 in 1:nps] for index1 in 1:5]
  part_1_qubit = [[randState(2, nps) for index2 in 1:nps] for index1 in 1:5]
  for i in 1:30
    rho_next, rhos_list = part_2_rho_next(part_2_qubit, part_3_qubit, part_4_qubit, part_1_qubit, nps, 1)
    optval = train(rho_next)
    temp_X_list = [[current_rho.value for current_rho in rhos] for rhos in rhos_list]
    for index1 in 1:6
      for index2 in 1:nps
        nlize(temp_X_list[index1][index2])
      end
    end
    part_3_qubit = [[temp_X_list[index1][index2] * tr(part_2_qubit[index1][index2]) for index2 in 1:nps] for index1 in 1:10]

    rho_next, rhos_list = part_2_rho_next(part_2_qubit, part_3_qubit, part_4_qubit, part_1_qubit, nps, 2)
    optval = train(rho_next)
    temp_X_list = [[current_rho.value for current_rho in rhos] for rhos in rhos_list]
    for index1 in 1:6
      for index2 in 1:nps
        nlize(temp_X_list[index1][index2])
      end
    end
    part_3_qubit = [[temp_X_list[index1][index2] * tr(part_2_qubit[index1][index2]) for index2 in 1:nps] for index1 in 1:10]
  end

  # 3|2
  for i in 1:30
    rho_next, rhos_list = prod_3_rho_next(part_2_qubit, part_3_qubit, nps, 1)
    optval = train(rho_next)
    temp_X_list = [[current_rho.value for current_rho in rhos] for rhos in rhos_list]
    for index1 in 1:6
      for index2 in 1:nps
        nlize(temp_X_list[index1][index2])
      end
    end
    part_3_qubit = [[temp_X_list[index1][index2] * tr(part_2_qubit[index1][index2]) for index2 in 1:nps] for index1 in 1:10]

    rho_next, rhos_list = prod_3_rho_next(part_2_qubit, part_3_qubit, nps, 2)
    optval = train(rho_next)
    temp_X_list = [[current_rho.value for current_rho in rhos] for rhos in rhos_list]
    for index1 in 1:6
      for index2 in 1:nps
        nlize(temp_X_list[index1][index2])
      end
    end
    part_2_qubit = [[temp_X_list[index1][index2] * tr(part_3_qubit[index1][index2]) for index2 in 1:nps] for index1 in 1:10]
  end

  # 2|2|1   3|1|1

  # 2|2|1

  # 2|1|1|1
end


main_train()
