using Convex, SCS, MosekTools
using LinearAlgebra
using Random, RandomMatrices

orho = [0.0  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  ;
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

exchange_2_qubit = [1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1]

exchange_12 = [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. ;
0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. ;
0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. ;
0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. ;
0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. ;
0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. ;
0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. ;
0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. ;
0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. ;
0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. ;
0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. ;
0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. ;
0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. ;
0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. ;
0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. ;
0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. ]

exchange_13 = [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. ;
0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. ;
0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. ;
0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. ;
0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. ;
0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. ;
0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. ;
0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. ;
0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. ;
0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. ;
0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. ;
0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. ;
0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. ;
0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. ;
0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. ;
0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. ]

exchange_14 = [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. ;
0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. ;
0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. ;
0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. ;
0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. ;
0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. ;
0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. ;
0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. ;
0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. ;
0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. ;
0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. ;
0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. ;
0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. ;
0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. ;
0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. ;
0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. ]

exchange_23 = [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. ;
0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. ;
0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. ;
0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. ;
0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. ;
0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. ;
0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. ;
0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. ;
0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. ;
0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. ;
0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. ;
0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. ;
0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. ;
0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. ;
0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. ;
0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. ]

exchange_24 = [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. ;
0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. ;
0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. ;
0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. ;
0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. ;
0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. ;
0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. ;
0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. ;
0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. ;
0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. ;
0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. ;
0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. ;
0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. ;
0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. ;
0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. ;
0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. ]

exchange_34 = [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. ;
0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. ;
0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. ;
0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. ;
0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. ;
0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. ;
0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. ;
0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. ;
0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. ;
0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. ;
0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. ;
0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. ;
0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. ;
0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. ;
0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. ;
0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. ]

function decompose_and_reconstruct_positive(A::AbstractMatrix, nps)
  F = eigen(A)
  V = F.vectors
  D = Diagonal(F.values)
  
  for i in 1:size(D, 1)
    D[i, i] = real(D[i, i])
    if real(D[i, i]) < 0
      D[i, i] = 0.0
    end
  end

  A_reconstructed = V * D * V'
  for i in 1:size(A_reconstructed, 1)
    A_reconstructed[i, i] = real(A_reconstructed[i, i])
  end

  new_A = A_reconstructed/(nps * tr(A_reconstructed))
  if any(isnan.(new_A)) || any(isinf.(new_A))
    return randState(size(A_reconstructed, 1)) / nps
  else
    return new_A
  end
end

function randState(dim)
  V=randn(Complex{Float64},dim);
  V/=norm(V);
  return V = V*conj(transpose(V))
end

struct MultiState
  mat::Matrix{Complex{Float64}}
  dims::Array{Int,1}
end;

rho = MultiState(orho, [2, 2, 2, 2])

function test_part_2(X_list_2, nps, train_part)
  if train_part == 1
    rho_1s = [HermitianSemidefinite(2) for i in 1:nps]

    rho_next = sum(kron(rho_1s[i], X_list_2[i]) for i in 1:nps) +
    sum((exchange_12 * kron(rho_1s[i], X_list_2[i]) * exchange_12) for i in 1:nps) +
    sum((exchange_13 * kron(rho_1s[i], X_list_2[i]) * exchange_13) for i in 1:nps) +
    sum((exchange_14 * kron(rho_1s[i], X_list_2[i]) * exchange_14) for i in 1:nps)

    return rho_next, rho_1s
  else
    rho_1s = [HermitianSemidefinite(8) for i in 1:nps]

    rho_next = sum(kron(X_list_2[i], rho_1s[i]) for i in 1:nps) +
    sum((exchange_12 * kron(X_list_2[i], rho_1s[i]) * exchange_12) for i in 1:nps) +
    sum((exchange_13 * kron(X_list_2[i], rho_1s[i]) * exchange_13) for i in 1:nps) +
    sum((exchange_14 * kron(X_list_2[i], rho_1s[i]) * exchange_14) for i in 1:nps)

    return rho_next, rho_1s
  end
end

function test_part_2_train()
  nps = 100
  for j in 1:20
    # X_list_2_1 = [randState(4, 1) for index2 in 1:nps]
    X_list_2_2 = [randState(2) for index2 in 1:nps]
    # push!(X_list_2_2, I(2)/2)
    for i in 1:20
      rho_next, rhos_list_2 = test_part_2(X_list_2_2, nps, 2)
      optval = train(rho_next)

      X_list_1_2 = checkout_rho_list_3(rhos_list_2, nps)
      
      rho_next, rhos_list_2 = test_part_2(X_list_1_2, nps, 1)
      optval = train(rho_next)

      X_list_2_2 = checkout_rho_list_1(rhos_list_2, nps)
    end
    println("one round ", j)
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

function checkout_rho_list_1(rho_list, nps)
  global error_count = 0
  result_list = Array{Matrix{Complex{Float64}},1}(UndefInitializer(), nps);
  tr_list = Array{Float64,1}(undef, nps);
  for current_index in 1:nps
    current_matrix = rho_list[current_index].value

    tr_list[current_index] = tr(current_matrix)
    evs = eigvals(current_matrix)
    revs = [real(it) for it in evs]
    ievs = [imag(it) for it in evs]

    if ievs'*ievs/(revs'*revs) > 1e-3 || minimum(revs) < 0 
      error_count += 1
      result_list[current_index] = decompose_and_reconstruct_positive(current_matrix, nps)
    else 
      result_list[current_index] = current_matrix / (nps * tr(current_matrix))
    end
  end

  println("sum ", nps, "error count ", error_count)
  return result_list
end

function checkout_rho_list_3(rho_list, nps)
  global error_count = 0
  result_list = Array{Matrix{Complex{Float64}},1}(UndefInitializer(), nps);
  tr_list = Array{Float64,1}(undef, nps);
  for current_index in 1:nps
    current_matrix = rho_list[current_index].value

    tr_list[current_index] = tr(current_matrix)
    evs = eigvals(current_matrix)
    revs = [real(it) for it in evs]
    ievs = [imag(it) for it in evs]

    if ievs'*ievs/(revs'*revs) > 1e-3 || minimum(revs) < 0 
      error_count += 1
      result_list[current_index] = decompose_and_reconstruct_positive(current_matrix, 1)
    else 
      result_list[current_index] = current_matrix / tr(current_matrix)
    end
  end

  println("sum ", nps, "error count ", error_count)
  return result_list
end

test_part_2_train()