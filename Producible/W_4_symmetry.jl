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

function decompose_and_reconstruct_positive(A::AbstractMatrix, trace_denominator)
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

  new_A = A_reconstructed/(trace_denominator * tr(A_reconstructed))
  if any(isnan.(new_A)) || any(isinf.(new_A))
    return randState(size(A_reconstructed, 1), trace_denominator)
  else
    return new_A
  end
end

struct MultiState
  mat::Matrix{Complex{Float64}}
  dims::Array{Int,1}
end;

function EncodeNumber(L::Int,dim_list::Array{Int,1})
	n = length(dd);
	code_result = Array{Int,1}(undef,n);
	number = L-1;
	for k in reverse(1:n)
		code_result[k] = mod(number, dim_list[k])+1;
		number = div(number, dim_list[k]);
	end;
	return code_result;
end;

function DecodeNumber(code_result::Array{Int,1},dim_list::Array{Int,1})
	dim_numbers = AccumSizes(dim_list);
	n = length(code_result);
	number = code_result[n]-1;
	for k in reverse(1:(n-1))
		number += dim_numbers[k+1] * (code_result[k]-1);
	end;
	return number+1;
end;
  
rho = MultiState(orho, [2, 2, 2, 2])

function randState(dim, nps)
  V=randn(Complex{Float64},dim);
  V/=norm(V);
  V = V*conj(transpose(V))
  return V / nps
end

function part_3_rho_next(X_list1, X_list2, nps, train_part)
  # 2|1|1
  if train_part == 1
    # X_list1 1    X_list2 1
    rho_1s = [HermitianSemidefinite(4) for i in 1:nps]

    rho_next = sum(kron(kron(rho_1s[i], X_list1[i]), X_list2[i]) for i in 1:nps) + 
    sum((exchange_23 * kron(kron(rho_1s[i], X_list1[i]), X_list2[i]) * exchange_23) for i in 1:nps) + 
    sum((exchange_24 * kron(kron(rho_1s[i], X_list1[i]), X_list2[i]) * exchange_24) for i in 1:nps) +
    sum(kron(kron(X_list1[i], rho_1s[i]), X_list2[i]) for i in 1:nps) +
    sum((exchange_34 * kron(kron(X_list1[i], rho_1s[i]), X_list2[i]) * exchange_34) for i in 1:nps) +
    sum(kron(kron(X_list1[i], X_list2[i]), rho_1s[i]) for i in 1:nps)

    return rho_next, rho_1s
  elseif train_part == 2
    # X_list1 2    X_list2 1 part 3
    rho_1s = [HermitianSemidefinite(2) for i in 1:nps]

    rho_next = sum(kron(kron(X_list1[i], rho_1s[i]), X_list2[i]) for i in 1:nps) + 
    sum((exchange_23 * kron(kron(X_list1[i], rho_1s[i]), X_list2[i]) * exchange_23) for i in 1:nps) + 
    sum((exchange_24 * kron(kron(X_list1[i], rho_1s[i]), X_list2[i]) * exchange_24) for i in 1:nps) +
    sum(kron(kron(rho_1s[i], X_list1[i]), X_list2[i]) for i in 1:nps) +
    sum((exchange_34 * kron(kron(rho_1s[i], X_list1[i]), X_list2[i]) * exchange_34) for i in 1:nps) +
    sum(kron(kron(rho_1s[i], X_list2[i]), X_list1[i]) for i in 1:nps)

    return rho_next, rho_1s
  else
    # X_list1 2 X_list2 1 part 2
    rho_1s = [HermitianSemidefinite(2) for i in 1:nps]

    rho_next = sum(kron(kron(X_list1[i], X_list2[i]), rho_1s[i]) for i in 1:nps) + 
    sum((exchange_23 * kron(kron(X_list1[i], X_list2[i]), rho_1s[i]) * exchange_23) for i in 1:nps) + 
    sum((exchange_24 * kron(kron(X_list1[i], X_list2[i]), rho_1s[i]) * exchange_24) for i in 1:nps) +
    sum(kron(kron(X_list2[i], X_list1[i]), rho_1s[i]) for i in 1:nps) +
    sum((exchange_34 * kron(kron(X_list2[i] , X_list1[i]), rho_1s[i]) * exchange_34) for i in 1:nps) +
    sum(kron(kron(X_list2[i], rho_1s[i]), X_list1[i]) for i in 1:nps)

    return rho_next, rho_1s
  end
end

function prod_2_rho_next(X_list, nps, train_part)
  rhoAs = [HermitianSemidefinite(4) for i in 1:nps]
  # rhoBs = [HermitianSemidefinite(4) for i in 1:nps]
  # rhoCs = [HermitianSemidefinite(4) for i in 1:nps]
  if train_part == 1
    rho_next = sum(kron((exchange_2_qubit * rhoAs[i] * exchange_2_qubit + rhoAs[i]) / 2, (exchange_2_qubit * X_list[i] * exchange_2_qubit + X_list[i]) / 2) for i in 1:nps) + 
    sum((exchange_23 * kron((exchange_2_qubit * rhoAs[i] * exchange_2_qubit + rhoAs[i]) / 2, (exchange_2_qubit * X_list[i] * exchange_2_qubit + X_list[i]) / 2) * exchange_23) for i in 1:nps) + 
    sum((exchange_24 * kron((exchange_2_qubit * rhoAs[i] * exchange_2_qubit + rhoAs[i]) / 2, (exchange_2_qubit * X_list[i] * exchange_2_qubit + X_list[i]) / 2) * exchange_24) for i in 1:nps)
    return rho_next, rhoAs
  else
    rho_next = sum(kron((exchange_2_qubit * X_list[i] * exchange_2_qubit + X_list[i]) / 2, (exchange_2_qubit * rhoAs[i] * exchange_2_qubit + rhoAs[i]) / 2) for i in 1:nps) + 
    sum((exchange_23 * kron((exchange_2_qubit * X_list[i] * exchange_2_qubit + X_list[i]) / 2, (exchange_2_qubit * rhoAs[i] * exchange_2_qubit + rhoAs[i]) / 2) * exchange_23) for i in 1:nps) + 
    sum((exchange_24 * kron((exchange_2_qubit * X_list[i] * exchange_2_qubit + X_list[i]) / 2, (exchange_2_qubit * rhoAs[i] * exchange_2_qubit + rhoAs[i]) / 2) * exchange_24) for i in 1:nps)
    return rho_next, rhoAs
  end
end

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

function part_2_rho_next(X_list, X_list_2, nps, train_part)
  rhoAs = [HermitianSemidefinite(4) for i in 1:nps]

  if train_part == 1
    rho_1s = [HermitianSemidefinite(2) for i in 1:nps]

    rho_next = sum(kron(rhoAs[i], X_list[i]) for i in 1:nps) + 
    sum((exchange_23 * kron(rhoAs[i], X_list[i]) * exchange_23) for i in 1:nps) + 
    sum((exchange_24 * kron(rhoAs[i], X_list[i]) * exchange_24) for i in 1:nps) +
    sum(kron(rho_1s[i], X_list_2[i]) for i in 1:nps) +
    sum((exchange_12 * kron(rho_1s[i], X_list_2[i]) * exchange_12) for i in 1:nps) +
    sum((exchange_13 * kron(rho_1s[i], X_list_2[i]) * exchange_13) for i in 1:nps) +
    sum((exchange_14 * kron(rho_1s[i], X_list_2[i]) * exchange_14) for i in 1:nps)

    return rho_next, rhoAs, rho_1s
  else
    rho_1s = [HermitianSemidefinite(8) for i in 1:nps]

    rho_next = sum(kron(X_list[i], rhoAs[i]) for i in 1:nps) + 
    sum((exchange_23 * kron(X_list[i], rhoAs[i]) * exchange_23) for i in 1:nps) + 
    sum((exchange_24 * kron(X_list[i], rhoAs[i]) * exchange_24) for i in 1:nps) +
    sum(kron(X_list_2[i], rho_1s[i]) for i in 1:nps) +
    sum((exchange_12 * kron(X_list_2[i], rho_1s[i]) * exchange_12) for i in 1:nps) +
    sum((exchange_13 * kron(X_list_2[i], rho_1s[i]) * exchange_13) for i in 1:nps) +
    sum((exchange_14 * kron(X_list_2[i], rho_1s[i]) * exchange_14) for i in 1:nps)

    return rho_next, rhoAs, rho_1s
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

function test_part_2_train()
  nps = 100
  for j in 1:20
    # X_list_2_1 = [randState(4, 1) for index2 in 1:nps]
    X_list_2_2 = [randState(8, 1) for index2 in 1:nps]
    for i in 1:3
      rho_next, rhos_list_2 = test_part_2(X_list_2_2, nps, 1)
      optval = train(rho_next)

      X_list_1_2 = checkout_rho_list(rhos_list_2, nps)
      
      rho_next, rhos_list_2 = test_part_2(X_list_1_2, nps, 2)
      optval = train(rho_next)

      X_list_2_2 = checkout_rho_list(rhos_list_2, nps)
    end
    println("one round ", j)
  end
end

function part_3_train()
  nps = 100
  for j in 1:20
    X_list_2 = [randState(2, 1) for index2 in 1:nps]
    X_list_3 = [randState(2, nps) for index2 in 1:nps]
    for i in 1:30
      rho_next, rho_list = part_3_rho_next(X_list_2, X_list_3, nps, 1)
      println("Part 2")
      optval = train(rho_next)
      X_list_1 = checkout_rho_list_nochange(rho_list, nps)

      rho_next, rho_list = part_3_rho_next(X_list_1, X_list_3, nps, 2)
      println("Part 1 A")
      optval = train(rho_next)
      X_list_2 = checkout_rho_list_nochange(rho_list, nps)
      
      rho_next, rho_list = part_3_rho_next(X_list_1, X_list_2, nps, 3)
      println("Part 1 B")
      optval = train(rho_next)
      X_list_3 = checkout_rho_list(rho_list, nps, 1)
    end
    println("one round ", j)
  end
end

function prod_2_train()
  nps = 100
  for j in 1:20
    X_list_2 = [randState(4, nps) for index2 in 1:nps]
    for i in 1:30
      rho_next, rho_list = prod_2_rho_next(X_list_2, nps, 1)
      optval = train(rho_next)
      X_list_1 = checkout_rho_list(rho_list, nps, nps)
      # handle_X_list_1_2([current_rho.value for current_rho in rho_list], X_list_2, optval, nps)

      rho_next, rho_list = prod_2_rho_next(X_list_1, nps, 2)
      optval = train(rho_next)
      X_list_2 = checkout_rho_list(rho_list, nps, nps)
      # handle_X_list_1_2(X_list_1, [current_rho.value for current_rho in rho_list], optval, nps)
    end
    println("one round ", j)
  end
end

function part_2_train()
  nps = 100
  for j in 1:20
    X_list_1_1 = [randState(4, 2 * nps) for index2 in 1:nps]
    X_list_1_2 = [randState(2, 2 * nps) for index2 in 1:nps]
    for i in 1:20
      rho_next, rhos_list_1, rhos_list_2 = part_2_rho_next(X_list_1_1, X_list_1_2, nps, 2)
      optval = train(rho_next)

      X_list_2_1 = checkout_rho_list(rhos_list_1, nps, 2 * nps)
      X_list_2_2 = checkout_rho_list(rhos_list_2, nps, 1)
      
      rho_next, rhos_list_1, rhos_list_2 = part_2_rho_next(X_list_2_1, X_list_2_2, nps, 1)
      optval = train(rho_next)

      X_list_1_1 = checkout_rho_list(rhos_list_1, nps, 2 * nps)
      X_list_1_2 = checkout_rho_list(rhos_list_2, nps, 2 * nps)
    end
    println("one round ", j)
  end
end

function handle_X_list_1_2(X_list_1, X_list_2, t, nps)
  next_result = sum(kron((exchange_2_qubit * X_list_1[i] * exchange_2_qubit + X_list_1[i]) / 2, (exchange_2_qubit * X_list_2[i] * exchange_2_qubit + X_list_2[i]) / 2) for i in 1:nps) + 
    sum((exchange_23 * kron((exchange_2_qubit * X_list_1[i] * exchange_2_qubit + X_list_1[i]) / 2, (exchange_2_qubit * X_list_2[i] * exchange_2_qubit + X_list_2[i]) / 2) * exchange_23) for i in 1:nps) + 
    sum((exchange_24 * kron((exchange_2_qubit * X_list_1[i] * exchange_2_qubit + X_list_1[i]) / 2, (exchange_2_qubit * X_list_2[i] * exchange_2_qubit + X_list_2[i]) / 2) * exchange_24) for i in 1:nps)

  II= zeros(Complex{Float64},prod(rho.dims),prod(rho.dims))+I;
  the_rho = t*rho.mat+(1.0-t)/prod(rho.dims)*II

  C = the_rho - next_result
  println("distance ", norm(C))
end

function checkout_rho_list_nochange(rho_list, nps)
  global error_count = 0
  result_list = Array{Matrix{Complex{Float64}},1}(UndefInitializer(), nps);

  for current_index in 1:nps
    current_matrix = rho_list[current_index].value
    result_list[current_index] = current_matrix

    evs = eigvals(current_matrix)
    revs = [real(it) for it in evs]
    ievs = [imag(it) for it in evs]

    if ievs'*ievs/(revs'*revs) > 1e-3 || minimum(revs) < 0 
      error_count += 1
    end
  end

  println("sum ", nps, "error count ", error_count)
  return result_list
end


function checkout_rho_list(rho_list, nps, trace_denominator)
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
      result_list[current_index] = decompose_and_reconstruct_positive(current_matrix, trace_denominator)
    else 
      result_list[current_index] = current_matrix / (trace_denominator * tr(current_matrix))
    end
  end

  println("sum ", nps, "error count ", error_count)
  return result_list
end

# part_3_train()
# prod_2_train()
part_2_train()
# test_part_2_train()
