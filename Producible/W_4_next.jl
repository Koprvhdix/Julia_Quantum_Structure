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

function decompose_and_reconstruct_positive(A::AbstractMatrix)
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

  new_A = A_reconstructed/tr(A_reconstructed)
  if any(isnan.(new_A)) || any(isinf.(new_A))
    return randState(size(A_reconstructed, 1))
  else
    return new_A
  end
end

function nlize(rho)
  evs = eigvals(rho)
  revs = [real(it) for it in evs]
  ievs = [imag(it) for it in evs]
  if ievs'*ievs/(revs'*revs)> 1e-3 || minimum(revs) < 0
    randState(size(rho, 1), 10)
    # decompose_and_reconstruct_positive(rho)
  else rho/(10 * tr(rho))
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

function randState(dim)
  V=randn(Complex{Float64},dim);
  V/=norm(V);
  return V*conj(transpose(V))
end

# function randState(dim, nps)
#   V=randn(Complex{Float64},dim);
#   V/=norm(V);
#   V = V*conj(transpose(V))
#   return V / nps
# end

function part_3_rho_next(X_list1, X_list2, nps, train_part)
  # 2|1|1
  if train_part == 1
    # X_list1 1    X_list2 1
    rho_1s = [HermitianSemidefinite(4) for i in 1:nps]
    rho_2s = [HermitianSemidefinite(4) for i in 1:nps]
    rho_3s = [HermitianSemidefinite(4) for i in 1:nps]
    rho_4s = [HermitianSemidefinite(4) for i in 1:nps]
    rho_5s = [HermitianSemidefinite(4) for i in 1:nps]
    rho_6s = [HermitianSemidefinite(4) for i in 1:nps]
    p_s = [[Variable(1,Positive()) for i in 1:nps] for j in 1:6]

    rho_next = sum(p_s[1][i] * kron(kron(rho_1s[i], X_list1[1][i]), X_list2[1][i]) for i in 1:nps) + 
    sum(p_s[2][i] * (exchange_23 * kron(kron(rho_2s[i], X_list1[2][i]), X_list2[2][i]) * exchange_23) for i in 1:nps) + 
    sum(p_s[3][i] * (exchange_24 * kron(kron(rho_3s[i], X_list1[3][i]), X_list2[3][i]) * exchange_24) for i in 1:nps) +
    sum(p_s[4][i] * kron(kron(X_list1[4][i], rho_4s[i]), X_list2[4][i]) for i in 1:nps) +
    sum(p_s[5][i] * (exchange_34 * kron(kron(X_list1[5][i], rho_5s[i]), X_list2[5][i]) * exchange_34) for i in 1:nps) +
    sum(p_s[6][i] * kron(kron(X_list1[6][i], X_list2[6][i]), rho_6s[i]) for i in 1:nps)

    return rho_next, [rho_1s, rho_2s, rho_3s, rho_4s, rho_5s, rho_6s], p_s
  elseif train_part == 2
    # X_list1 2    X_list2 1 part 3
    rho_1s = [HermitianSemidefinite(2) for i in 1:nps]
    rho_2s = [HermitianSemidefinite(2) for i in 1:nps]
    rho_3s = [HermitianSemidefinite(2) for i in 1:nps]
    rho_4s = [HermitianSemidefinite(2) for i in 1:nps]
    rho_5s = [HermitianSemidefinite(2) for i in 1:nps]
    rho_6s = [HermitianSemidefinite(2) for i in 1:nps]
    p_s = [[Variable(1,Positive()) for i in 1:nps] for j in 1:6]

    rho_next = sum(p_s[1][i] * kron(kron(X_list1[1][i], rho_1s[i]), X_list2[1][i]) for i in 1:nps) + 
    sum(p_s[2][i] * (exchange_23 * kron(kron(X_list1[2][i], rho_2s[i]), X_list2[2][i]) * exchange_23) for i in 1:nps) + 
    sum(p_s[3][i] * (exchange_24 * kron(kron(X_list1[3][i], rho_3s[i]), X_list2[3][i]) * exchange_24) for i in 1:nps) +
    sum(p_s[4][i] * kron(kron(rho_4s[i], X_list1[4][i]), X_list2[4][i]) for i in 1:nps) +
    sum(p_s[5][i] * (exchange_34 * kron(kron(rho_5s[i], X_list1[5][i]), X_list2[5][i]) * exchange_34) for i in 1:nps) +
    sum(p_s[6][i] * kron(kron(rho_6s[i], X_list2[6][i]), X_list1[6][i]) for i in 1:nps)

    return rho_next, [rho_1s, rho_2s, rho_3s, rho_4s, rho_5s, rho_6s], p_s
  else
    # X_list1 2    X_list2 1 part 2
    rho_1s = [HermitianSemidefinite(2) for i in 1:nps]
    rho_2s = [HermitianSemidefinite(2) for i in 1:nps]
    rho_3s = [HermitianSemidefinite(2) for i in 1:nps]
    rho_4s = [HermitianSemidefinite(2) for i in 1:nps]
    rho_5s = [HermitianSemidefinite(2) for i in 1:nps]
    rho_6s = [HermitianSemidefinite(2) for i in 1:nps]
    p_s = [[Variable(1,Positive()) for i in 1:nps] for j in 1:6]

    rho_next = sum(p_s[1][i] * kron(kron(X_list1[1][i], X_list2[1][i]), rho_1s[i]) for i in 1:nps) + 
    sum(p_s[2][i] * (exchange_23 * kron(kron(X_list1[2][i], X_list2[2][i]), rho_2s[i]) * exchange_23) for i in 1:nps) + 
    sum(p_s[3][i] * (exchange_24 * kron(kron(X_list1[3][i], X_list2[3][i]), rho_3s[i]) * exchange_24) for i in 1:nps) +
    sum(p_s[4][i] * kron(kron(X_list2[4][i], X_list1[4][i]), rho_4s[i]) for i in 1:nps) +
    sum(p_s[5][i] * (exchange_34 * kron(kron(X_list2[5][i] , X_list1[5][i]), rho_5s[i]) * exchange_34) for i in 1:nps) +
    sum(p_s[6][i] * kron(kron(X_list2[6][i], rho_6s[i]), X_list1[6][i]) for i in 1:nps)

    return rho_next, [rho_1s, rho_2s, rho_3s, rho_4s, rho_5s, rho_6s], p_s
  end
end

function prod_2_rho_next(X_list, nps, train_part)
  rhoAs = [HermitianSemidefinite(4) for i in 1:nps]
  rhoBs = [HermitianSemidefinite(4) for i in 1:nps]
  rhoCs = [HermitianSemidefinite(4) for i in 1:nps]
  if train_part == 1
    rho_next = sum(kron(rhoAs[i], X_list[1][i]) for i in 1:nps) + sum((exchange_23 * kron(rhoBs[i], X_list[2][i]) * exchange_23) for i in 1:nps) +  sum((exchange_24 * kron(rhoCs[i], X_list[3][i]) * exchange_24) for i in 1:nps)
    return rho_next, [rhoAs, rhoBs, rhoCs]
  else
    rho_next = sum(kron(X_list[1][i], rhoAs[i]) for i in 1:nps) + sum((exchange_23 * kron(X_list[2][i], rhoBs[i]) * exchange_23) for i in 1:nps) +  sum((exchange_24 * kron(X_list[3][i], rhoCs[i]) * exchange_24) for i in 1:nps)
    return rho_next, [rhoAs, rhoBs, rhoCs]
  end
end

function part_2_rho_next(X_list, X_list_2, nps, train_part)
  rhoAs = [HermitianSemidefinite(4) for i in 1:nps]
  rhoBs = [HermitianSemidefinite(4) for i in 1:nps]
  rhoCs = [HermitianSemidefinite(4) for i in 1:nps]
  if train_part == 1
    rho_1s = [HermitianSemidefinite(2) for i in 1:nps]
    rho_2s = [HermitianSemidefinite(2) for i in 1:nps]
    rho_3s = [HermitianSemidefinite(2) for i in 1:nps]
    rho_4s = [HermitianSemidefinite(2) for i in 1:nps]
    rho_next = sum(kron(rhoAs[i], X_list[1][i]) for i in 1:nps) + 
    sum((exchange_23 * kron(rhoBs[i], X_list[2][i]) * exchange_23) for i in 1:nps) + 
    sum((exchange_24 * kron(rhoCs[i], X_list[3][i]) * exchange_24) for i in 1:nps) +
    sum(kron(rho_1s[i], X_list_2[1][i]) for i in 1:nps) +
    sum((exchange_12 * kron(rho_2s[i], X_list_2[2][i]) * exchange_12) for i in 1:nps) +
    sum((exchange_13 * kron(rho_3s[i], X_list_2[3][i]) * exchange_13) for i in 1:nps) +
    sum((exchange_14 * kron(rho_4s[i], X_list_2[4][i]) * exchange_14) for i in 1:nps)
    return rho_next, [rhoAs, rhoBs, rhoCs], [rho_1s, rho_2s, rho_3s, rho_4s]
  else
    rho_1s = [HermitianSemidefinite(8) for i in 1:nps]
    rho_2s = [HermitianSemidefinite(8) for i in 1:nps]
    rho_3s = [HermitianSemidefinite(8) for i in 1:nps]
    rho_4s = [HermitianSemidefinite(8) for i in 1:nps]
    rho_next = sum(kron(X_list[1][i], rhoAs[i]) for i in 1:nps) + 
    sum((exchange_23 * kron(X_list[2][i], rhoBs[i]) * exchange_23) for i in 1:nps) + 
    sum((exchange_24 * kron(X_list[3][i], rhoCs[i]) * exchange_24) for i in 1:nps) +
    sum(kron(X_list_2[1][i], rho_1s[i]) for i in 1:nps) +
    sum((exchange_12 * kron(X_list_2[2][i], rho_2s[i]) * exchange_12) for i in 1:nps) +
    sum((exchange_13 * kron(X_list_2[3][i], rho_3s[i]) * exchange_13) for i in 1:nps) +
    sum((exchange_14 * kron(X_list_2[4][i], rho_4s[i]) * exchange_14) for i in 1:nps)
    return rho_next, [rhoAs, rhoBs, rhoCs], [rho_1s, rho_2s, rho_3s, rho_4s]
  end
end

function train(rho_next, p_s)
  t = Variable(1,Positive());

  II= zeros(Complex{Float64},prod(rho.dims),prod(rho.dims))+I;
  problem= maximize(t);
  problem.constraints += ((t*rho.mat+(1.0-t)/prod(rho.dims)*II) == rho_next);
  problem.constraints += (sum(sum(p_s[i]) for i in 1:6) == 1);


  solve!(problem, Mosek.Optimizer, silent_solver=true)
  println(problem.optval)
  problem.optval
end

function part_3_train()
  nps = 100
  for j in 1:20
    X_list_2 = [[randState(2) for index2 in 1:nps] for index1 in 1:6]
    X_list_3 = [[randState(2) for index2 in 1:nps] for index1 in 1:6]
    for i in 1:30
      rho_next, rhos_list, p_s = part_3_rho_next(X_list_2, X_list_3, nps, 1)
      optval = train(rho_next, p_s)
      X_list_1 = [[current_rho.value for current_rho in rhos] for rhos in rhos_list]

      rho_next, rhos_list, p_s = part_3_rho_next(X_list_1, X_list_3, nps, 2)
      optval = train(rho_next, p_s)
      X_list_2 = [[current_rho.value for current_rho in rhos] for rhos in rhos_list]
      
      rho_next, rhos_list, p_s = part_3_rho_next(X_list_1, X_list_2, nps, 3)
      optval = train(rho_next, p_s)
      X_list_3 = [[current_rho.value for current_rho in rhos] for rhos in rhos_list]
    end
    println("one round ", j)
  end
end

function prod_2_train()
  nps = 100
  for j in 1:20
    X_list_2 = [[randState(4) for index2 in 1:nps] for index1 in 1:3]
    for i in 1:30
      rho_next, rhos_list = prod_2_rho_next(X_list_2, nps, 1)
      optval = train(rho_next)
      X_list_1 = [[nlize(current_rho.value) for current_rho in rhos] for rhos in rhos_list]

      rho_next, rhos_list = prod_2_rho_next(X_list_1, nps, 2)
      optval = train(rho_next)
      X_list_2 = [[nlize(current_rho.value) for current_rho in rhos] for rhos in rhos_list]
    end
    println("one round ", j)
  end
end

function part_2_train()
  nps = 100
  for j in 1:20
    X_list_2_1 = [[randState(4, nps) for index2 in 1:nps] for index1 in 1:3]
    X_list_2_2 = [[randState(8, nps) for index2 in 1:nps] for index1 in 1:4]
    for i in 1:30
      rho_next, rhos_list_1, rhos_list_2 = part_2_rho_next(X_list_2_1, X_list_2_2, nps, 1)
      optval = train(rho_next)
      X_list_1_1 = [[nlize(current_rho.value) for current_rho in rhos] for rhos in rhos_list_1]
      X_list_1_2 = [[nlize(current_rho.value) for current_rho in rhos] for rhos in rhos_list_2]
      
      rho_next, rhos_list_1, rhos_list_2 = part_2_rho_next(X_list_1_1, X_list_1_2, nps, 2)
      optval = train(rho_next)
      X_list_2_1 = [[nlize(current_rho.value) for current_rho in rhos] for rhos in rhos_list_1]
      X_list_2_2 = [[nlize(current_rho.value) for current_rho in rhos] for rhos in rhos_list_2]
    end
    println("one round ", j)
  end
end

part_3_train()
# prod_2_train()
# part_2_train()
