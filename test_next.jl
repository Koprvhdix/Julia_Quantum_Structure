using LinearAlgebra
using Mosek;
using MosekTools; 
using SCS; 
using Convex;
using MathOptInterface;

exchange = [1. 0. 0. 0. 0. 0. 0. 0. ;
      0. 0. 1. 0. 0. 0. 0. 0. ;
      0. 1. 0. 0. 0. 0. 0. 0. ;
      0. 0. 0. 1. 0. 0. 0. 0. ;
      0. 0. 0. 0. 1. 0. 0. 0. ;
      0. 0. 0. 0. 0. 0. 1. 0. ;
      0. 0. 0. 0. 0. 1. 0. 0. ;
      0. 0. 0. 0. 0. 0. 0. 1. ]

orho = [0.   0.   0.   0.   0.   0.   0.   0.  ;
      0.   1/3  1/3  0.   1/3  0.   0.   0.  ;
      0.   1/3  1/3  0.   1/3  0.   0.   0.  ;
      0.   0.   0.   0.   0.   0.   0.   0.  ;
      0.   1/3  1/3  0.   1/3  0.   0.   0.  ;
      0.   0.   0.   0.   0.   0.   0.   0.  ;
      0.   0.   0.   0.   0.   0.   0.   0.  ;
      0.   0.   0.   0.   0.   0.   0.   0.  ]

# function RandomBlochPolytope(d::Int=2, L::Int=100)
#   QPolytope= Array{Array{Complex{Float64},2},1}(undef,L);
#   for k in 1:L
#       V=randn(Complex{Float64},d);
#       V/=norm(V);
#       QPolytope[k]=V*conj(transpose(V));
#   end
#   return QPolytope;
# end

function RandomBlochPolytope(d::Int=2, L::Int=100)
  QPolytope= Array{Array{Complex{Float64},2},1}(undef,L);
  for k in 1:L
      QPolytope[k]=randState(d);
  end
  return QPolytope;
end

function randState(dim)
  re = randn(dim)+ [i*im for i in randn(dim)]
  # println(randn(Complex{Float64},d))
  println(re)
  re*re'/(re'*re)
end

# function RandomBlochPolytope(d::Int=2, L::Int=100)
#   QPolytope= Array{Array{Float64,2},1}(undef,L);
#   for k in 1:L
#       V=randn(Float64,d);
#       V/=norm(V);
#       QPolytope[k]=V*conj(transpose(V));
#   end
#   return QPolytope;
# end

function AccumSizes(dims::Array{Int,1})
  n= length(dims);
sizes=Array{Int,1}(undef,n);
s=1;
  for k in reverse(1:n)
  s*= dims[k];
  sizes[k]=s;
  end;
return sizes;
end;

struct MultiState
	mat::Matrix{Complex{Float64}}
	dims::Array{Int,1}
end;

function LocalIndex(L::Int,dd::Array{Int,1})
	n=length(dd);
	sizes= AccumSizes(dd);
	kk=Array{Int,1}(undef,n);
	s=L-1;
	for k in reverse(1:n)
		kk[k]= mod(s,dd[k])+1;
		s= div(s,dd[k]);
	end;
	#kA=div(L-1,dd[2])+1;
	#kB=mod(L-1,dd[2])+1;
	return kk;
end;

function LinearIndex(kk::Array{Int,1},dd::Array{Int,1})
	sizes= AccumSizes(dd);
	n= length(kk);
	s=kk[n]-1;
	for k in reverse(1:(n-1))
		s+= sizes[k+1]*(kk[k]-1);
	end;
	return s+1;
end;

rho = MultiState(orho, [2, 2, 2])

QPolytope = RandomBlochPolytope()

t=Variable(1,Positive());
N=length(QPolytope);
# X1=Array{Any,1}(undef,N);
# X2=Array{Any,1}(undef,N);
# X3=Array{Any,1}(undef,N);
# for k in 1:N
#   X1[k]= HermitianSemidefinite(4,4);
#   X2[k]= HermitianSemidefinite(4,4);
#   X3[k]= HermitianSemidefinite(4,4);
# end;
X1 = [HermitianSemidefinite(4,4) for i in 1:N]
X2 = [HermitianSemidefinite(4,4) for i in 1:N]
X3 = [HermitianSemidefinite(4,4) for i in 1:N]
global S=zeros(Complex{Float64},prod(rho.dims),prod(rho.dims));
for k in 1:N
  global S+= kron(QPolytope[k],X1[k]);
  global S+= kron(X3[k],QPolytope[k]);
  
  # C= zeros(Complex{Float64},8,8);
  # for l in 1:8
  # kk= LocalIndex(l,[2,2,2]); tmp=kk[1]; kk[1]=kk[2]; kk[2]= tmp;
  # p=LinearIndex(kk,[2,2,2]);
  #       C[p,l]=1.0;
  # end;
  # global S += C*kron(QPolytope[k],X2[k])*C; 
  global S += exchange*kron(X2[k], QPolytope[k])*exchange; 
end;

II= zeros(Complex{Float64},prod(rho.dims),prod(rho.dims))+I;
problem= maximize(t);
problem.constraints+= ((t*rho.mat+(1.0-t)/prod(rho.dims)*II) == S);
solve!(problem, Mosek.Optimizer(LOG=0));
print(problem.status,"\n");
print("robustness to white noise: ",problem.optval, "\n"); 
