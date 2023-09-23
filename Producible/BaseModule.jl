module BaseModule
  struct MultiState
    mat::Matrix{Complex{Float64}}
    dims::Array{Int,1}
  end;
  export MultiState

  function randState(dim)
    V=randn(Complex{Float64},dim);
    V/=norm(V);
    return V*conj(transpose(V))
  end;
  export randState
end
