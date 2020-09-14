using TensorDSL
using BenchmarkTools
using PyCall

dim = 60
V = VectorSpace(dim)
A = VariableTensor("A", V, V', RnSpace(6))
x = FreeIndex(V, "x")
knl = Kernel(((4*A)[x, x']⊗transpose(A))[1,2, x', x])

val = permutedims(rand(Float64, (dim, dim, 6)), [i for i in reverse(collect(1:length(A.shape)))])

b = @benchmarkable execute($knl, "A"=>$val)
run(b)