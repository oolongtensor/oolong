include("../TreeVisitor/ZeroRemover.jl")

using Test

V3 = VectorSpace(3)
V2 = VectorSpace(2)

A = VariableTensor(V3, V2)
Z = ZeroTensor(V3, V2)

@testset "Visitors" begin
    @testset "ZeroRemoval" begin
        @test removezero!(A + Z) == A
        @test removezero!((A + Z)[1,2]) == A[1,2]
    end
end