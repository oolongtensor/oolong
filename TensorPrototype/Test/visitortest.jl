include("../TreeVisitor/TreeVisitor.jl")

using Test

V3 = VectorSpace(3)
V2 = VectorSpace(2)

A = VariableTensor(V3, V2)
Z = ZeroTensor(V3, V2)

@testset "Visitors" begin
    @testset "ZeroRemoval" begin
        @test traversal( A + Z, removezero!, false).children == [A]
    end
end