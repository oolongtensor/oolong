include("StructEquality.jl")
include("ImportACounter.jl")

b_counter = CounterX(1)

println(a_counter.id)
println(b_counter.id)
println(a_counter == b_counter) # False
