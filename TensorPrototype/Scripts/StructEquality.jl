struct ImmutableX
    x
end

mutable struct MutableX
    x
end

a = ImmutableX(1)
b = ImmutableX(1)
a_mutable = MutableX(1)
b_mutable = MutableX(1)
a_array = ImmutableX([1])
b_array = ImmutableX([1])

println(a === b) # True
println(a_mutable === b_mutable) #False
println(a_array === b_array) # False
