# Python-Julia Compatibility

## Arrays
Python arrays are indexed from 0, whereas Julia-arrays are indexed from 1. This
tends to be easy to get around by adding/substracting 1 to indices, but needs
to be remembered.

Julia arrays are column-major, whereas Python-arrays are row-major. This might
affect performance.

## Objects
PyCall tends to convert Python objects to Julia, which sometimes loses
information. For example Python's namedtuple is converted to tuple in Julia.
There are two ways around this: first one can just avoid Python-Julia
conversions - which is what Oolong does. Secondly, [pycall](https://github.com/JuliaPy/PyCall.jl#calling-python)-function
can be used to return Python-objects.

## Initialisation
[Calling PyCall from modules](https://github.com/JuliaPy/PyCall.jl#using-pycall-from-julia-modules) is not quite as straightforward as one might think. The link provides a good quide.
