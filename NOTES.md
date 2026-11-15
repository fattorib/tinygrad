On CPU, it seems that there are two types of axes that we currently aren't using. This is a good way to understand more about Tinygrad's compiler and codegen in a (hopefully simple) way:

- OpenMP should be able to be used for a single GLOBAL axis
- We should be able to support AVX2 and AVX512 (float_8 and float_16) ops as UPCAST axes

At the end of this, we should see:
1. Speedups on existing BLAS-type ops (thinking elementwise, contractions, etc)
2. All tests behaviour _should_ be unchanged...?



Upcast:
    - This is the process of vectorization, we should in theory be able to expand the vectorization in `ClangRenderer` to also use a float_8 type
    - The first step where this changes is in the UPCAST
        Seems like this happens in hand_coded_optimizations (tinygrad/codegen/opt/heuristic.py) 
            - There are multiple stages in which this 4element float is applied, so we actually need to hit them all
                or add additional support for a larger unroll. Majority is just checking dimension divisibility rules 
                (for axis, upcast_amount in itertools.product(k.upcastable_dims(), ([128] if not len(upcasted_axis) else []) if is_dsp else [3,4]):)
                This controls which we check for upcast and applies the first we see in the list

            And then we need to modify the stack / render too
            - Doesn't seem too bad

            - Make a custom renderer for float_8, update heuristic to condition on this, if possible
            - Change the PM renderer to use this 

    The optimization we are applying is OptOps.SPLIT