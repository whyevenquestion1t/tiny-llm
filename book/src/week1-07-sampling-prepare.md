# Sampling and Preparing for Week 2

## Task 1: Sampling

## Task 2: Prepare for Week 2

In week 2, we will optimize the serving infrastructure of the Qwen2 model. We will write some C++ code and Metal kernel
to make some operations run faster.

To prepare for the week to come, you should set up the environment so that it can compile C++ extensions and Metal kernels.
You should make sure the reference solution can be compiled by running:

```
pdm run build-ext-ref
```

You should be able to pass all tests of the reference solution of week 2 by running:

```bash
pdm run test-refsol -- -- -k week_2
pdm run main --solution ref --loader week2
```

{{#include copyright.md}}
