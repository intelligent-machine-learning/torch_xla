# -*- coding: utf-8 -*-
#
#   Lynx is a deep learning automatic compilation system primarily aimed at
# computation optimization and automatic distributed optimization. It is
# built upon the open-source foundations of TorchXLA and OpenXLA/LLVM.
#
#   It is primarily divided into two components: a compiler and a runtime.
# The compiler is responsible for lowering PyTorch's computation graph into
# an intermediate representation known as StableHLO and performing various
# compile-time optimizations, such as operator fusion and automatic
# parallelization. Additionally, it generates executable binary programs
# using LLVM. The runtime component primarily handles the scheduling of
# the compiled SPMD (Single Program, Multiple Data) programs to be executed
# on computing nodes.
#
