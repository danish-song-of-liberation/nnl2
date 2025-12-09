```text
 .--..--..--..--..--..--..--..--..--. 
/ .. \.. \.. \.. \.. \.. \.. \.. \.. \
\ \/\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ \/ /
 \/ /`--'`--'`--'`--'`--'`--'`--'\/ / 
 / /\                            / /\ 
/ /\ \               _ ____     / /\ \
\ \/ /   _ __  _ __ | |___ \    \ \/ /
 \/ /   | '_ \| '_ \| | __) |    \/ / 
 / /\   | | | | | | | |/ __/     / /\ 
/ /\ \  |_| |_|_| |_|_|_____|   / /\ \
\ \/ /                          \ \/ /
 \/ /                            \/ / 
 / /\.--..--..--..--..--..--..--./ /\ 
/ /\ \.. \.. \.. \.. \.. \.. \.. \/\ \
\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `' /
 `--'`--'`--'`--'`--'`--'`--'`--'`--' 
```

# nnl2
Common Lisp (CL) neural network library

*About the Author:* This framework is being developed by a 15-year-old as a personal solo project. All code, all bugs, all mine!

I write the framework mainly for myself because writing on torch or tensorflow develops into eternal procrastination and unwillingness to write code on it.

Framework has a first version (nnl), you can see it in my repositories.

# Why didn't I decide to finish the first version of NNL?

to be more specific, due to problems with BPTT, recurrent graphs, and poor library selection

# Framework Architecture

The framework is divided into three main components:

1. Tensor System (100% complete)
2. Autodiff System (60% complete) (MVP is 100% ready) (so yet only reverse mode is ready)
3. Neural Networks Implementations (10% complete)

Its only beginning.

# Documentation

You can see docs at /nnl2/src

# Benchmarks 

You can see benchmarks at /nnl2/benchmarks
