# libdl
The examples are currently compiled within the ALL directive, so run:
```
make
```
and then run the generated files in the build/examples/ folder.
Additionally, this has been made an additional step in the gitlab CI (see pipelines).

Important: The weights are initialized randomly such that at every run they are different. This means, occasionally xor does not converge to the desired output. i.e stuck in a local minima (approximately one in 10-15 runs, depending on the parameters).

**xor problem**
* xor.cpp - simple example (added to CI)
* xor_bigger_network - example to illustrate the current implementation of a NEural network can be scaled up
