# What's new in grobx 0.155.1

Changes since 0.73.3.

- Calls over GF(2) are 43.8 times as fast as in 0.101, the earliest version measured over GF(2),
  and hold 79% less memory at their peak, as the geometric mean over 16 benchmark instances: faster
  reductions in packed rows, matrix builds, symbolic preprocessing, pair updates and the criteria M,
  F and B, the substitution of linear generators, and the signature run on systems such as bivium.
- Calls over GF(p) are 1.43 times as fast as in 0.73 and hold 27% less memory at their peak, both
  as the geometric mean over 58 benchmark instances.
- Calls over ℚ are 1.22 times as fast as in 0.73 and hold 30% less memory at their peak, over 44
  benchmark instances.
- cyclic-9 over GF(65521) takes less than half the time 0.73 took.
- A deadline, a cancel, or a memory ceiling stops a call within about 0.1 s, over ℚ too.
- The order deglex is new. Sage maps its deglex, dp, and lp to grobx, and Maple its grlex.
- The CLI's `--memory` takes the suffixes k, m, g, and t, in either case, and its `--report` writes
  to a device or a pipe.
- grobx no longer changes the process's malloc settings: the heap pad and the mmap threshold are
  gone.
- No error is dropped and no fallback runs silently: every failure reaches the caller.
- The C interface refuses to return a text that holds a NUL byte, with an error.
- `grobx.Result` carries `sink_error`, the first write error of the call report's file.
- The call report holds the time a plain run takes to enter the elements it starts from, and the
  processor's brand without trailing NUL bytes.
- The checker's cheap mode keeps its memory within its batch.
- A replay that does not fit its record names the step of the record, counted from 1.
