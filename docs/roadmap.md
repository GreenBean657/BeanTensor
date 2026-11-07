# BeanTensor Roadmap Decisions

Scope and design decisions for post-MVP work, recorded per the post-MVP
workload's discipline: one or two lines each, dated. Semantics documentation
lives in `architecture.md`; results live in their own files; this file is the
decision log.

## 2026-11-05 — Bind-clean checked header set (Task 1)

- `verifications/bind_clean_public_api.sh` checks `Module/*.h`,
  `Random/*.h`, `Data/*.h`, `Metrics/*.h`, and `Serialization/*.h` (minus
  `Module/ModuleStream.h`).
- `Tensor/Tensor.h` and `Tensor/Ops/TensorOps.h` are **excluded** from the
  checked set: their scanner findings are namespace-scope `detail` internals
  (autograd trampolines, dispatcher hooks), which are not Python-binding
  surface, plus `Tensor::read_element`/`write_element`'s `void*` buffer
  parameter — an explicit exception (C++-only byte I/O; bindings use a
  future buffer protocol). Revisit this exclusion when the actual pybind11
  binding effort begins.

## 2026-11-05 — `.bt` model export format (Task 3)

- **Codec ruling:** compression is a from-scratch greedy LZ77 (64 KiB window,
  `Serialization/Lz77.{h,cpp}`); zlib and every external library is banned.
  A record is stored compressed only when its LZ form is smaller.
- **Optimizer state ruling:** Adam/SGD moment state is NOT exported in this
  task (stretch goal deferred); `.bt` carries architecture + parameter values
  + BatchNorm running statistics only.
- **Factory ruling:** the save-side traversal dispatches through virtual
  `Module::spec()` / `state_tensors()` / `named_children()` hooks (no switch on
  class names); the load side uses one central factory table in
  `Serialization/Serialization.cpp` keyed by the spec's type string.
- **Dtype ruling:** tensor records carry the dtype; on load the record's dtype
  is authoritative and the rebuilt tensor is `convert_dtype`d to it before raw
  bytes land, so save → load is bit-exact per tensor on one machine.

### `.bt` v1 byte layout (all integers little-endian)

```
magic        4 bytes   'B' 'T' '0' '1'
version      u32       currently 1; newer files raise UnsupportedVersion
flags        u32       reserved, written 0
spec         recursive module node:
               type          u32 len + UTF-8 bytes (== Module::name())
               n_hyper       u32
               per hyper:    u32 klen + key bytes, u32 vlen + value bytes
               n_children    u32
               per child:    u32 namelen + registry name bytes, then spec
n_tensors    u32
records      n_tensors × tensor record, in this order:
               1. parameters — Module::named_parameters() subtree-flat order,
                  names already qualified ("0.weight", ...)
               2. state tensors — Module::state_tensors() per module, walked
                  via named_children(), names prefixed with the module path
                  ("0.running_mean", ...)
             per record:
               name          u32 len + bytes
               dtype         u8 tag (1=Int8 2=Int16 3=Int32 4=Int64 5=UInt16
                             6=UInt32 7=UInt64 8=Float16 9=BFloat16 10=Float32
                             11=Float64 12=Float16x2 13=BFloat16x2 14=Int4x2;
                             tags are append-only, never reassigned)
               rank          u32, then rank × u64 extents
               crc32         u32 (IEEE, over the RAW bytes)
               compression   u8 (0 = raw, 1 = LZ77 token stream)
               raw_size      u64
               stored_size   u64
               payload       stored_size bytes
```

LZ77 token stream: `0x00` + u8 count + count literal bytes (runs of 1..255);
`0x01` + u8 (match_len − 3) + u16 LE offset (match_len 3..258, offset
1..65535); `0x02` end of stream. Every record is CRC-verified against its raw
bytes after decode; the reader also rejects trailing bytes after the last
record and any record whose name/dtype/shape disagrees with the rebuilt
architecture.

### Version 2 checkpoint sections (2026-11-05)

`save(..., CheckpointLevel)` extends the file; section presence travels in the
header's flags field (bit 0 = gradients, bit 1 = Adam), so readers know exactly
what to expect. `Weights` writes a version-1 file (no sections, flags 0).

```
after the v1 tensor records:
grads     only when flags bit 0 set:
            u32 n (= parameter count), then per parameter, in parameter
            record order: u8 present; when 1, a tensor record whose name is
            "<param>/grad" (the grad buffer's own pinned dtype)
adam      only when flags bit 1 set:
            u32 n (= parameter count), then per parameter: u8 present; when 1,
            u64 step_count, then two tensor records "<param>/adam_m" and
            "<param>/adam_v" (the moment buffers, parameter dtype)
```

Load restores whatever sections exist, after the device move, via
`Tensor::restore_grad` and `Tensor::adam_restore_moments` — a `Full`
checkpoint resumes training bit-reproducibly (grads + moments + step counter).
Adam state access is the C++-only `has_adam_state` / `adam_step_count` /
`adam_moments` / `adam_restore_moments` surface on Tensor.

## 2026-11-06 — Demos and benchmarking decisions (Tasks 4, 5)

- **Demo layout:** demos stay separate top-level programs (`main.cpp` MLP,
  `cnn.cpp` CNN) sharing `demos/Mnist.h` (IDX parsing + flat/NCHW upload).
  The library itself has no dataset reader; the CSV loader (Task 6) is the
  planned general path.
- **GPU op-level benchmarking is out of scope** for the benchmark harness:
  timing queued async GPU work needs per-iteration device-sync semantics the
  harness does not implement, and GPU traffic must go through `gpu-test.sh`.
  End-to-end GPU numbers live in `docs/mnist-results.md` and
  `docs/cnn-results.md`.
- **Benchmark methodology:** one warmup + median of 5 timed iterations per
  row, sync inside the timed region; sizes derive from the detected LLC at
  runtime with floors; `benchmark_smoke` fails the suite if any family stops
  producing rows.
