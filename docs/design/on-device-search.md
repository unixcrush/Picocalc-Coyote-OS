# On-device search index

A card-resident index for the PicoCalc. Round 1 is keyword search (BM25) that needs no
network, no model and no host. Round 2 adds embeddings: the host builds vectors, the device
fetches a query vector from a swappable OpenAI-compatible embedding endpoint and scores them
locally. This document pins the file format so the two rounds, and the host-side builder, stay
compatible.

## Why the split

The RP2350 cannot embed a query — a small transformer costs tens of seconds per query on an
M33. What it can do is arithmetic: 10,000 × 256-dim int8 dot products is ~2.5M MACs, tens of
milliseconds. So the expensive work happens on a host (`wang`, `nomic-embed-text` via Ollama,
768 or 256 dims), and the device only scores vectors that were built earlier and copied to the
card.

BM25 is included because "search that stops working when the server does" is a trap. It also
covers exact-term queries better than vectors do, so both are useful even when the network is
up.

## Non-goals for round 1

- No UI changes, no settings rows, no menu entries. Nothing in `UI/` may be touched.
- No network calls. The embedding client is round 2 and lives behind its own module.
- No new dependencies in either language (Python: standard library only; firmware: none).
- No changes to `ai_config`, providers, or credentials.

## Index format `CYX1` (little-endian)

Fixed header, 160 bytes:

| offset | size | field |
|---|---|---|
| 0 | 4 | magic, ASCII `CYX1` — `43 59 58 31` in file order |
| 4 | 2 | version = 1 |
| 6 | 2 | flags: bit0 = vectors normalized, bit1 = vectors are int8 |
| 8 | 2 | dim (0 when there are no vectors) |
| 10 | 2 | dtype: 0 = none, 1 = int8, 2 = f32 |
| 12 | 4 | chunk_count |
| 16 | 4 | term_count |
| 20 | 4 | avgdl_q8 — average document length in tokens × 256, rounded |
| 24 | 4 | crc32 of every byte after this header field (offset 28 to EOF) |
| 28 | 4 | reserved, 0 |
| 32 | 64 | model_id, NUL padded (empty in round 1) |
| 96 | 6 × 8 | section offset + length pairs: `u32 off`, `u32 len` |
| 144 | 16 | reserved, 0 |

Section order is fixed: 0 terms, 1 postings, 2 docs, 3 text, 4 vectors, 5 reserved. Offsets are
absolute from the start of the file; sections are contiguous in that order and 4-byte aligned.
An empty section has `len = 0` and `off` set to where the next section starts.

**Terms** (section 0), `term_count` × 12 bytes, sorted ascending by `hash`:

| size | field |
|---|---|
| 4 | `u32 hash` — FNV-1a 32-bit of the token bytes |
| 4 | `u32 postings_off` — byte offset relative to the start of the postings section |
| 2 | `u16 df` — how many chunks contain the term |
| 2 | reserved, 0 |

**Postings** (section 1): for each term in the same order, `df` × 6 bytes: `u32 chunk_id`,
`u16 tf`. No padding between terms.

**Docs** (section 2), `chunk_count` × 12 bytes: `u32 text_off` (relative to the text section),
`u32 text_len` (bytes, excluding the terminating NUL), `u16 dl` (tokens in this chunk),
`u16 flags` (0).

**Text** (section 3): each chunk as `title\nbody` followed by a NUL. `text_len` counts the
bytes before the NUL.

**Vectors** (section 4), round 2 only: `chunk_count` × `dim` bytes, int8, one vector per chunk
in chunk order, each L2-normalized before quantization so a dot product is a cosine. Empty in
round 1 (`dim = 0`, `len = 0`).

The reader must reject a file whose magic, version, section bounds, or CRC do not match, and
must never read past a section's declared length.

## Tokenization and BM25

Both the Python builder and the C reader must implement exactly this, or the index will not
match:

1. Bytes ≥ 0x80 are skipped (ASCII only for now). A token is a maximal run of `[a-z0-9_]` after
   lowercasing `A-Z`.
2. Keep tokens of length 2–32. Longer runs are split at 32.
3. Drop tokens in the stopword list (identical list in both implementations, defined in the
   Python module and mirrored in the C header as a sorted array).
4. `hash = FNV-1a 32`: `h = 2166136261; for each byte: h ^= b; h *= 16777619`.
5. Chunking: the builder splits a file at blank lines into paragraphs, then packs consecutive
   paragraphs into chunks of at most 600 characters, never splitting a paragraph unless it
   alone exceeds the limit. Each chunk's `title` is the file's first `#` heading (or the file
   name) and its `body` is the packed text.

Score for one query term in one chunk:

```
idf = ln(1 + (N - df + 0.5) / (df + 0.5))
score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
k1 = 1.2, b = 0.75, N = chunk_count, avgdl = avgdl_q8 / 256
```

Chunks scoring 0 are not returned. Ties break by lower `chunk_id`, so results are deterministic.
Duplicate query terms count once.

## Device API (`ai/search_index.h`)

```c
#define SEARCH_MAX_HITS 8

typedef enum {
    SEARCH_OK = 0,
    SEARCH_ERR_ARGUMENT,
    SEARCH_ERR_IO,
    SEARCH_ERR_FORMAT,
    SEARCH_ERR_TOO_LARGE,
} search_status_t;

typedef struct {
    uint32_t chunk_id;
    int32_t score_q16;   /* BM25 score x 65536, for ordering and display */
} search_hit_t;

typedef struct search_index search_index_t;  /* fixed-size, no heap */

search_status_t search_index_open(search_index_t *index, const char *path);
void search_index_close(search_index_t *index);
/* Ranks the chunks matching query. Returns SEARCH_OK with *hit_count == 0 when nothing
 * matches; an empty or all-stopword query is not an error. */
search_status_t search_index_query(search_index_t *index, const char *query,
                                   search_hit_t *hits, size_t max_hits, size_t *hit_count);
/* Copies a chunk's title (up to the first '\n') or body into buf as NUL-terminated UTF-8. */
search_status_t search_index_read_title(search_index_t *index, uint32_t chunk_id,
                                        char *buf, size_t capacity);
search_status_t search_index_read_body(search_index_t *index, uint32_t chunk_id,
                                       char *buf, size_t capacity);
```

Rules:

- No `malloc`. The struct holds its own buffers; keep it under 16 KB so a caller can hold one
  as a static. Read from the file in 4 KB chunks with stdio, as `storage/file_store.c` does —
  `fopen`'s handle is the one allocation the C library makes and cannot be avoided here.
- **Opening checks the header and streams the whole file once for the CRC.** A whole-file CRC
  cannot be verified from the header alone, so open costs one full read of the index
  (three seconds for 3 MB on a slow card). Open once per session and keep the struct, rather
  than reopening per query. If that becomes painful, add a `search_index_verify()` and let open
  check only the header and section bounds.
- A query reads only the postings of the terms it contains (binary search over the sorted term
  table), never the whole index.
- The module must build and pass host tests without the Pico SDK or network.

### Behaviours round 1 pins

- Posting lists list chunk ids in strictly increasing order; the reader rejects anything else as
  a format error. The builder always writes them that way.
- Scores are accumulated as Q16 integers, each term rounded with `floor(x * 65536 + 0.5)`,
  identically in C and Python, so rankings agree exactly across the two implementations.
- A query with more than 16 distinct non-stopword terms is `SEARCH_ERR_TOO_LARGE`; `max_hits`
  above `SEARCH_MAX_HITS` (8) is capped.
- Non-ASCII bytes are dropped without ending a token, so `Zürich` indexes as `zrich`. Treat this
  as a limitation to revisit with the embedding round, not a feature.
- A chunk's length counts its title as well as its body.
- A title is the first `# ` heading in the file, otherwise the file name.

## Host builder (`tools/search_index/build_index.py`)

```
python3 tools/search_index/build_index.py --corpus <dir> --out <file> [--limit N]
python3 tools/search_index/build_index.py --index <file> --query "text" [--top N]
```

- Walks `*.md` and `*.txt` recursively, in sorted path order.
- Builds BM25 sections and writes a valid `CYX1` file. `model_id` is empty, `dim` is 0, and the
  vectors section is empty in round 1.
- `--query` reads an existing index, tokenizes the same way, scores, and prints
  `chunk_id<TAB>score<TAB>title` per result — this is the reference implementation used to
  check the C ranking.
- Standard library only.

## Tests

C, in `tests/test_search_index.c`, registered in `tests/CMakeLists.txt` and called from
`test_ai_config()`'s runner (`tests/test_main.c`):

- The test writes its own fixture file (literal bytes assembled in the test) so it does not
  depend on Python. Cover: a query matching several chunks ranks by BM25, not by tf alone; a
  stopword-only query returns zero hits with `SEARCH_OK`; a truncated file, bad magic, bad
  version, out-of-bounds section and bad CRC each return `SEARCH_ERR_FORMAT`; `NULL` arguments
  return `SEARCH_ERR_ARGUMENT`; `max_hits` smaller than the number of matches truncates to the
  best ones.
- Ranking ties break by chunk id.

Python, in `tools/search_index/test_build_index.py` (stdlib `unittest`):
- Tokenizer: case folding, stopwords, length bounds, `_` and digits kept.
- Header fields, section contiguity, term sorting, postings counts, CRC.
- A 4-document corpus where the expected ranking is obvious.

## Acceptance

1. `cmake -S tests -B build-tests && cmake --build build-tests && ./build-tests/coyote_ai_tests`
   prints `N checks, 0 failures`.
2. `python3 -m unittest discover -s tools/search_index -p 'test_*.py'` passes.
3. `python3 tools/search_index/build_index.py --corpus <fixture dir> --out /tmp/a.idx` then
   `--index /tmp/a.idx --query "..."` returns the expected order.
4. The firmware still cross-builds for `pico2_w` with no new warnings.

## Round 2: vectors and hybrid search

Round 1 left section 4 empty. Round 2 fills it and adds the query side. The device still never
embeds text locally: it asks an OpenAI-compatible endpoint for the query vector and scores that
against the card.

### Builder

```
python3 tools/search_index/build_index.py --corpus DIR --out FILE \
    --embed-url http://wang.local:11434 --model nomic-embed-text
python3 tools/search_index/build_index.py ... --vectors FILE   # offline, for tests
```

- Posts chunks in batches (at most 32 per request) to `{embed-url}/v1/embeddings` with
  `{"input": [...], "model": "..."}`. Accepts `{"data":[{"embedding":[...]}], ...}` in input
  order; a response whose count or dimension disagrees is an error, not a silent truncation.
- L2-normalizes each vector, then quantizes with `q = round(v * 127)`, clamped to [-127, 127].
  Sets `dim`, `model_id` (the model string, truncated to 63 bytes), flag bit0 (normalized) and
  bit1 (int8).
- `--vectors FILE` reads precomputed vectors, one JSON array per line in chunk order, and writes
  the same layout. This is how the tests run without a network.
- `--query` gains `--embed-url`/`--model` so the reference query path can score a vector too,
  and reports BM25, vector and fused results separately.

### Configuration

Two new fields in `ai_config`, both persisted in `ai.ini`:

- `embed_url` — base URL of the embedding service, e.g. `http://wang.local:11434`. Empty means
  "use the selected provider's endpoint"; that is the default and needs no typing.
- `embed_model` — model name, default `nomic-embed-text`.

No credential is added in this round: LAN embedding endpoints (Ollama) are keyless. A keyed
endpoint needs a credential slot, which is a later round; do not reuse the provider key for a
different host.

### Reader

```c
search_status_t search_index_query_vectors(search_index_t *index, const int8_t *query,
                                           size_t dim, search_hit_t *hits, size_t max_hits,
                                           size_t *hit_count);
search_status_t search_index_query_hybrid(search_index_t *index, const char *text,
                                          const int8_t *query_vector, size_t dim,
                                          search_hit_t *hits, size_t max_hits, size_t *hit_count);
```

- `dim` must equal the index `dim`, otherwise `SEARCH_ERR_DIM` (new status). An index with
  `dim == 0` has no vectors: the vector call returns `SEARCH_ERR_NO_VECTORS`, the hybrid call
  falls back to BM25 and returns `SEARCH_OK`.
- Vector score is the int32 dot product of two int8 vectors divided by `127 * 127`, expressed
  as Q16 like the BM25 score.
- Hybrid uses reciprocal rank fusion over the two ranked lists: `score = sum(1 / (60 + rank))`
  with rank starting at 1, deduplicated by chunk id, ordered by fused score descending and then
  by lower chunk id. It never returns a chunk neither list returned.
- The vectors section is streamed in `SEARCH_IO_CHUNK` pieces; only the top-k are kept, so N
  chunks cost N × dim bytes of reads and no extra memory.

### Embedding client (device)

`ai/embed_client.{c,h}` builds the request and parses the response; it owns no UI and no
scheduling.

```c
typedef enum {
    EMBED_OK = 0,
    EMBED_ERR_ARGUMENT,
    EMBED_ERR_DIM,
    EMBED_ERR_HTTP,
    EMBED_ERR_PARSE,
} embed_status_t;

/* Text limit; longer input is EMBED_ERR_ARGUMENT. */
#define EMBED_TEXT_MAX 512

/* Builds the request body. Returns the length written, or 0. */
size_t embed_client_body(char *body, size_t capacity, const char *model, const char *text);

/* Parses {"data":[{"embedding":[...]}]} into dim int8 values, L2-normalized and quantized
 * exactly as the builder does. Returns EMBED_ERR_DIM when the vector length disagrees. */
embed_status_t embed_client_parse(const char *json, size_t length, int8_t *out, size_t dim);
```

- JSON parsing must tolerate whitespace, exponents and negative values, and reject a response
  with no `embedding` array.
- Quantization here and in the builder must be the same arithmetic; a test compares them.
- The HTTP transport is not part of this round: the caller owns the request and feeds it the
  response body. This keeps the round verifiable without a radio and without touching
  `app_services` or `UI/`.

### Tests

C, `tests/test_search_index.c` additions:
- Vector scan over a fixture with known vectors ranks by cosine, not by magnitude.
- `dim` mismatch leads to `SEARCH_ERR_DIM`; no vectors leads to `SEARCH_ERR_NO_VECTORS`; hybrid
  falls back to BM25 and still returns `SEARCH_OK`.
- RRF ordering for a chunk that is 1st in BM25 and 3rd in vectors, and vice versa.
- `embed_client_parse` on canned responses: whitespace, exponents, negatives, too few values,
  no data, empty body; and that parsing the builder's stored vector reproduces it exactly.

Python:
- Batch splitting, model_id and dim in the header, normalization, quantization, and that a
  response of the wrong length is rejected.

### Behaviours round 2 pins

- A vector query ranks every chunk, including those with a negative cosine; the caller decides
  how far down the list to show.
- Hybrid takes `SEARCH_MAX_HITS` from each list before fusing, so a chunk outside the top eight
  of both lists is never returned.
- On an index with no vectors, hybrid returns the BM25 result whatever vector or length it is
  handed, even none.
- An index whose `dtype` is float32 has no vector support yet: the vector query returns
  `SEARCH_ERR_FORMAT` rather than guessing.
- The response parser rejects an empty model name, a reply carrying more than one embedding, and
  an all-zero vector.
- Chunks are embedded exactly as stored (`title\nbody`). `nomic-embed-text` recommends
  `search_document:` / `search_query:` prefixes; they are not applied here because the builder
  must stay model-agnostic. Adding `--doc-prefix` / `--query-prefix` (default empty) is the open
  improvement.
- Python's compensated `sum()` is avoided in the builder so its accumulation matches the C
  loop bit for bit; both sides round half to even.

### Acceptance for round 2

1. Host C suite and Python suite pass with no pre-existing expectation changed.
2. A real index built from the vault with `nomic-embed-text` on wang: the Python reference and
   the C reader agree on the top-5 for both BM25 and vector queries.
3. Firmware still cross-builds for `pico2_w`, 0 warnings.

## Round 3: vector scoring on core 1

Vector and hybrid search read every vector on the card, so a query keeps core 0 busy for as
long as the card takes to stream N × dim bytes. Round 3 moves the arithmetic of that pass to
RP2350 core 1 so the main loop and UI can run between steps. The synchronous APIs are
untouched and still the reference: the offloaded path must return exactly their hits.

### Ownership

| Thing | Owner | Why |
| --- | --- | --- |
| The open index, its `FILE *`, the SD card, FatFs | core 0, always | the card is not shared, and `usb_disk_mode` must be able to take it |
| Wi-Fi / lwIP / TLS, UI, keyboard | core 0 | unchanged; none of it moves |
| BM25 half of a hybrid query | core 0, synchronous | card-bound, not CPU-bound; see limits |
| int8 dot products, top-k of one block | core 1 | pure CPU over memory the job carries |

Core 1 never calls `fopen`, `fread`, `fclose`, FatFs or stdio, and never allocates.
`ai/search_worker.c` includes none of them; its only undefined symbols are `memcpy`, the block
scorer and the three platform hooks. No file is held open "across cores": core 0 opens the
index, and core 1 is only ever given bytes.

### Design: core 0 reads, core 1 scores one block at a time

The simpler safe choice was to stream, not to snapshot. A whole vectors section does not
fit a bounded buffer (768 dims × 1000 chunks is 750 KB), so a snapshot would cap index size.
Instead core 0 reads one block of whole vectors (at most `SEARCH_WORKER_BLOCK_BYTES`, 4096,
so 5 vectors at dim 768) into the index's existing `io` buffer and submits it. The worker
**copies** the query and block into its own fixed buffers, so core 0 may reuse `io` and read
the next block while core 1 scores the last. Core 1 returns the best `SEARCH_MAX_HITS` of that
block; core 0 merges them in rising chunk order with `search_index_keep_hit`. The global top
eight is always inside the union of the blocks' top eights, and ties still go to the lower
chunk id because blocks are merged in order, so the result equals `search_index_query_vectors`
(tests compare them, including a tie across a block boundary).

New pure helpers in `search_index.h` (no `FILE`): `search_index_score_block`,
`search_index_keep_hit`, `search_index_fuse` (the RRF step, now shared with
`search_index_query_hybrid`), and `search_index_read_vector_block` (core 0's read).

### API

`ai/search_worker.h` (the worker; hardware-free):

```c
search_worker_status_t search_worker_init(void);
search_worker_status_t search_worker_submit_vector(const int8_t *query, size_t dim,
        const uint8_t *block, size_t block_length, uint32_t first_chunk_id);
search_worker_state_t  search_worker_poll(void);
search_worker_status_t search_worker_take_result(search_worker_result_t *result);
search_worker_status_t search_worker_shutdown(void);
int search_worker_run_pending(void);   /* core 1's side; tests call it in place of core 1 */
```

`ai/search_vector_job.h` (core 0's driver over an open index):
`search_vector_job_begin` / `_begin_hybrid`, `_poll`, `_result`, `_status`, `_cancel`. Each
`_poll` does at most one bounded read, one submit and one result take, and never waits.

Bounds: `dim` 1..`SEARCH_WORKER_MAX_DIM` (1024), block ≤ 4096 bytes and a whole number of
vectors. Anything else is refused at submit and nothing is queued. Static memory is about 6 KB
(the job slot 5.2 KB, a job 1.1 KB).

### Concurrency model

```
STOPPED --init--> IDLE --submit--> QUEUED --core1--> RUNNING --core1--> DONE --take--> IDLE
                   ^ shutdown (IDLE only) -> STOPPED
```

- One word, `state`, is the only thing both cores race on. Each state has one owner (core 0 in
  STOPPED, IDLE and DONE; core 1 in QUEUED and RUNNING) and only the owner touches the slot's
  other fields. Hand-over is a release store, pick-up an acquire load
  (`__atomic_*`, which the compiler turns into `dmb`), so a core sees everything the other
  wrote before it handed over. Core 1 claims the job with a compare-exchange QUEUED → RUNNING,
  so a stray or doubled wake does nothing.
- Submit while not IDLE returns `SEARCH_WORKER_BUSY` and leaves the running job alone; so does
  shutdown. An untaken result keeps the worker DONE, so it also blocks a new job: one result
  is never overwritten.
- Wake-up is the SDK multicore FIFO (`search_worker_pico.c`): core 1 sleeps in
  `multicore_fifo_pop_blocking()` (WFE), core 0 pushes one doorbell word with a zero timeout.
  At most one job is in flight, so the 8-deep FIFO cannot fill. Core 0 never spins: it learns of
  completion by polling `state` from the main loop. `FreeRTOS` is not used.
- Shutdown (IDLE only) resets core 1 with `multicore_reset_core1()`; `init` can start it again.
  Core 1 calls `multicore_lockout_victim_init()` so a future flash write on core 0 can pause it;
  nothing writes flash today.
- The platform seam is three functions (`search_worker_platform_start/wake/stop`). The firmware
  links `search_worker_pico.c`; the host tests define fakes and call `search_worker_run_pending`
  themselves, so they need no Pico SDK.

### Limits of this slice

- **Not hardware-tested.** No board was connected. Verified: the host suite (state transitions,
  rejection while busy, result delivery, dim and length bounds, shutdown and re-init,
  equivalence with the synchronous queries), a ThreadSanitizer run of the same worker code with a
  pthread standing in for core 1 (20,000 jobs and shutdown/re-init cycles, clean), and that the
  firmware compiles and links for `pico2_w` with `-Wall -Wextra -Werror` on the new files.
  Real multicore launch, WFE wake-up, the default core 1 stack (2 KB, enough for this code's
  frames but unmeasured), SRAM contention with the SD DMA, and the actual speed-up are unproven.
- **The gain is small until proven.** A 4 KB block is about 4,000 multiply-accumulates, an estimated ~0.1 ms on core 1 (not measured); the
  SD read that feeds it is the larger cost and stays on core 0. What this buys is that the
  scoring and the top-k bookkeeping overlap the next read, and that core 0's work per poll is one
  bounded read. It does not make the card faster. Measure before relying on it.
- Hybrid runs its BM25 half in `_begin_hybrid`, synchronously on core 0. BM25 is a few small
  reads, but it does block the loop for their duration.
- Only the int8 path. A float32 index is `SEARCH_ERR_FORMAT`, as in the synchronous query.
- `main.c`, `UI/` and the provider code are untouched: nothing calls `search_worker_init()` yet.
  Whoever wires a search UI must: init once, drive `search_vector_job_poll` from the idle hook,
  and cancel and drain a job before `usb_disk_mode` unmounts the card.
- A core 1 that hangs leaves a job PENDING forever; there is no timeout.
