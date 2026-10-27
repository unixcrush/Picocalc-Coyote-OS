#!/usr/bin/env python3
"""Build a CYX1 keyword search index for the PicoCalc, or query one.

    build_index.py --corpus <dir> --out <file> [--limit N]
                   [--embed-url URL [--model NAME] | --vectors FILE [--model NAME]]
    build_index.py --index <file> --query "text" [--top N] [--embed-url URL [--model NAME]]

docs/design/on-device-search.md pins the file format, the tokenizer, the BM25
scoring, the vector quantization and the fusion rule.  The firmware reader
(ai/search_index.c) and embedding client (ai/embed_client.c) implement the
same; --query is the reference their rankings are checked against.

Standard library only.
"""

import argparse
import fractions
import json
import math
import os
import re
import struct
import sys
import urllib.request
import zlib

MAGIC = b"CYX1"
VERSION = 1
HEADER_SIZE = 160
SECTION_COUNT = 6
TERMS, POSTINGS, DOCS, TEXT, VECTORS, RESERVED = range(SECTION_COUNT)
TERM_ENTRY = struct.Struct("<IIHH")
POSTING = struct.Struct("<IH")
DOC_ENTRY = struct.Struct("<IIHH")

TOKEN_MIN = 2
TOKEN_MAX = 32
CHUNK_CHARS = 600
MAX_HITS = 8  # SEARCH_MAX_HITS in ai/search_index.h
K1 = 1.2
B = 0.75

FLAG_NORMALIZED = 1
FLAG_INT8 = 2
DTYPE_NONE, DTYPE_INT8, DTYPE_F32 = 0, 1, 2
DTYPE_SIZES = {DTYPE_NONE: 0, DTYPE_INT8: 1, DTYPE_F32: 4}
DIM_MAX = 0xFFFF
MODEL_ID_MAX = 63
EMBED_BATCH = 32
EMBED_MODEL_DEFAULT = "nomic-embed-text"  # AI_CONFIG_EMBED_MODEL_DEFAULT
EMBED_TIMEOUT = 300
RRF_K = 60

# Mirrored as SEARCH_STOPWORDS in ai/search_index.h, which must stay sorted.
STOPWORDS = frozenset("""
about above after again against all am an and any are as at be because been
before being below between both but by can did do does doing down during each
few for from further had has have having he her here hers herself him himself
his how if in into is it its itself just me more most my myself no nor not now
of off on once only or other our ours ourselves out over own same she should so
some such than that the their theirs them themselves then there these they this
those through to too under until up very was we were what when where which
while who whom why will with you your yours yourself yourselves
""".split())

_RUN = re.compile(rb"[a-z0-9_]+")
_HEADING = re.compile(r"^#[ \t]+(\S.*?)\s*$")
_HIGH_BYTES = bytes(range(0x80, 0x100))


class FormatError(ValueError):
    """The file is not a valid CYX1 index."""


def fnv1a32(data):
    h = 2166136261
    for b in data:
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def tokenize(text):
    """The tokens of text (str or bytes), in order, as bytes.

    Bytes >= 0x80 are dropped without ending a token, A-Z is folded, runs of
    [a-z0-9_] are cut into 32-byte pieces, and pieces shorter than 2 bytes or
    in STOPWORDS are dropped.
    """
    if isinstance(text, str):
        text = text.encode("utf-8")
    text = text.translate(None, _HIGH_BYTES).lower()
    tokens = []
    for run in _RUN.findall(text):
        for start in range(0, len(run), TOKEN_MAX):
            piece = run[start:start + TOKEN_MAX]
            if len(piece) >= TOKEN_MIN and piece.decode("ascii") not in STOPWORDS:
                tokens.append(piece)
    return tokens


def query_hashes(query):
    """The distinct term hashes of a query, first occurrence first."""
    seen = []
    for token in tokenize(query):
        h = fnv1a32(token)
        if h not in seen:
            seen.append(h)
    return seen


def bm25_idf(n, df):
    return math.log(1.0 + (n - df + 0.5) / (df + 0.5))


def bm25_q16(idf, tf, dl, avgdl_q8):
    """One term's contribution in 1/65536ths; written as in ai/search_index.c."""
    avgdl = avgdl_q8 / 256.0
    norm = K1 * (1.0 - B + B * dl / avgdl)
    part = idf * (tf * (K1 + 1.0)) / (tf + norm)
    return int(math.floor(part * 65536.0 + 0.5))


# ---- vectors --------------------------------------------------------------

def quantize(values):
    """L2-normalize a float vector and quantize it to int8, as
    embed_client_parse in ai/embed_client.c does, operation for operation:
    the squares are summed in order (sum() would compensate and differ), each
    value is divided by the norm, then multiplied by 127, rounded half to even
    and clamped to [-127, 127]."""
    total = 0.0
    for v in values:
        total += v * v
    norm = math.sqrt(total)
    if not values or not math.isfinite(norm) or norm == 0.0:
        raise ValueError("the vector cannot be normalized")
    return [max(-127, min(127, round(v / norm * 127.0))) for v in values]


def vector_q16(dot):
    """A dot product of two int8 vectors as Q16 cosine: floor(dot * 65536 /
    (127 * 127) + 0.5) in exact integers, held to int32, as in
    ai/search_index.c."""
    q16 = (2 * dot * 65536 + 127 * 127) // (2 * 127 * 127)
    return max(-0x80000000, min(0x7FFFFFFF, q16))


def rrf_fuse(*ranked_lists, top=MAX_HITS):
    """Reciprocal rank fusion of [(chunk_id, score)] lists, best first: each
    chunk scores sum(1 / (60 + rank)), rank from 1.  [(chunk_id, score_q16)]
    by exact fused score descending, then lower chunk id; score_q16 is the
    fused score x 65536 rounded half up."""
    fused = {}
    for ranked in ranked_lists:
        for rank, (chunk_id, _score) in enumerate(ranked, 1):
            fused[chunk_id] = fused.get(chunk_id, 0) + fractions.Fraction(1, RRF_K + rank)
    order = sorted(fused.items(), key=lambda cs: (-cs[1], cs[0]))
    return [(c, math.floor(f * 65536 + fractions.Fraction(1, 2))) for c, f in order[:top]]


def truncate_model_id(model):
    """model as at most MODEL_ID_MAX bytes of UTF-8, cut between characters."""
    data = model.encode("utf-8")[:MODEL_ID_MAX]
    return data.decode("utf-8", errors="ignore").encode("utf-8")


def _float_vector(values, what):
    if not isinstance(values, list) or not values:
        raise ValueError("%s is not a non-empty array" % what)
    for v in values:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError("%s holds a value that is not a number" % what)
    try:
        return [float(v) for v in values]
    except OverflowError:
        raise ValueError("%s holds a number too large for a double" % what)


def _reject_constant(name):
    raise ValueError("%s is not a JSON number" % name)


def post_json(url, payload, timeout=EMBED_TIMEOUT):
    """POST payload as JSON to url and return the decoded JSON reply."""
    request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"), parse_constant=_reject_constant)


def embeddings_url(embed_url):
    return embed_url.rstrip("/") + "/v1/embeddings"


def embed_texts(texts, embed_url, model, post=None, batch=EMBED_BATCH):
    """Float vectors for texts from an OpenAI-compatible endpoint, in order.

    texts go at most batch to a request.  A reply that holds the wrong number
    of vectors, or a vector whose length disagrees with the first, is an
    error, never truncated or padded.
    """
    post = post or post_json
    url = embeddings_url(embed_url)
    vectors = []
    for start in range(0, len(texts), batch):
        inputs = texts[start:start + batch]
        reply = post(url, {"input": inputs, "model": model})
        data = reply.get("data") if isinstance(reply, dict) else None
        if not isinstance(data, list):
            raise ValueError("the embedding reply has no data array")
        if len(data) != len(inputs):
            raise ValueError("asked for %d embeddings, got %d" % (len(inputs), len(data)))
        for i, item in enumerate(data):
            what = "embedding %d" % (start + i)
            vector = _float_vector(item.get("embedding") if isinstance(item, dict) else None, what)
            if vectors and len(vector) != len(vectors[0]):
                raise ValueError("%s has %d values, not %d" % (what, len(vector), len(vectors[0])))
            vectors.append(vector)
    return vectors


def read_vectors(path):
    """Precomputed float vectors, one JSON array per line, blank lines
    ignored."""
    vectors = []
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if line.strip():
                what = "%s:%d" % (path, line_number)
                try:
                    values = json.loads(line, parse_constant=_reject_constant)
                except ValueError as e:
                    raise ValueError("%s: %s" % (what, e))
                vectors.append(_float_vector(values, what))
    return vectors


def chunk_vectors(chunks, embed_url=None, model=None, vectors_path=None, post=None):
    """int8 vectors for chunks, from the endpoint or a --vectors file."""
    if vectors_path is not None:
        floats = read_vectors(vectors_path)
    else:
        floats = embed_texts([title + "\n" + body for title, body in chunks], embed_url, model, post)
    if len(floats) != len(chunks):
        raise ValueError("%d vectors for %d chunks" % (len(floats), len(chunks)))
    for i, vector in enumerate(floats):
        if len(vector) != len(floats[0]):
            raise ValueError("vector %d has %d values, not %d" % (i, len(vector), len(floats[0])))
    return [quantize(vector) for vector in floats]


# ---- corpus ---------------------------------------------------------------

def corpus_files(root):
    """*.md and *.txt under root, recursively, sorted by relative path."""
    found = []
    for directory, _dirs, names in os.walk(root):
        for name in names:
            if name.endswith(".md") or name.endswith(".txt"):
                path = os.path.join(directory, name)
                found.append((os.path.relpath(path, root).replace(os.sep, "/"), path))
    found.sort()
    return [path for _rel, path in found]


def file_title(text, path):
    for line in text.split("\n"):
        match = _HEADING.match(line)
        if match:
            return match.group(1)
    return os.path.basename(path)


def paragraphs(text):
    """Blocks of text separated by blank lines, trailing spaces trimmed."""
    result = []
    current = []
    for line in text.split("\n"):
        line = line.rstrip()
        if line:
            current.append(line)
        elif current:
            result.append("\n".join(current))
            current = []
    if current:
        result.append("\n".join(current))
    return result


def split_long(paragraph, limit=CHUNK_CHARS):
    """Cut a paragraph longer than limit at whitespace where it can."""
    pieces = []
    while len(paragraph) > limit:
        cut = max(paragraph.rfind(" ", 0, limit + 1), paragraph.rfind("\n", 0, limit + 1))
        if cut <= 0:
            cut = limit
        pieces.append(paragraph[:cut].rstrip())
        paragraph = paragraph[cut:].lstrip()
    if paragraph:
        pieces.append(paragraph)
    return pieces


def pack(paras, limit=CHUNK_CHARS):
    """Pack consecutive paragraphs, joined by a blank line, into chunks of at
    most limit characters.  Only a paragraph longer than limit is split."""
    chunks = []
    current = ""
    for para in paras:
        for piece in split_long(para, limit):
            if not current:
                current = piece
            elif len(current) + 2 + len(piece) <= limit:
                current += "\n\n" + piece
            else:
                chunks.append(current)
                current = piece
    if current:
        chunks.append(current)
    return chunks


def chunk_file(path):
    """(title, body) chunks of one file."""
    with open(path, "rb") as f:
        text = f.read().decode("utf-8", errors="replace")
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\0", "")
    title = file_title(text, path).replace("\n", " ")
    return [(title, body) for body in pack(paragraphs(text))]


def chunk_corpus(root, limit=None):
    chunks = []
    for path in corpus_files(root):
        for chunk in chunk_file(path):
            if limit is not None and len(chunks) >= limit:
                return chunks
            chunks.append(chunk)
    return chunks


# ---- writing --------------------------------------------------------------

def _align4(n):
    return (n + 3) & ~3


def build(chunks, vectors=None, model_id=""):
    """The bytes of a CYX1 index over (title, body) chunks.

    A chunk's stored text is "title\\nbody"; its tokens, and so its length dl,
    are those of that whole text.  vectors, when given, holds one int8 vector
    per chunk (see quantize), all the same length; model_id names the model
    that made them.  Without vectors the file is the round 1 layout exactly.
    """
    postings = {}  # hash -> {chunk_id: tf}
    docs = []
    text = bytearray()
    total_dl = 0
    for chunk_id, (title, body) in enumerate(chunks):
        if "\n" in title:
            raise ValueError("a title may not contain a newline")
        stored = (title + "\n" + body).encode("utf-8")
        if b"\0" in stored:
            raise ValueError("chunk text may not contain NUL")
        tokens = tokenize(stored)
        if len(tokens) > 0xFFFF:
            raise ValueError("chunk %d has too many tokens" % chunk_id)
        for token in tokens:
            tfs = postings.setdefault(fnv1a32(token), {})
            tfs[chunk_id] = tfs.get(chunk_id, 0) + 1
        docs.append((len(text), len(stored), len(tokens)))
        text += stored + b"\0"
        total_dl += len(tokens)

    n = len(chunks)
    avgdl_q8 = (total_dl * 256 * 2 + n) // (2 * n) if n else 0

    terms_bytes = bytearray()
    postings_bytes = bytearray()
    for h in sorted(postings):
        tfs = postings[h]
        if len(tfs) > 0xFFFF:
            raise ValueError("term %08x is in too many chunks" % h)
        terms_bytes += TERM_ENTRY.pack(h, len(postings_bytes), len(tfs), 0)
        for chunk_id in sorted(tfs):
            postings_bytes += POSTING.pack(chunk_id, min(tfs[chunk_id], 0xFFFF))
    docs_bytes = b"".join(DOC_ENTRY.pack(off, length, dl, 0) for off, length, dl in docs)

    # An empty corpus has nothing to embed and is written as round 1 wrote it.
    flags, dim, dtype, vectors_bytes, model_bytes = 0, 0, DTYPE_NONE, b"", b""
    if vectors is not None:
        if len(vectors) != n:
            raise ValueError("%d vectors for %d chunks" % (len(vectors), n))
    if vectors:
        dim = len(vectors[0])
        if not 0 < dim <= DIM_MAX:
            raise ValueError("vector dimension %d is out of range" % dim)
        for chunk_id, vector in enumerate(vectors):
            if len(vector) != dim:
                raise ValueError("chunk %d has a %d-value vector, not %d" % (chunk_id, len(vector), dim))
            if any(not -128 <= q <= 127 for q in vector):
                raise ValueError("chunk %d's vector is not int8" % chunk_id)
        vectors_bytes = b"".join(struct.pack("<%db" % dim, *vector) for vector in vectors)
        flags, dtype = FLAG_NORMALIZED | FLAG_INT8, DTYPE_INT8
        model_bytes = truncate_model_id(model_id)

    bodies = [bytes(terms_bytes), bytes(postings_bytes), docs_bytes, bytes(text), vectors_bytes, b""]
    sections = []
    payload = bytearray()
    offset = HEADER_SIZE
    for body in bodies:
        start = _align4(offset)
        payload += b"\0" * (start - offset)
        sections.append((start, len(body)))
        payload += body
        offset = start + len(body)
    if offset > 0xFFFFFFFF:
        raise ValueError("index too large")

    def header(crc):
        h = bytearray(HEADER_SIZE)
        struct.pack_into("<4sHHHHIIIII", h, 0, MAGIC, VERSION, flags, dim, dtype, n, len(postings), avgdl_q8,
                         crc, 0)
        h[32:32 + len(model_bytes)] = model_bytes
        for i, (off, length) in enumerate(sections):
            struct.pack_into("<II", h, 96 + 8 * i, off, length)
        return h

    data = header(0) + payload
    crc = zlib.crc32(bytes(data[28:])) & 0xFFFFFFFF
    return bytes(header(crc) + payload)


# ---- reading --------------------------------------------------------------

class Index:
    """A parsed CYX1 index, checked as the firmware checks it."""

    def __init__(self, data):
        if len(data) < HEADER_SIZE:
            raise FormatError("truncated header")
        magic, version, self.flags, self.dim, self.dtype, self.chunk_count, self.term_count, \
            self.avgdl_q8, crc, _reserved = struct.unpack_from("<4sHHHHIIIII", data, 0)
        if magic != MAGIC or version != VERSION:
            raise FormatError("bad magic or version")
        self.model_id = data[32:96].rstrip(b"\0")
        self.sections = [struct.unpack_from("<II", data, 96 + 8 * i) for i in range(SECTION_COUNT)]
        expected = HEADER_SIZE
        for off, length in self.sections:
            if off != expected or off + length > len(data):
                raise FormatError("bad section bounds")
            expected = _align4(off + length)
        if self.sections[-1][0] + self.sections[-1][1] != len(data):
            raise FormatError("bad section bounds")
        if self.sections[TERMS][1] != self.term_count * TERM_ENTRY.size or \
                self.sections[DOCS][1] != self.chunk_count * DOC_ENTRY.size:
            raise FormatError("bad section lengths")
        if self.dtype not in DTYPE_SIZES or (self.dtype == DTYPE_NONE and self.dim != 0) or \
                self.sections[VECTORS][1] != self.chunk_count * self.dim * DTYPE_SIZES[self.dtype]:
            raise FormatError("bad vectors section")
        if zlib.crc32(data[28:]) & 0xFFFFFFFF != crc:
            raise FormatError("bad crc")
        self.data = data

    def section(self, i):
        off, length = self.sections[i]
        return self.data[off:off + length]

    def terms(self):
        terms = self.section(TERMS)
        return [TERM_ENTRY.unpack_from(terms, i * TERM_ENTRY.size) for i in range(self.term_count)]

    def postings(self, postings_off, df):
        section = self.section(POSTINGS)
        return [POSTING.unpack_from(section, postings_off + i * POSTING.size) for i in range(df)]

    def doc(self, chunk_id):
        return DOC_ENTRY.unpack_from(self.section(DOCS), chunk_id * DOC_ENTRY.size)

    def text(self, chunk_id):
        off, length, _dl, _flags = self.doc(chunk_id)
        return bytes(self.section(TEXT)[off:off + length])

    def title(self, chunk_id):
        return self.text(chunk_id).split(b"\n", 1)[0].decode("utf-8", errors="replace")

    def query(self, text, top=MAX_HITS):
        """[(chunk_id, score_q16)], best first, ties by lower chunk id."""
        by_hash = {h: (off, df) for h, off, df, _r in self.terms()}
        scores = {}
        for h in query_hashes(text):
            if h not in by_hash:
                continue
            off, df = by_hash[h]
            idf = bm25_idf(self.chunk_count, df)
            for chunk_id, tf in self.postings(off, df):
                dl = self.doc(chunk_id)[2]
                scores[chunk_id] = scores.get(chunk_id, 0) + bm25_q16(idf, tf, dl, self.avgdl_q8)
        ranked = sorted(((c, s) for c, s in scores.items() if s > 0), key=lambda cs: (-cs[1], cs[0]))
        return ranked[:top]

    def vector(self, chunk_id):
        off = chunk_id * self.dim
        return list(struct.unpack_from("<%db" % self.dim, self.section(VECTORS), off))

    def query_vector(self, query, top=MAX_HITS):
        """[(chunk_id, score_q16)] by int8 dot product with query, best first,
        ties by lower chunk id.  Every chunk is ranked, whatever its score."""
        if self.dim == 0:
            raise ValueError("the index has no vectors")
        if len(query) != self.dim:
            raise ValueError("the query has %d values, the index %d" % (len(query), self.dim))
        if self.dtype != DTYPE_INT8:
            raise FormatError("vectors are not int8")
        scores = [(c, vector_q16(sum(a * b for a, b in zip(query, self.vector(c)))))
                  for c in range(self.chunk_count)]
        return sorted(scores, key=lambda cs: (-cs[1], cs[0]))[:top]

    def query_hybrid(self, text, query, top=MAX_HITS):
        """BM25 and vector rankings, MAX_HITS deep each, fused by rrf_fuse.
        An index without vectors gives the BM25 ranking itself."""
        if self.dim == 0:
            return self.query(text, top)
        return rrf_fuse(self.query(text), self.query_vector(query), top=top)


def load(path):
    with open(path, "rb") as f:
        return Index(f.read())


# ---- command line ---------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--corpus", help="directory of *.md and *.txt files to index")
    parser.add_argument("--out", help="index file to write")
    parser.add_argument("--limit", type=int, help="index at most N chunks")
    parser.add_argument("--embed-url", help="base URL of an OpenAI-compatible embedding service, "
                                            "e.g. http://wang.local:11434")
    parser.add_argument("--model", help="embedding model (default %s with --embed-url); "
                                        "recorded in the index" % EMBED_MODEL_DEFAULT)
    parser.add_argument("--vectors", help="precomputed vectors, one JSON array per line in chunk order")
    parser.add_argument("--index", help="index file to query")
    parser.add_argument("--query", help="query text")
    parser.add_argument("--top", type=int, default=MAX_HITS, help="results to print (default %(default)s)")
    args = parser.parse_args(argv)

    if args.embed_url is not None and args.vectors is not None:
        parser.error("give --embed-url or --vectors, not both")
    model = args.model
    if model is None and args.embed_url is not None:
        model = EMBED_MODEL_DEFAULT

    if args.corpus is not None and args.out is not None:
        chunks = chunk_corpus(args.corpus, args.limit)
        vectors = None
        if args.embed_url is not None or args.vectors is not None:
            try:
                vectors = chunk_vectors(chunks, args.embed_url, model, args.vectors)
            except (OSError, ValueError) as e:
                print("vectors: %s" % e, file=sys.stderr)
                return 1
        data = build(chunks, vectors, model or "")
        with open(args.out, "wb") as f:
            f.write(data)
        print("%d chunks, %d bytes -> %s" % (len(chunks), len(data), args.out), file=sys.stderr)
        return 0
    if args.index is not None and args.query is not None:
        try:
            index = load(args.index)
        except FormatError as e:
            print("%s: %s" % (args.index, e), file=sys.stderr)
            return 1
        if args.vectors is not None:
            parser.error("--vectors is for building; query with --embed-url")
        if args.embed_url is None:
            print_hits(index, index.query(args.query, args.top))
            return 0
        try:
            query = quantize(embed_texts([args.query], args.embed_url, model)[0])
            vector_hits = index.query_vector(query, args.top)
        except (OSError, ValueError) as e:
            print("vectors: %s" % e, file=sys.stderr)
            return 1
        print("# bm25")
        print_hits(index, index.query(args.query, args.top))
        print("# vector")
        print_hits(index, vector_hits)
        print("# hybrid")
        print_hits(index, index.query_hybrid(args.query, query, args.top))
        return 0
    parser.error("give --corpus and --out, or --index and --query")
    return 2


def print_hits(index, hits):
    for chunk_id, score in hits:
        print("%d\t%.6f\t%s" % (chunk_id, score / 65536.0, index.title(chunk_id)))


if __name__ == "__main__":
    sys.exit(main())
