"""Tests for build_index.py.  Run from the repository root:

    python3 -m unittest discover -s tools/search_index -p 'test_*.py' -v
"""

import contextlib
import hashlib
import io
import json
import os
import re
import struct
import sys
import tempfile
import unittest
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import build_index as bi  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def write_tree(root, files):
    for rel, text in files.items():
        path = os.path.join(root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)


class TokenizerTest(unittest.TestCase):
    def test_case_folding(self):
        self.assertEqual(bi.tokenize("Radio RADIO rAdIo"), [b"radio"] * 3)

    def test_stopwords_dropped(self):
        self.assertEqual(bi.tokenize("The radio and the antenna"), [b"radio", b"antenna"])
        self.assertEqual(bi.tokenize("the and of"), [])

    def test_length_bounds(self):
        self.assertEqual(bi.tokenize("x y zz"), [b"zz"])
        self.assertEqual(bi.tokenize("q" * 32), [b"q" * 32])
        # A longer run is cut into 32-byte pieces; a 1-byte remainder is dropped.
        self.assertEqual(bi.tokenize("q" * 33), [b"q" * 32])
        self.assertEqual(bi.tokenize("q" * 34), [b"q" * 32, b"qq"])
        self.assertEqual(bi.tokenize("q" * 70), [b"q" * 32, b"q" * 32, b"q" * 6])

    def test_underscore_and_digits_kept(self):
        self.assertEqual(bi.tokenize("snake_case i2c 2026 _x"), [b"snake_case", b"i2c", b"2026", b"_x"])

    def test_punctuation_separates(self):
        self.assertEqual(bi.tokenize("wifi-setup,radio.cfg"), [b"wifi", b"setup", b"radio", b"cfg"])

    def test_high_bytes_skipped(self):
        # Bytes >= 0x80 are dropped from the stream; they do not end a token.
        self.assertEqual(bi.tokenize("cafés Zürich"), [b"cafs", b"zrich"])
        self.assertEqual(bi.tokenize("éé é"), [])

    def test_fnv1a(self):
        self.assertEqual(bi.fnv1a32(b""), 0x811C9DC5)
        self.assertEqual(bi.fnv1a32(b"a"), 0xE40C292C)
        self.assertEqual(bi.fnv1a32(b"foobar"), 0xBF9CF968)

    def test_query_terms_distinct(self):
        self.assertEqual(bi.query_hashes("radio Radio RADIO antenna"),
                         [bi.fnv1a32(b"radio"), bi.fnv1a32(b"antenna")])

    def test_stopwords_match_c_header(self):
        with open(os.path.join(REPO, "ai", "search_index.h"), encoding="utf-8") as f:
            header = f.read()
        block = re.search(r"#define SEARCH_STOPWORDS(.*?)\n\n", header, re.S).group(1)
        words = re.findall(r'"([^"]*)"', block)
        # Sorted bytewise, for the reader's binary search.
        self.assertEqual(words, sorted(words, key=lambda w: w.encode("ascii")))
        self.assertEqual(len(words), len(set(words)))
        self.assertEqual(set(words), set(bi.STOPWORDS))
        for word in words:
            # Every stopword is itself a token, or it could never match.
            self.assertEqual(bi._RUN.fullmatch(word.encode("ascii")).group(0), word.encode("ascii"))
            self.assertTrue(bi.TOKEN_MIN <= len(word) <= bi.TOKEN_MAX)


class ChunkingTest(unittest.TestCase):
    def test_paragraphs_packed(self):
        paras = ["a" * 200, "b" * 200, "c" * 200, "d" * 10]
        chunks = bi.pack(paras)
        self.assertEqual(chunks, ["a" * 200 + "\n\n" + "b" * 200, "c" * 200 + "\n\n" + "d" * 10])
        self.assertTrue(all(len(c) <= bi.CHUNK_CHARS for c in chunks))

    def test_long_paragraph_split(self):
        para = " ".join(["word%03d" % i for i in range(200)])  # 1599 chars
        chunks = bi.pack(["short", para, "tail"])
        self.assertTrue(all(len(c) <= bi.CHUNK_CHARS for c in chunks))
        self.assertEqual(chunks[0], "short")
        self.assertEqual(" ".join(" ".join(chunks[1:]).split()), para + " tail")
        unbroken = bi.pack(["x" * 1300])
        self.assertEqual(unbroken, ["x" * 600, "x" * 600, "x" * 100])

    def test_title_from_heading_or_name(self):
        with tempfile.TemporaryDirectory() as root:
            write_tree(root, {"a.md": "intro\n\n# Wifi Setup\n\nbody", "b.txt": "no heading\n## sub"})
            self.assertEqual(bi.chunk_file(os.path.join(root, "a.md"))[0][0], "Wifi Setup")
            self.assertEqual(bi.chunk_file(os.path.join(root, "b.txt"))[0][0], "b.txt")

    def test_corpus_walk_sorted_and_filtered(self):
        with tempfile.TemporaryDirectory() as root:
            write_tree(root, {"z.md": "z", "sub/b.txt": "b", "a.md": "a", "c.rst": "c", "sub/a/x.md": "x"})
            rels = [os.path.relpath(p, root).replace(os.sep, "/") for p in bi.corpus_files(root)]
            self.assertEqual(rels, ["a.md", "sub/a/x.md", "sub/b.txt", "z.md"])
            self.assertEqual(len(bi.chunk_corpus(root, limit=2)), 2)


FIXTURE_CHUNKS = [
    ("Notes", "The keyboard reads keys over i2c and the display draws text over spi; the radio is "
              "mentioned here, and the radio again, among many other words about matrix scanning "
              "debounce timing layers modifiers repeat rates and power"),
    ("Wireless", "radio antenna tuning"),
    ("Battery", "battery charge gauge battery cells"),
    ("Battery", "battery charge gauge battery cells"),
    ("Sound é", "pwm audio speaker tones"),
    ("Radio antenna", "antenna placement near the radio shield"),
]


class FormatTest(unittest.TestCase):
    def setUp(self):
        self.data = bi.build(FIXTURE_CHUNKS)
        self.index = bi.Index(self.data)

    def test_header_fields(self):
        d = self.data
        self.assertEqual(d[0:4], b"CYX1")
        self.assertEqual(d[0:4], bytes([0x43, 0x59, 0x58, 0x31]))
        version, flags, dim, dtype, chunks, terms, avgdl_q8, crc, reserved = \
            struct.unpack_from("<HHHHIIIII", d, 4)
        self.assertEqual((version, flags, dim, dtype, reserved), (1, 0, 0, 0, 0))
        self.assertEqual(chunks, 6)
        self.assertEqual(terms, 38)
        total_dl = sum(self.index.doc(i)[2] for i in range(chunks))
        self.assertEqual(total_dl, 52)
        self.assertEqual(avgdl_q8, round(52 * 256 / 6))
        self.assertEqual(d[32:96], bytes(64))
        self.assertEqual(d[144:160], bytes(16))

    def test_sections_contiguous_and_aligned(self):
        sections = self.index.sections
        self.assertEqual(sections[0][0], bi.HEADER_SIZE)
        for (off, length), (next_off, _) in zip(sections, sections[1:]):
            self.assertEqual(off % 4, 0)
            self.assertEqual(next_off, (off + length + 3) & ~3)
        self.assertEqual(sections[-1][0] + sections[-1][1], len(self.data))
        self.assertEqual(sections[bi.VECTORS][1], 0)
        self.assertEqual(sections[bi.RESERVED][1], 0)

    def test_terms_sorted_by_hash(self):
        hashes = [h for h, _off, _df, _r in self.index.terms()]
        self.assertEqual(hashes, sorted(set(hashes)))

    def test_postings(self):
        expected = {}
        for chunk_id, (title, body) in enumerate(FIXTURE_CHUNKS):
            for token in bi.tokenize(title + "\n" + body):
                tfs = expected.setdefault(bi.fnv1a32(token), {})
                tfs[chunk_id] = tfs.get(chunk_id, 0) + 1
        next_off = 0
        for h, off, df, reserved in self.index.terms():
            self.assertEqual(reserved, 0)
            self.assertEqual(off, next_off)  # in term order, no padding
            postings = self.index.postings(off, df)
            self.assertEqual(dict(postings), expected[h])
            self.assertEqual([c for c, _tf in postings], sorted(expected[h]))
            next_off += df * bi.POSTING.size
        self.assertEqual(next_off, self.index.sections[bi.POSTINGS][1])
        radio = [t for t in self.index.terms() if t[0] == bi.fnv1a32(b"radio")][0]
        self.assertEqual(self.index.postings(radio[1], radio[2]), [(0, 2), (1, 1), (5, 2)])

    def test_docs_and_text(self):
        text = self.index.section(bi.TEXT)
        for chunk_id, (title, body) in enumerate(FIXTURE_CHUNKS):
            off, length, dl, flags = self.index.doc(chunk_id)
            stored = (title + "\n" + body).encode("utf-8")
            self.assertEqual(text[off:off + length], stored)
            self.assertEqual(text[off + length], 0)
            self.assertEqual(dl, len(bi.tokenize(stored)))
            self.assertEqual(flags, 0)
        self.assertEqual(self.index.title(4), "Sound é")

    def test_crc(self):
        crc = struct.unpack_from("<I", self.data, 24)[0]
        self.assertEqual(crc, zlib.crc32(self.data[28:]) & 0xFFFFFFFF)
        damaged = bytearray(self.data)
        damaged[-8] ^= 0x20
        with self.assertRaisesRegex(bi.FormatError, "crc"):
            bi.Index(bytes(damaged))

    def test_rejects_damage(self):
        with self.assertRaises(bi.FormatError):
            bi.Index(self.data[:100])
        with self.assertRaises(bi.FormatError):
            bi.Index(self.data[:-4])
        with self.assertRaises(bi.FormatError):
            bi.Index(b"1YXC" + self.data[4:])
        with self.assertRaises(bi.FormatError):
            bi.Index(self.data[:4] + b"\x02\x00" + self.data[6:])

    def test_scores_match_c_fixture(self):
        # tests/test_search_index.c expects these exact values from the C reader.
        self.assertEqual(self.index.query("radio"), [(5, 66035), (1, 58262), (0, 41711)])
        self.assertEqual(self.index.query("Antenna, RADIO?"), [(5, 164125), (1, 144805), (0, 41711)])
        self.assertEqual(self.index.query("pwm"), [(4, 122090)])
        self.assertEqual(self.index.query("battery"), [(2, 113523), (3, 113523)])
        self.assertEqual(self.index.query("radio radio RADIO"), self.index.query("radio"))
        self.assertEqual(self.index.query("radio", top=1), [(5, 66035)])
        self.assertEqual(self.index.query("the and of"), [])

    def test_empty_corpus(self):
        index = bi.Index(bi.build([]))
        self.assertEqual((index.chunk_count, index.term_count, index.avgdl_q8), (0, 0, 0))
        self.assertEqual(index.query("anything"), [])

    def test_deterministic(self):
        self.assertEqual(bi.build(FIXTURE_CHUNKS), self.data)


class CorpusRankingTest(unittest.TestCase):
    """Four documents where the right order is plain from the text."""

    FILES = {
        "battery.md": "# Battery\n\nThe battery charges over USB-C. A full battery lasts a day; "
                      "the battery gauge reads the cells over i2c.",
        "wifi.md": "# Wi-Fi\n\nJoin a network from the connection settings. The radio scans for "
                   "networks and the password is stored on the card.",
        "keyboard.txt": "Keyboard notes\n\nThe keyboard talks i2c to the Pico. Keys repeat when held.",
        "sound/speaker.md": "# Sound\n\nThe speaker plays PWM tones. Battery level does not change the volume.",
    }

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.corpus = os.path.join(self.tmp.name, "corpus")
        write_tree(self.corpus, self.FILES)
        self.out = os.path.join(self.tmp.name, "a.idx")
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(bi.main(["--corpus", self.corpus, "--out", self.out]), 0)
        self.index = bi.load(self.out)
        # Sorted path order: battery.md, keyboard.txt, sound/speaker.md, wifi.md.
        self.titles = [self.index.title(i) for i in range(self.index.chunk_count)]

    def tearDown(self):
        self.tmp.cleanup()

    def ranked_titles(self, query):
        return [self.titles[c] for c, _s in self.index.query(query)]

    def test_chunk_order_and_titles(self):
        self.assertEqual(self.titles, ["Battery", "keyboard.txt", "Sound", "Wi-Fi"])

    def test_rankings(self):
        self.assertEqual(self.ranked_titles("battery"), ["Battery", "Sound"])
        self.assertEqual(self.ranked_titles("i2c"), ["keyboard.txt", "Battery"])
        self.assertEqual(self.ranked_titles("wifi password"), ["Wi-Fi"])
        self.assertEqual(self.ranked_titles("pwm speaker volume"), ["Sound"])
        self.assertEqual(self.ranked_titles("the of and"), [])

    def test_query_command(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(bi.main(["--index", self.out, "--query", "battery", "--top", "1"]), 0)
        lines = out.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        chunk_id, score, title = lines[0].split("\t")
        self.assertEqual((chunk_id, title), ("0", "Battery"))
        self.assertGreater(float(score), 0.0)

    def test_query_command_rejects_bad_index(self):
        with open(self.out, "r+b") as f:
            f.write(b"XXXX")
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(bi.main(["--index", self.out, "--query", "battery"]), 1)


# ---- round 2: vectors and hybrid search ----------------------------------

# One normalized int8 vector per FIXTURE_CHUNKS chunk; tests/test_search_index.c
# builds the same index (chunk_vectors) and expects the same scores.
FIXTURE_VECTORS = [[127, 0, 0, 0], [110, 63, 0, 0], [0, 127, 0, 0], [0, 0, 127, 0], [-127, 0, 0, 0],
                   [90, 90, 0, 0]]
VECTOR_QUERY = [127, 0, 0, 0]


def fake_endpoint(dim=3, count_delta=0, dims=None):
    """A post_json stand-in: records each request and answers with one vector
    per input (count_delta more or fewer), the n-th vector over all requests
    of length dims[n] if given."""
    calls = []
    answered = []

    def post(url, payload):
        calls.append((url, payload))
        data = []
        for k in range(len(payload["input"]) + count_delta):
            n = len(calls) * 100 + k
            length = dims[len(answered)] if dims is not None else dim
            answered.append(length)
            data.append({"object": "embedding", "index": k, "embedding": [float(n + 1)] + [0.5] * (length - 1)})
        return {"object": "list", "data": data, "model": payload["model"]}

    return post, calls


class QuantizeTest(unittest.TestCase):
    def test_normalized_then_quantized(self):
        self.assertEqual(bi.quantize([3.0, 4.0]), [76, 102])  # 0.6 and 0.8 of 127
        self.assertEqual(bi.quantize([0.0, -2.0, 0.0]), [0, -127, 0])
        # Scale does not matter, direction does.
        self.assertEqual(bi.quantize([1e-6, 2e-6]), bi.quantize([1000.0, 2000.0]))
        for vector in ([0.1, 0.2, -0.3], [5.0] * 768, [-1.0, 1e-9]):
            q = bi.quantize(vector)
            self.assertTrue(all(-127 <= v <= 127 for v in q))
            self.assertAlmostEqual(sum(v * v for v in q) ** 0.5, 127.0, delta=len(q) ** 0.5)

    def test_matches_c_client(self):
        # tests/test_embed_client.c parses the same values to the same bytes.
        real = json.loads("[0.0123, -0.4567, 1.5e-3, -2.25E-2, 0.3333333333333333, -0.0, 7.0e-1, -0.125]")
        self.assertEqual(bi.quantize(real), [2, -64, 0, -3, 47, 0, 98, -17])
        # Norm 254: exact halves, rounded to even.
        self.assertEqual(bi.quantize(json.loads("[-1e0, 0.5E1, -5.9e+1, 247.0]")), [0, 2, -30, 124])
        self.assertEqual(bi.quantize([3.0, 4.0]), [76, 102])

    def test_stored_vectors_requantize_to_themselves(self):
        for vector in FIXTURE_VECTORS + [[2, -64, 0, -3, 47, 0, 98, -17]]:
            self.assertEqual(bi.quantize([float(v) for v in vector]), vector)

    def test_unnormalizable(self):
        for vector in ([], [0.0, 0.0], [1e200, 1e200]):
            with self.assertRaises(ValueError):
                bi.quantize(vector)

    def test_vector_q16(self):
        self.assertEqual(bi.vector_q16(127 * 127), 65536)
        self.assertEqual(bi.vector_q16(-127 * 127), -65536)
        self.assertEqual(bi.vector_q16(0), 0)
        self.assertEqual(bi.vector_q16(90 * 127), 46443)  # 46443.47
        self.assertEqual(bi.vector_q16(1 << 30), 0x7FFFFFFF)


class VectorFormatTest(unittest.TestCase):
    def setUp(self):
        self.plain = bi.build(FIXTURE_CHUNKS)
        self.data = bi.build(FIXTURE_CHUNKS, FIXTURE_VECTORS, "test-model")
        self.index = bi.Index(self.data)

    def test_round1_bytes_unchanged(self):
        # The digest round 1's builder gave for this fixture.
        self.assertEqual(hashlib.sha256(self.plain).hexdigest(),
                         "89fa1e81cb6f37811dc4f999446bff7a0eb987986f5a309bcdd32ea807004e5b")
        self.assertEqual(bi.build(FIXTURE_CHUNKS, None, "ignored"), self.plain)

    def test_header(self):
        flags, dim, dtype = struct.unpack_from("<HHH", self.data, 6)
        self.assertEqual((flags, dim, dtype), (bi.FLAG_NORMALIZED | bi.FLAG_INT8, 4, bi.DTYPE_INT8))
        self.assertEqual(self.data[32:96], b"test-model" + bytes(54))
        self.assertEqual(self.index.model_id, b"test-model")
        self.assertEqual(struct.unpack_from("<I", self.data, 24)[0], zlib.crc32(self.data[28:]) & 0xFFFFFFFF)

    def test_vectors_section(self):
        off, length = self.index.sections[bi.VECTORS]
        self.assertEqual(off, len(self.plain))  # right after the text, which ends aligned here
        self.assertEqual(length, 6 * 4)
        self.assertEqual(self.data[off:off + 4], bytes([127, 0, 0, 0]))
        self.assertEqual(self.data[off + 16:off + 20], bytes([0x81, 0, 0, 0]))  # -127
        self.assertEqual([self.index.vector(c) for c in range(6)], FIXTURE_VECTORS)
        self.assertEqual(self.index.sections[bi.RESERVED], (off + length, 0))
        # The BM25 sections are the round 1 bytes.
        self.assertEqual(self.data[bi.HEADER_SIZE:off], self.plain[bi.HEADER_SIZE:])

    def test_unaligned_vectors_padded(self):
        data = bi.build(FIXTURE_CHUNKS, [[1, 2, 3]] * 6, "m")
        index = bi.Index(data)
        self.assertEqual(index.sections[bi.VECTORS][1], 18)
        self.assertEqual(index.sections[bi.RESERVED][0], index.sections[bi.VECTORS][0] + 20)
        self.assertEqual(len(data), index.sections[bi.RESERVED][0])

    def test_model_id_truncated(self):
        data = bi.build(FIXTURE_CHUNKS, FIXTURE_VECTORS, "m" * 80)
        self.assertEqual(data[32:96], b"m" * 63 + b"\0")
        # Never cut inside a character.
        data = bi.build(FIXTURE_CHUNKS, FIXTURE_VECTORS, "m" * 62 + "\u00e9")
        self.assertEqual(data[32:96], b"m" * 62 + bytes(2))

    def test_bad_vectors_refused(self):
        with self.assertRaises(ValueError):
            bi.build(FIXTURE_CHUNKS, FIXTURE_VECTORS[:5])
        with self.assertRaises(ValueError):
            bi.build(FIXTURE_CHUNKS, FIXTURE_VECTORS[:5] + [[1, 2, 3]])
        with self.assertRaises(ValueError):
            bi.build(FIXTURE_CHUNKS, [[]] * 6)
        with self.assertRaises(ValueError):
            bi.build(FIXTURE_CHUNKS, [[128, 0, 0, 0]] * 6)

    def test_reader_checks_vectors_section(self):
        damaged = bytearray(self.data)
        struct.pack_into("<H", damaged, 8, 5)  # dim no longer matches the section
        struct.pack_into("<I", damaged, 24, zlib.crc32(bytes(damaged[28:])) & 0xFFFFFFFF)
        with self.assertRaisesRegex(bi.FormatError, "vectors"):
            bi.Index(bytes(damaged))

    def test_scores_match_c_fixture(self):
        # tests/test_search_index.c expects these exact values from the C reader.
        self.assertEqual(self.index.query_vector(VECTOR_QUERY),
                         [(0, 65536), (1, 56763), (5, 46443), (2, 0), (3, 0), (4, -65536)])
        self.assertEqual(self.index.query_vector([0, 90, 0, 90])[:3], [(2, 46443), (5, 32912), (1, 23039)])
        self.assertEqual(self.index.query_vector(VECTOR_QUERY, top=2), [(0, 65536), (1, 56763)])
        # BM25 ranks 5, 1, 0 for "radio"; 5 is 1st and 3rd, 0 is 3rd and 1st.
        self.assertEqual(self.index.query("radio"), [(5, 66035), (1, 58262), (0, 41711)])
        self.assertEqual(self.index.query_hybrid("radio", VECTOR_QUERY),
                         [(0, 2115), (5, 2115), (1, 2114), (2, 1024), (3, 1008), (4, 993)])
        self.assertEqual(self.index.query_hybrid("the and of", VECTOR_QUERY)[:3], [(0, 1074), (1, 1057), (5, 1040)])

    def test_rrf_is_exact(self):
        # 1/61 + 1/63 > 2/62, though each term rounded to Q16 first ties them.
        fused = bi.rrf_fuse([(9, 0), (8, 0), (7, 0)], [(7, 0), (8, 0), (9, 0)])
        self.assertEqual([c for c, _s in fused], [7, 9, 8])
        self.assertEqual(bi.rrf_fuse([], []), [])
        self.assertEqual(bi.rrf_fuse([(4, 1)]), [(4, 1074)])

    def test_dim_and_missing_vectors(self):
        with self.assertRaises(ValueError):
            self.index.query_vector([127, 0, 0])
        plain = bi.Index(self.plain)
        with self.assertRaises(ValueError):
            plain.query_vector(VECTOR_QUERY)
        # Without vectors, hybrid is BM25.
        self.assertEqual(plain.query_hybrid("Antenna, RADIO?", VECTOR_QUERY), plain.query("Antenna, RADIO?"))


class EmbedTest(unittest.TestCase):
    CHUNKS = [("T%d" % i, "body %d" % i) for i in range(70)]

    def test_batches_of_32_in_order(self):
        post, calls = fake_endpoint(dim=3)
        vectors = bi.embed_texts([t + "\n" + b for t, b in self.CHUNKS], "http://wang.local:11434/", "nomic", post)
        self.assertEqual([len(p["input"]) for _u, p in calls], [32, 32, 6])
        self.assertEqual({u for u, _p in calls}, {"http://wang.local:11434/v1/embeddings"})
        self.assertEqual({p["model"] for _u, p in calls}, {"nomic"})
        self.assertEqual(calls[1][1]["input"][0], "T32\nbody 32")
        self.assertEqual(len(vectors), 70)
        self.assertEqual(vectors[33][0], 202.0)  # second request, second item: input order kept

    def test_wrong_count_rejected(self):
        for delta in (-1, 1):
            post, _calls = fake_endpoint(count_delta=delta)
            with self.assertRaisesRegex(ValueError, "embeddings"):
                bi.embed_texts(["a", "b"], "http://h", "m", post)

    def test_wrong_dimension_rejected(self):
        post, _calls = fake_endpoint(dims=[3, 2])
        with self.assertRaisesRegex(ValueError, "values"):
            bi.embed_texts(["a", "b"], "http://h", "m", post)
        # Nor may a later batch change the dimension.
        post, _calls = fake_endpoint(dims=[3] * 32 + [4])
        with self.assertRaisesRegex(ValueError, "values"):
            bi.embed_texts(["x"] * 33, "http://h", "m", post, batch=32)

    def test_malformed_replies_rejected(self):
        for reply in ({}, {"data": {}}, {"data": [{}]}, {"data": [{"embedding": []}]},
                      {"data": [{"embedding": ["1"]}]}, {"data": [{"embedding": [True]}]}, []):
            with self.assertRaises(ValueError):
                bi.embed_texts(["a"], "http://h", "m", lambda _u, _p, reply=reply: reply)

    def test_chunk_vectors_quantizes(self):
        post, _calls = fake_endpoint(dim=2)
        vectors = bi.chunk_vectors(self.CHUNKS[:2], "http://h", "m", post=post)
        self.assertEqual(vectors, [bi.quantize([101.0, 0.5]), bi.quantize([102.0, 0.5])])


class VectorCommandTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.corpus = os.path.join(self.tmp.name, "corpus")
        write_tree(self.corpus, CorpusRankingTest.FILES)
        self.out = os.path.join(self.tmp.name, "v.idx")
        self.vectors = os.path.join(self.tmp.name, "vectors.jsonl")
        # battery, keyboard, sound, wifi
        self.floats = [[0.9, 0.1, 0.0], [0.0, 1.0, 0.2], [0.3, 0.0, -0.4], [-1e-1, 2e-1, 9e-1]]

    def tearDown(self):
        self.tmp.cleanup()

    def write_vectors(self, rows):
        with open(self.vectors, "w", encoding="utf-8") as f:
            f.write("\n".join(json.dumps(r) for r in rows) + "\n")

    def build(self, *extra):
        with contextlib.redirect_stderr(io.StringIO()):
            return bi.main(["--corpus", self.corpus, "--out", self.out] + list(extra))

    def test_vectors_file(self):
        self.write_vectors(self.floats)
        self.assertEqual(self.build("--vectors", self.vectors, "--model", "nomic-embed-text"), 0)
        index = bi.load(self.out)
        self.assertEqual((index.dim, index.dtype, index.flags), (3, bi.DTYPE_INT8, 3))
        self.assertEqual(index.model_id, b"nomic-embed-text")
        self.assertEqual([index.vector(c) for c in range(4)], [bi.quantize(v) for v in self.floats])
        self.assertEqual(index.query_vector(bi.quantize([0.0, 0.1, 1.0]))[0][0], 3)

    def test_vectors_file_must_match_chunks(self):
        self.write_vectors(self.floats[:3])
        self.assertEqual(self.build("--vectors", self.vectors), 1)
        self.write_vectors(self.floats[:3] + [[1.0, 2.0]])
        self.assertEqual(self.build("--vectors", self.vectors), 1)
        with open(self.vectors, "w", encoding="utf-8") as f:
            f.write("[1, 2, 3]\n[NaN, 1, 1]\n[1, 1, 1]\n[1, 1, 1]\n")
        self.assertEqual(self.build("--vectors", self.vectors), 1)
        self.assertFalse(os.path.exists(self.out))

    def test_embed_url(self):
        post, calls = fake_endpoint(dim=5)
        original = bi.post_json
        bi.post_json = post
        try:
            self.assertEqual(self.build("--embed-url", "http://h:1"), 0)
        finally:
            bi.post_json = original
        index = bi.load(self.out)
        self.assertEqual((index.dim, index.model_id), (5, bi.EMBED_MODEL_DEFAULT.encode()))
        # Each chunk is embedded as it is stored: "title\nbody".
        self.assertEqual(calls[0][1]["input"], [index.text(c).decode("utf-8") for c in range(4)])

    def test_query_reports_three_rankings(self):
        self.write_vectors(self.floats)
        self.assertEqual(self.build("--vectors", self.vectors, "--model", "m"), 0)
        original = bi.post_json
        bi.post_json = lambda _url, payload: {"data": [{"embedding": [0.0, 0.1, 1.0]}]}
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                self.assertEqual(bi.main(["--index", self.out, "--query", "battery", "--embed-url", "http://h"]), 0)
        finally:
            bi.post_json = original
        text = out.getvalue()
        sections = re.split(r"^# (bm25|vector|hybrid)\n", text, flags=re.M)
        self.assertEqual(sections[0], "")
        self.assertEqual(sections[1::2], ["bm25", "vector", "hybrid"])
        bm25, vector, hybrid = [[line.split("\t") for line in body.splitlines()] for body in sections[2::2]]
        self.assertEqual([r[2] for r in bm25], ["Battery", "Sound"])
        self.assertEqual(vector[0][2], "Wi-Fi")
        self.assertEqual(len(vector), 4)
        self.assertEqual({r[2] for r in hybrid}, {"Battery", "Sound", "Wi-Fi", "keyboard.txt"})

    def test_options_conflict(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                bi.main(["--corpus", self.corpus, "--out", self.out, "--vectors", self.vectors,
                         "--embed-url", "http://h"])


if __name__ == "__main__":
    unittest.main()
