#include "ai/search_index.h"

#include <math.h>
#include <string.h>

#define BM25_K1 1.2
#define BM25_B 0.75

static const char *const stopwords[] = {SEARCH_STOPWORDS};

static uint16_t get_u16(const uint8_t *p) {
    return (uint16_t)(p[0] | (p[1] << 8));
}

static uint32_t get_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

uint32_t search_index_hash(const char *bytes, size_t length) {
    uint32_t hash = 2166136261u;
    size_t i;

    for (i = 0u; i < length; i++) {
        hash ^= (uint8_t)bytes[i];
        hash *= 16777619u;
    }
    return hash;
}

int search_index_is_stopword(const char *token, size_t length) {
    size_t low = 0u;
    size_t high = sizeof(stopwords) / sizeof(stopwords[0]);

    if (token == NULL) {
        return 0;
    }
    while (low < high) {
        size_t mid = low + (high - low) / 2u;
        const char *word = stopwords[mid];
        size_t word_length = strlen(word);
        size_t common = word_length < length ? word_length : length;
        int order = memcmp(token, word, common);

        if (order == 0) {
            order = length < word_length ? -1 : (length > word_length ? 1 : 0);
        }
        if (order == 0) {
            return 1;
        }
        if (order < 0) {
            high = mid;
        } else {
            low = mid + 1u;
        }
    }
    return 0;
}

/* CRC-32 (IEEE 802.3, as zlib.crc32), bitwise so it needs no table. */
static uint32_t crc32_update(uint32_t crc, const uint8_t *data, size_t length) {
    size_t i;
    int bit;

    crc = ~crc;
    for (i = 0u; i < length; i++) {
        crc ^= data[i];
        for (bit = 0; bit < 8; bit++) {
            crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
        }
    }
    return ~crc;
}

/* ---- file access ------------------------------------------------------- */

static search_status_t read_at(search_index_t *index, uint32_t offset, void *destination, size_t length) {
    if ((uint64_t)offset + length > index->file_size) {
        return SEARCH_ERR_FORMAT;
    }
    if (length == 0u) {
        return SEARCH_OK;
    }
    if (fseek(index->file, (long)offset, SEEK_SET) != 0 ||
        fread(destination, 1u, length, index->file) != length) {
        return SEARCH_ERR_IO;
    }
    index->bytes_read += (uint32_t)length;
    return SEARCH_OK;
}

/* Read from a section, never past its declared length. */
static search_status_t section_read(search_index_t *index, unsigned section, uint64_t offset,
                                    void *destination, size_t length) {
    const search_section_t *s = &index->sections[section];

    if (offset > s->len || length > s->len - offset) {
        return SEARCH_ERR_FORMAT;
    }
    return read_at(index, s->off + (uint32_t)offset, destination, length);
}

/* ---- open -------------------------------------------------------------- */

static search_status_t check_sections(const search_index_t *index) {
    uint64_t expected = SEARCH_INDEX_HEADER_SIZE;
    uint64_t element_size;
    unsigned i;

    /* Contiguous, in order, each starting on the 4-byte boundary after the
     * one before, and the last one ending the file. */
    for (i = 0u; i < SEARCH_INDEX_SECTION_COUNT; i++) {
        uint64_t end = (uint64_t)index->sections[i].off + index->sections[i].len;

        if (index->sections[i].off != expected || end > index->file_size) {
            return SEARCH_ERR_FORMAT;
        }
        expected = (end + 3u) & ~(uint64_t)3u;
    }
    if ((uint64_t)index->sections[SEARCH_SECTION_RESERVED].off + index->sections[SEARCH_SECTION_RESERVED].len !=
        index->file_size) {
        return SEARCH_ERR_FORMAT;
    }

    if (index->sections[SEARCH_SECTION_TERMS].len != (uint64_t)index->term_count * SEARCH_TERM_ENTRY_SIZE ||
        index->sections[SEARCH_SECTION_DOCS].len != (uint64_t)index->chunk_count * SEARCH_DOC_ENTRY_SIZE ||
        index->sections[SEARCH_SECTION_POSTINGS].len % SEARCH_POSTING_SIZE != 0u) {
        return SEARCH_ERR_FORMAT;
    }

    switch (index->dtype) {
    case 0u:
        element_size = 0u;
        if (index->dim != 0u) {
            return SEARCH_ERR_FORMAT;
        }
        break;
    case 1u:
        element_size = 1u;
        break;
    case 2u:
        element_size = 4u;
        break;
    default:
        return SEARCH_ERR_FORMAT;
    }
    if (index->sections[SEARCH_SECTION_VECTORS].len !=
        (uint64_t)index->chunk_count * index->dim * element_size) {
        return SEARCH_ERR_FORMAT;
    }

    if (index->chunk_count > 0u && index->avgdl_q8 == 0u) {
        return SEARCH_ERR_FORMAT;
    }
    return SEARCH_OK;
}

static search_status_t check_crc(search_index_t *index, uint32_t expected) {
    uint32_t offset = 28u;
    uint32_t crc = 0u;

    while (offset < index->file_size) {
        size_t length = index->file_size - offset;
        search_status_t status;

        if (length > sizeof(index->io)) {
            length = sizeof(index->io);
        }
        status = read_at(index, offset, index->io, length);
        if (status != SEARCH_OK) {
            return status;
        }
        crc = crc32_update(crc, index->io, length);
        offset += (uint32_t)length;
    }
    return crc == expected ? SEARCH_OK : SEARCH_ERR_FORMAT;
}

static search_status_t load_header(search_index_t *index) {
    const uint8_t *header = index->io;
    search_status_t status;
    unsigned i;

    if (index->file_size < SEARCH_INDEX_HEADER_SIZE) {
        return SEARCH_ERR_FORMAT;
    }
    status = read_at(index, 0u, index->io, SEARCH_INDEX_HEADER_SIZE);
    if (status != SEARCH_OK) {
        return status;
    }
    if (memcmp(header, SEARCH_INDEX_MAGIC, 4u) != 0 || get_u16(header + 4) != SEARCH_INDEX_VERSION) {
        return SEARCH_ERR_FORMAT;
    }
    index->flags = get_u16(header + 6);
    index->dim = get_u16(header + 8);
    index->dtype = get_u16(header + 10);
    index->chunk_count = get_u32(header + 12);
    index->term_count = get_u32(header + 16);
    index->avgdl_q8 = get_u32(header + 20);
    for (i = 0u; i < SEARCH_INDEX_SECTION_COUNT; i++) {
        index->sections[i].off = get_u32(header + 96 + 8u * i);
        index->sections[i].len = get_u32(header + 100 + 8u * i);
    }
    status = check_sections(index);
    if (status != SEARCH_OK) {
        return status;
    }
    return check_crc(index, get_u32(header + 24));
}

search_status_t search_index_open(search_index_t *index, const char *path) {
    long size;
    search_status_t status;

    if (index == NULL || path == NULL) {
        return SEARCH_ERR_ARGUMENT;
    }
    memset(index, 0, sizeof(*index));
    index->file = fopen(path, "rb");
    if (index->file == NULL) {
        return SEARCH_ERR_IO;
    }
    /* Reads are already in SEARCH_IO_CHUNK pieces: stdio need not buffer. */
    setvbuf(index->file, NULL, _IONBF, 0u);

    status = SEARCH_ERR_IO;
    if (fseek(index->file, 0L, SEEK_END) == 0 && (size = ftell(index->file)) >= 0L) {
        if ((unsigned long)size > UINT32_MAX) {
            status = SEARCH_ERR_TOO_LARGE;
        } else {
            index->file_size = (uint32_t)size;
            status = load_header(index);
        }
    }
    if (status != SEARCH_OK) {
        search_index_close(index);
    }
    return status;
}

void search_index_close(search_index_t *index) {
    if (index == NULL) {
        return;
    }
    if (index->file != NULL) {
        fclose(index->file);
    }
    memset(index, 0, sizeof(*index));
}

/* ---- query ------------------------------------------------------------- */

typedef struct {
    uint32_t hashes[SEARCH_MAX_QUERY_TERMS];
    size_t count;
    int overflow;
} query_terms_t;

static void add_token(query_terms_t *terms, const char *token, size_t length) {
    uint32_t hash;
    size_t i;

    if (length < SEARCH_TOKEN_MIN || search_index_is_stopword(token, length)) {
        return;
    }
    hash = search_index_hash(token, length);
    for (i = 0u; i < terms->count; i++) {
        if (terms->hashes[i] == hash) {
            return; /* a repeated term counts once */
        }
    }
    if (terms->count == SEARCH_MAX_QUERY_TERMS) {
        terms->overflow = 1;
        return;
    }
    terms->hashes[terms->count++] = hash;
}

/*
 * Bytes >= 0x80 are dropped without ending a token, A-Z is folded, and a run
 * of [a-z0-9_] is cut into SEARCH_TOKEN_MAX pieces.
 */
static void tokenize(query_terms_t *terms, const char *text) {
    char token[SEARCH_TOKEN_MAX];
    size_t length = 0u;
    const unsigned char *p;

    memset(terms, 0, sizeof(*terms));
    for (p = (const unsigned char *)text; *p != '\0'; p++) {
        unsigned char c = *p;

        if (c >= 0x80u) {
            continue;
        }
        if (c >= 'A' && c <= 'Z') {
            c = (unsigned char)(c - 'A' + 'a');
        }
        if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_') {
            token[length++] = (char)c;
            if (length == SEARCH_TOKEN_MAX) {
                add_token(terms, token, length);
                length = 0u;
            }
        } else {
            add_token(terms, token, length);
            length = 0u;
        }
    }
    add_token(terms, token, length);
}

/* Binary search of the term table.  *found is 0 when hash is not there. */
static search_status_t find_term(search_index_t *index, uint32_t hash, search_cursor_t *cursor, int *found) {
    uint32_t low = 0u;
    uint32_t high = index->term_count;
    uint8_t entry[SEARCH_TERM_ENTRY_SIZE];

    *found = 0;
    while (low < high) {
        uint32_t mid = low + (high - low) / 2u;
        uint32_t entry_hash;
        search_status_t status =
            section_read(index, SEARCH_SECTION_TERMS, (uint64_t)mid * SEARCH_TERM_ENTRY_SIZE, entry, sizeof(entry));

        if (status != SEARCH_OK) {
            return status;
        }
        entry_hash = get_u32(entry);
        if (entry_hash == hash) {
            uint32_t postings_off = get_u32(entry + 4);
            uint16_t df = get_u16(entry + 8);

            if (df == 0u || df > index->chunk_count ||
                (uint64_t)postings_off + (uint64_t)df * SEARCH_POSTING_SIZE >
                    index->sections[SEARCH_SECTION_POSTINGS].len) {
                return SEARCH_ERR_FORMAT;
            }
            memset(cursor, 0, sizeof(*cursor));
            cursor->postings_off = postings_off;
            cursor->df = df;
            cursor->idf = log(1.0 + ((double)index->chunk_count - df + 0.5) / (df + 0.5));
            *found = 1;
            return SEARCH_OK;
        }
        if (entry_hash < hash) {
            low = mid + 1u;
        } else {
            high = mid;
        }
    }
    return SEARCH_OK;
}

/*
 * Move a cursor to its next posting, refilling its slice of io (slice_count
 * postings at slot slot) when it runs dry.  Chunk ids must rise strictly
 * within a list and name a chunk; tf must be nonzero.
 */
static search_status_t advance(search_index_t *index, search_cursor_t *cursor, size_t slot, size_t slice_count) {
    uint8_t *slice = index->io + slot * slice_count * SEARCH_POSTING_SIZE;
    const uint8_t *posting;
    uint32_t chunk_id;
    uint16_t tf;

    if (cursor->next == cursor->df) {
        cursor->has_head = 0u;
        return SEARCH_OK;
    }
    if (cursor->cursor == cursor->buffered) {
        size_t count = (size_t)(cursor->df - cursor->next);
        search_status_t status;

        if (count > slice_count) {
            count = slice_count;
        }
        status = section_read(index, SEARCH_SECTION_POSTINGS,
                              (uint64_t)cursor->postings_off + (uint64_t)cursor->next * SEARCH_POSTING_SIZE, slice,
                              count * SEARCH_POSTING_SIZE);
        if (status != SEARCH_OK) {
            return status;
        }
        cursor->buffered = (uint16_t)count;
        cursor->cursor = 0u;
    }
    posting = slice + (size_t)cursor->cursor * SEARCH_POSTING_SIZE;
    chunk_id = get_u32(posting);
    tf = get_u16(posting + 4);
    if (chunk_id >= index->chunk_count || tf == 0u || (cursor->next > 0u && chunk_id <= cursor->last_chunk_id)) {
        return SEARCH_ERR_FORMAT;
    }
    cursor->cursor++;
    cursor->next++;
    cursor->chunk_id = chunk_id;
    cursor->last_chunk_id = chunk_id;
    cursor->tf = tf;
    cursor->has_head = 1u;
    return SEARCH_OK;
}

/* A chunk's document entry, through a cache that reads forward. */
static search_status_t doc_entry(search_index_t *index, uint32_t chunk_id, const uint8_t **entry) {
    if (chunk_id >= index->chunk_count) {
        return SEARCH_ERR_FORMAT;
    }
    if (index->doc_cache_count == 0u || chunk_id < index->doc_cache_first ||
        chunk_id - index->doc_cache_first >= index->doc_cache_count) {
        uint32_t count = index->chunk_count - chunk_id;
        search_status_t status;

        if (count > SEARCH_DOC_CACHE_ENTRIES) {
            count = SEARCH_DOC_CACHE_ENTRIES;
        }
        index->doc_cache_count = 0u;
        status = section_read(index, SEARCH_SECTION_DOCS, (uint64_t)chunk_id * SEARCH_DOC_ENTRY_SIZE,
                              index->doc_cache, (size_t)count * SEARCH_DOC_ENTRY_SIZE);
        if (status != SEARCH_OK) {
            return status;
        }
        index->doc_cache_first = chunk_id;
        index->doc_cache_count = count;
    }
    *entry = index->doc_cache + (size_t)(chunk_id - index->doc_cache_first) * SEARCH_DOC_ENTRY_SIZE;
    return SEARCH_OK;
}

/*
 * One term's BM25 contribution in 1/65536ths.  Written term for term as
 * bm25_q16 in build_index.py so the two round alike.
 */
static int32_t bm25_q16(double idf, uint32_t tf, uint32_t dl, uint32_t avgdl_q8) {
    double avgdl = avgdl_q8 / 256.0;
    double norm = BM25_K1 * (1.0 - BM25_B + BM25_B * dl / avgdl);
    double part = idf * (tf * (BM25_K1 + 1.0)) / (tf + norm);

    return (int32_t)floor(part * 65536.0 + 0.5);
}

/* Insert into hits[0..*count), kept best first; equal scores keep the lower
 * chunk id first, which holds because chunks arrive in rising id order. */
static void keep_hit(search_hit_t *hits, size_t capacity, size_t *count, uint32_t chunk_id, int32_t score) {
    size_t at = *count;

    while (at > 0u && hits[at - 1u].score_q16 < score) {
        at--;
    }
    if (at >= capacity) {
        return;
    }
    if (*count < capacity) {
        (*count)++;
    }
    memmove(&hits[at + 1u], &hits[at], (*count - 1u - at) * sizeof(hits[0]));
    hits[at].chunk_id = chunk_id;
    hits[at].score_q16 = score;
}

search_status_t search_index_query(search_index_t *index, const char *query,
                                   search_hit_t *hits, size_t max_hits, size_t *hit_count) {
    query_terms_t terms;
    size_t term_count = 0u;
    size_t slice_count;
    size_t i;
    search_status_t status;

    if (hit_count != NULL) {
        *hit_count = 0u;
    }
    if (index == NULL || query == NULL || hits == NULL || hit_count == NULL || index->file == NULL) {
        return SEARCH_ERR_ARGUMENT;
    }
    if (max_hits > SEARCH_MAX_HITS) {
        max_hits = SEARCH_MAX_HITS;
    }

    tokenize(&terms, query);
    if (terms.overflow) {
        return SEARCH_ERR_TOO_LARGE;
    }
    for (i = 0u; i < terms.count; i++) {
        int found;

        status = find_term(index, terms.hashes[i], &index->cursors[term_count], &found);
        if (status != SEARCH_OK) {
            return status;
        }
        term_count += (size_t)found;
    }
    if (term_count == 0u || max_hits == 0u) {
        return SEARCH_OK;
    }

    /* The terms' postings lists are merged by chunk id, each read forward
     * through an equal share of io. */
    slice_count = sizeof(index->io) / term_count / SEARCH_POSTING_SIZE;
    for (i = 0u; i < term_count; i++) {
        status = advance(index, &index->cursors[i], i, slice_count);
        if (status != SEARCH_OK) {
            return status;
        }
    }
    for (;;) {
        uint32_t chunk_id = 0u;
        int any = 0;
        const uint8_t *entry;
        uint32_t dl;
        int32_t score = 0;

        for (i = 0u; i < term_count; i++) {
            if (index->cursors[i].has_head && (!any || index->cursors[i].chunk_id < chunk_id)) {
                chunk_id = index->cursors[i].chunk_id;
                any = 1;
            }
        }
        if (!any) {
            break;
        }
        status = doc_entry(index, chunk_id, &entry);
        if (status != SEARCH_OK) {
            return status;
        }
        dl = get_u16(entry + 8);
        for (i = 0u; i < term_count; i++) {
            search_cursor_t *cursor = &index->cursors[i];

            if (cursor->has_head && cursor->chunk_id == chunk_id) {
                score += bm25_q16(cursor->idf, cursor->tf, dl, index->avgdl_q8);
                status = advance(index, cursor, i, slice_count);
                if (status != SEARCH_OK) {
                    return status;
                }
            }
        }
        if (score > 0) {
            keep_hit(hits, max_hits, hit_count, chunk_id, score);
        }
    }
    return SEARCH_OK;
}

/* ---- vectors ----------------------------------------------------------- */

static int32_t dot_i8(const int8_t *a, const uint8_t *b, size_t count) {
    int32_t sum = 0;
    size_t i;

    for (i = 0u; i < count; i++) {
        sum += (int32_t)a[i] * (int8_t)b[i];
    }
    return sum;
}

/* floor(numerator / denominator) for a positive denominator. */
static int64_t floor_div(int64_t numerator, int64_t denominator) {
    int64_t quotient = numerator / denominator;

    if (numerator % denominator != 0 && numerator < 0) {
        quotient--;
    }
    return quotient;
}

/* dot x 65536 / (127 x 127), rounded half up, in integers; vector_q16 in
 * build_index.py.  Never exactly half way: 127 x 127 is odd.  A file or query
 * whose vectors are not normalized can pass int32, so the score is held to
 * it. */
static int32_t vector_q16(int32_t dot) {
    int64_t q16 = floor_div((int64_t)dot * 2 * 65536 + 127 * 127, 2 * 127 * 127);

    if (q16 > INT32_MAX) {
        return INT32_MAX;
    }
    if (q16 < INT32_MIN) {
        return INT32_MIN;
    }
    return (int32_t)q16;
}

search_status_t search_index_query_vectors(search_index_t *index, const int8_t *query,
                                           size_t dim, search_hit_t *hits, size_t max_hits,
                                           size_t *hit_count) {
    const search_section_t *vectors;
    uint64_t offset = 0u;
    uint32_t chunk_id = 0u;
    size_t done = 0u; /* of the current chunk's vector */
    int32_t dot = 0;

    if (hit_count != NULL) {
        *hit_count = 0u;
    }
    if (index == NULL || query == NULL || hits == NULL || hit_count == NULL || index->file == NULL) {
        return SEARCH_ERR_ARGUMENT;
    }
    if (index->dim == 0u) {
        return SEARCH_ERR_NO_VECTORS;
    }
    if (dim != index->dim) {
        return SEARCH_ERR_DIM;
    }
    vectors = &index->sections[SEARCH_SECTION_VECTORS];
    /* Open held the section to chunk_count x dim elements; only int8 is
     * scored. */
    if (index->dtype != SEARCH_DTYPE_INT8 || vectors->len != (uint64_t)index->chunk_count * dim) {
        return SEARCH_ERR_FORMAT;
    }
    if (max_hits > SEARCH_MAX_HITS) {
        max_hits = SEARCH_MAX_HITS;
    }

    /* One pass, front to back.  A vector may straddle two reads. */
    while (offset < vectors->len) {
        size_t length = vectors->len - offset > sizeof(index->io) ? sizeof(index->io)
                                                                  : (size_t)(vectors->len - offset);
        size_t at = 0u;
        search_status_t status = section_read(index, SEARCH_SECTION_VECTORS, offset, index->io, length);

        if (status != SEARCH_OK) {
            return status;
        }
        while (at < length) {
            size_t take = dim - done < length - at ? dim - done : length - at;

            dot += dot_i8(query + done, index->io + at, take);
            done += take;
            at += take;
            if (done == dim) {
                keep_hit(hits, max_hits, hit_count, chunk_id, vector_q16(dot));
                chunk_id++;
                done = 0u;
                dot = 0;
            }
        }
        offset += length;
    }
    return SEARCH_OK;
}

search_status_t search_index_read_vector_block(search_index_t *index, uint32_t first_chunk_id, uint8_t *buf,
                                               size_t capacity, size_t *chunk_count) {
    uint64_t remaining;
    uint64_t take;
    search_status_t status;

    if (chunk_count != NULL) {
        *chunk_count = 0u;
    }
    if (index == NULL || buf == NULL || chunk_count == NULL || index->file == NULL) {
        return SEARCH_ERR_ARGUMENT;
    }
    if (index->dim == 0u) {
        return SEARCH_ERR_NO_VECTORS;
    }
    if (index->dtype != SEARCH_DTYPE_INT8 ||
        index->sections[SEARCH_SECTION_VECTORS].len != (uint64_t)index->chunk_count * index->dim) {
        return SEARCH_ERR_FORMAT;
    }
    if (first_chunk_id >= index->chunk_count) {
        return SEARCH_OK;
    }
    remaining = index->chunk_count - first_chunk_id;
    take = capacity / index->dim;
    if (take > remaining) {
        take = remaining;
    }
    if (take == 0u) {
        return SEARCH_ERR_TOO_LARGE; /* not even one vector fits */
    }
    status = section_read(index, SEARCH_SECTION_VECTORS, (uint64_t)first_chunk_id * index->dim, buf,
                          (size_t)(take * index->dim));
    if (status == SEARCH_OK) {
        *chunk_count = (size_t)take;
    }
    return status;
}

search_status_t search_index_score_block(const int8_t *query, size_t dim, const uint8_t *block,
                                         size_t block_length, uint32_t first_chunk_id,
                                         search_hit_t *hits, size_t max_hits, size_t *hit_count) {
    size_t vectors;
    size_t i;

    if (hit_count != NULL) {
        *hit_count = 0u;
    }
    if (query == NULL || hits == NULL || hit_count == NULL || (block == NULL && block_length != 0u)) {
        return SEARCH_ERR_ARGUMENT;
    }
    if (dim == 0u || block_length % dim != 0u) {
        return SEARCH_ERR_ARGUMENT;
    }
    vectors = block_length / dim;
    if ((uint64_t)first_chunk_id + vectors > (uint64_t)UINT32_MAX + 1u) {
        return SEARCH_ERR_ARGUMENT;
    }
    if (max_hits > SEARCH_MAX_HITS) {
        max_hits = SEARCH_MAX_HITS;
    }
    for (i = 0u; i < vectors; i++) {
        keep_hit(hits, max_hits, hit_count, first_chunk_id + (uint32_t)i,
                 vector_q16(dot_i8(query, block + i * dim, dim)));
    }
    return SEARCH_OK;
}

void search_index_keep_hit(search_hit_t *hits, size_t capacity, size_t *count, uint32_t chunk_id,
                           int32_t score_q16) {
    keep_hit(hits, capacity, count, chunk_id, score_q16);
}

/* A chunk's fused score, kept as the exact fraction numerator / denominator. */
typedef struct {
    uint32_t chunk_id;
    uint32_t numerator;
    uint32_t denominator;
} fused_t;

static void fuse_rank(fused_t *fused, size_t *count, uint32_t chunk_id, size_t rank) {
    uint32_t denominator = SEARCH_RRF_K + (uint32_t)rank;
    size_t i;

    for (i = 0u; i < *count; i++) {
        if (fused[i].chunk_id == chunk_id) {
            /* a/b + 1/d = (a*d + b) / (b*d) */
            fused[i].numerator = fused[i].numerator * denominator + fused[i].denominator;
            fused[i].denominator *= denominator;
            return;
        }
    }
    fused[*count].chunk_id = chunk_id;
    fused[*count].numerator = 1u;
    fused[*count].denominator = denominator;
    (*count)++;
}

/* Whether a ranks before b: the larger fused score, then the lower chunk id. */
static int fused_before(const fused_t *a, const fused_t *b) {
    uint64_t left = (uint64_t)a->numerator * b->denominator;
    uint64_t right = (uint64_t)b->numerator * a->denominator;

    return left != right ? left > right : a->chunk_id < b->chunk_id;
}

void search_index_fuse(const search_hit_t *keyword, size_t keyword_count, const search_hit_t *nearest,
                       size_t nearest_count, search_hit_t *hits, size_t max_hits, size_t *hit_count) {
    fused_t fused[2u * SEARCH_MAX_HITS];
    size_t fused_count = 0u;
    size_t i;
    size_t j;

    if (keyword_count > SEARCH_MAX_HITS) {
        keyword_count = SEARCH_MAX_HITS;
    }
    if (nearest_count > SEARCH_MAX_HITS) {
        nearest_count = SEARCH_MAX_HITS;
    }
    if (max_hits > SEARCH_MAX_HITS) {
        max_hits = SEARCH_MAX_HITS;
    }
    for (i = 0u; i < keyword_count; i++) {
        fuse_rank(fused, &fused_count, keyword[i].chunk_id, i + 1u);
    }
    for (i = 0u; i < nearest_count; i++) {
        fuse_rank(fused, &fused_count, nearest[i].chunk_id, i + 1u);
    }

    /* Insertion sort: there are at most 2 x SEARCH_MAX_HITS. */
    for (i = 1u; i < fused_count; i++) {
        fused_t moving = fused[i];

        for (j = i; j > 0u && fused_before(&moving, &fused[j - 1u]); j--) {
            fused[j] = fused[j - 1u];
        }
        fused[j] = moving;
    }
    for (i = 0u; i < fused_count && i < max_hits; i++) {
        /* x 65536 rounded half up; at most 2/61 x 65536, so it fits. */
        uint64_t numerator = (uint64_t)fused[i].numerator * 2u * 65536u + fused[i].denominator;

        hits[i].chunk_id = fused[i].chunk_id;
        hits[i].score_q16 = (int32_t)(numerator / (2u * (uint64_t)fused[i].denominator));
    }
    *hit_count = i;
}

search_status_t search_index_query_hybrid(search_index_t *index, const char *text,
                                          const int8_t *query_vector, size_t dim,
                                          search_hit_t *hits, size_t max_hits, size_t *hit_count) {
    search_hit_t keyword[SEARCH_MAX_HITS];
    search_hit_t nearest[SEARCH_MAX_HITS];
    size_t keyword_count;
    size_t nearest_count;
    search_status_t status;

    if (hit_count != NULL) {
        *hit_count = 0u;
    }
    if (index == NULL || text == NULL || hits == NULL || hit_count == NULL || index->file == NULL) {
        return SEARCH_ERR_ARGUMENT;
    }
    if (index->dim == 0u) {
        return search_index_query(index, text, hits, max_hits, hit_count);
    }
    if (query_vector == NULL) {
        return SEARCH_ERR_ARGUMENT;
    }
    if (dim != index->dim) {
        return SEARCH_ERR_DIM;
    }
    if (max_hits > SEARCH_MAX_HITS) {
        max_hits = SEARCH_MAX_HITS;
    }

    status = search_index_query(index, text, keyword, SEARCH_MAX_HITS, &keyword_count);
    if (status == SEARCH_OK) {
        status = search_index_query_vectors(index, query_vector, dim, nearest, SEARCH_MAX_HITS, &nearest_count);
    }
    if (status != SEARCH_OK) {
        return status;
    }
    search_index_fuse(keyword, keyword_count, nearest, nearest_count, hits, max_hits, hit_count);
    return SEARCH_OK;
}

/* ---- text -------------------------------------------------------------- */

/* Where a chunk's text lies in the text section. */
static search_status_t chunk_text(search_index_t *index, uint32_t chunk_id, uint32_t *offset, uint32_t *length) {
    const uint8_t *entry;
    search_status_t status = doc_entry(index, chunk_id, &entry);
    uint32_t text_len = index->sections[SEARCH_SECTION_TEXT].len;

    if (status != SEARCH_OK) {
        return status;
    }
    *offset = get_u32(entry);
    *length = get_u32(entry + 4);
    /* The text and its terminating NUL must lie inside the section. */
    if (*offset > text_len || (uint64_t)*length + 1u > (uint64_t)text_len - *offset) {
        return SEARCH_ERR_FORMAT;
    }
    return SEARCH_OK;
}

/* The position of the first '\n' in the length bytes at offset, or length. */
static search_status_t find_newline(search_index_t *index, uint32_t offset, uint32_t length, uint32_t *position) {
    uint32_t done = 0u;

    while (done < length) {
        size_t count = length - done;
        const uint8_t *newline;
        search_status_t status;

        if (count > sizeof(index->io)) {
            count = sizeof(index->io);
        }
        status = section_read(index, SEARCH_SECTION_TEXT, (uint64_t)offset + done, index->io, count);
        if (status != SEARCH_OK) {
            return status;
        }
        newline = memchr(index->io, '\n', count);
        if (newline != NULL) {
            *position = done + (uint32_t)(newline - index->io);
            return SEARCH_OK;
        }
        done += (uint32_t)count;
    }
    *position = length;
    return SEARCH_OK;
}

/* The longest prefix of the first length bytes of text that does not end
 * inside a UTF-8 sequence. */
static size_t utf8_prefix(const char *text, size_t length) {
    size_t start = length;
    unsigned char lead;
    size_t need;

    while (start > 0u && length - start < 3u && ((unsigned char)text[start - 1u] & 0xC0u) == 0x80u) {
        start--;
    }
    if (start == 0u) {
        return length;
    }
    lead = (unsigned char)text[start - 1u];
    if (lead < 0xC0u) {
        return length;
    }
    need = lead >= 0xF0u ? 4u : (lead >= 0xE0u ? 3u : 2u);
    return length - (start - 1u) < need ? start - 1u : length;
}

static search_status_t copy_text(search_index_t *index, uint32_t offset, uint32_t length, char *buf,
                                 size_t capacity) {
    size_t count = length;
    search_status_t status;

    buf[0] = '\0';
    if (count > capacity - 1u) {
        count = capacity - 1u;
    }
    status = section_read(index, SEARCH_SECTION_TEXT, offset, buf, count);
    if (status != SEARCH_OK) {
        buf[0] = '\0';
        return status;
    }
    if (count < length) {
        count = utf8_prefix(buf, count);
        buf[count] = '\0';
        return SEARCH_ERR_TOO_LARGE;
    }
    buf[count] = '\0';
    return SEARCH_OK;
}

static search_status_t read_part(search_index_t *index, uint32_t chunk_id, int body, char *buf, size_t capacity) {
    uint32_t offset;
    uint32_t length;
    uint32_t newline;
    search_status_t status;

    if (index == NULL || buf == NULL || capacity == 0u || index->file == NULL) {
        return SEARCH_ERR_ARGUMENT;
    }
    buf[0] = '\0';
    if (chunk_id >= index->chunk_count) {
        return SEARCH_ERR_ARGUMENT;
    }
    status = chunk_text(index, chunk_id, &offset, &length);
    if (status == SEARCH_OK) {
        status = find_newline(index, offset, length, &newline);
    }
    if (status != SEARCH_OK) {
        return status;
    }
    if (!body) {
        return copy_text(index, offset, newline, buf, capacity);
    }
    if (newline == length) {
        return SEARCH_OK; /* a title and no body */
    }
    return copy_text(index, offset + newline + 1u, length - newline - 1u, buf, capacity);
}

search_status_t search_index_read_title(search_index_t *index, uint32_t chunk_id, char *buf, size_t capacity) {
    return read_part(index, chunk_id, 0, buf, capacity);
}

search_status_t search_index_read_body(search_index_t *index, uint32_t chunk_id, char *buf, size_t capacity) {
    return read_part(index, chunk_id, 1, buf, capacity);
}
