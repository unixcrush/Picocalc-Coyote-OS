#include "ai/embed_client.h"

#include "ai/json_stream.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ---- request ----------------------------------------------------------- */

/* Append text, raw or as the inside of a JSON string, at *at; 0 if it does
 * not fit with the NUL after it. */
static int append(char *body, size_t capacity, size_t *at, const char *text, int escape) {
    size_t length = strlen(text);

    if (escape) {
        if (json_escape_string(text, length, body + *at, capacity - *at, &length) != 0) {
            return 0;
        }
    } else {
        if (length >= capacity - *at) {
            return 0;
        }
        memcpy(body + *at, text, length + 1u);
    }
    *at += length;
    return 1;
}

size_t embed_client_body(char *body, size_t capacity, const char *model, const char *text) {
    size_t at = 0u;
    size_t text_length;

    if (body == NULL || capacity == 0u) {
        return 0u;
    }
    body[0] = '\0';
    if (model == NULL || text == NULL || model[0] == '\0') {
        return 0u;
    }
    text_length = strlen(text);
    if (text_length == 0u || text_length > EMBED_TEXT_MAX) {
        return 0u;
    }
    if (!append(body, capacity, &at, "{\"input\":[\"", 0) || !append(body, capacity, &at, text, 1) ||
        !append(body, capacity, &at, "\"],\"model\":\"", 0) || !append(body, capacity, &at, model, 1) ||
        !append(body, capacity, &at, "\"}", 0)) {
        body[0] = '\0';
        return 0u;
    }
    return at;
}

/* ---- response ---------------------------------------------------------- */

/* Nesting deeper than this, in a member being skipped, is refused. */
#define SKIP_DEPTH_MAX 64u

typedef struct {
    const char *p;
    const char *end;
} cursor_t;

static void skip_space(cursor_t *c) {
    while (c->p < c->end && (*c->p == ' ' || *c->p == '\t' || *c->p == '\n' || *c->p == '\r')) {
        c->p++;
    }
}

static int peek(const cursor_t *c, char expected) {
    return c->p < c->end && *c->p == expected;
}

static int take(cursor_t *c, char expected) {
    if (!peek(c, expected)) {
        return 0;
    }
    c->p++;
    return 1;
}

/* A string, left escaped: *text and *length span what lies between the
 * quotes. */
static int read_string(cursor_t *c, const char **text, size_t *length) {
    const char *start;

    if (!take(c, '"')) {
        return 0;
    }
    start = c->p;
    while (c->p < c->end && *c->p != '"') {
        if ((unsigned char)*c->p < 0x20u) {
            return 0;
        }
        if (*c->p == '\\') {
            c->p++;
            if (c->p == c->end) {
                return 0;
            }
        }
        c->p++;
    }
    if (c->p == c->end) {
        return 0;
    }
    *text = start;
    *length = (size_t)(c->p - start);
    c->p++;
    return 1;
}

static int is_digit(const cursor_t *c) {
    return c->p < c->end && *c->p >= '0' && *c->p <= '9';
}

/* A JSON number, as strictly as the grammar has it, into a finite double. */
static int read_number(cursor_t *c, double *value) {
    const char *start = c->p;
    char digits[EMBED_NUMBER_MAX + 1u];
    char *stop;
    size_t length;

    take(c, '-');
    if (take(c, '0')) {
        /* no leading zeros */
    } else if (is_digit(c)) {
        while (is_digit(c)) {
            c->p++;
        }
    } else {
        return 0;
    }
    if (take(c, '.')) {
        if (!is_digit(c)) {
            return 0;
        }
        while (is_digit(c)) {
            c->p++;
        }
    }
    if (take(c, 'e') || take(c, 'E')) {
        if (!take(c, '+')) {
            take(c, '-');
        }
        if (!is_digit(c)) {
            return 0;
        }
        while (is_digit(c)) {
            c->p++;
        }
    }
    length = (size_t)(c->p - start);
    if (length > EMBED_NUMBER_MAX) {
        return 0;
    }
    memcpy(digits, start, length);
    digits[length] = '\0';
    *value = strtod(digits, &stop);
    return stop == digits + length && isfinite(*value);
}

static int read_literal(cursor_t *c, const char *word) {
    size_t length = strlen(word);

    if ((size_t)(c->end - c->p) < length || memcmp(c->p, word, length) != 0) {
        return 0;
    }
    c->p += length;
    return 1;
}

/* Step over one value of any kind.  Containers are matched bracket for
 * bracket but what they hold is not otherwise checked. */
static int skip_value(cursor_t *c) {
    uint64_t arrays = 0u; /* bit n: level n is an array */
    unsigned depth = 0u;
    const char *text;
    size_t length;
    double number;

    do {
        skip_space(c);
        if (c->p == c->end) {
            return 0;
        }
        switch (*c->p) {
        case '"':
            if (!read_string(c, &text, &length)) {
                return 0;
            }
            break;
        case '{':
        case '[':
            if (depth == SKIP_DEPTH_MAX) {
                return 0;
            }
            arrays = (arrays & ~((uint64_t)1u << depth)) | ((uint64_t)(*c->p == '[') << depth);
            depth++;
            c->p++;
            break;
        case '}':
        case ']':
            if (depth == 0u || ((arrays >> (depth - 1u)) & 1u) != (uint64_t)(*c->p == ']')) {
                return 0;
            }
            depth--;
            c->p++;
            break;
        case ',':
        case ':':
            if (depth == 0u) {
                return 0;
            }
            c->p++;
            break;
        case 't':
            if (!read_literal(c, "true")) {
                return 0;
            }
            break;
        case 'f':
            if (!read_literal(c, "false")) {
                return 0;
            }
            break;
        case 'n':
            if (!read_literal(c, "null")) {
                return 0;
            }
            break;
        default:
            if (!read_number(c, &number)) {
                return 0;
            }
            break;
        }
    } while (depth > 0u);
    return 1;
}

/*
 * Step to the next member of an object whose '{' has been read: 1 with the
 * cursor at the member's value and *is_key telling whether its name is key,
 * 0 at the closing '}', which is consumed, and -1 for anything else.
 */
static int next_member(cursor_t *c, int *first, const char *key, int *is_key) {
    const char *name;
    size_t length;

    skip_space(c);
    if (take(c, '}')) {
        return 0;
    }
    if (!*first) {
        if (!take(c, ',')) {
            return -1;
        }
        skip_space(c);
    }
    *first = 0;
    if (!read_string(c, &name, &length)) {
        return -1;
    }
    skip_space(c);
    if (!take(c, ':')) {
        return -1;
    }
    skip_space(c);
    *is_key = length == strlen(key) && memcmp(name, key, length) == 0;
    return 1;
}

/* Where the embedding array starts, and what a first pass over it found. */
typedef struct {
    const char *start; /* its '[' */
    size_t count;
    double sum_of_squares;
} embedding_t;

/* Read an array of numbers, summing their squares in order, or with out
 * non-NULL quantizing them into out by norm. */
static int read_vector(cursor_t *c, embedding_t *embedding, int8_t *out, double norm) {
    size_t count = 0u;
    double sum = 0.0;

    if (!take(c, '[')) {
        return 0;
    }
    skip_space(c);
    if (!take(c, ']')) {
        do {
            double value;

            skip_space(c);
            if (!read_number(c, &value)) {
                return 0;
            }
            if (out != NULL) {
                double q = rint(value / norm * 127.0);

                out[count] = (int8_t)(q > 127.0 ? 127 : (q < -127.0 ? -127 : (int)q));
            } else {
                /* Held apart from the sum so no compiler fuses the two into
                 * one multiply-add: the builder rounds each separately. */
                volatile double square = value * value;

                sum += square;
            }
            count++;
            skip_space(c);
        } while (take(c, ','));
        if (!take(c, ']')) {
            return 0;
        }
    }
    embedding->count = count;
    embedding->sum_of_squares = sum;
    return 1;
}

/* {"embedding":[...], ...}: the one element data may hold. */
static int read_item(cursor_t *c, embedding_t *embedding) {
    int first = 1;
    int found = 0;
    int is_key;
    int step;

    if (!take(c, '{')) {
        return 0;
    }
    while ((step = next_member(c, &first, "embedding", &is_key)) == 1) {
        if (is_key && !found) {
            embedding->start = c->p;
            if (!read_vector(c, embedding, NULL, 0.0)) {
                return 0;
            }
            found = 1;
        } else if (!skip_value(c)) {
            return 0;
        }
    }
    return step == 0 && found;
}

/* "data":[{...}] with exactly one element. */
static int read_data(cursor_t *c, embedding_t *embedding) {
    if (!take(c, '[')) {
        return 0;
    }
    skip_space(c);
    if (!read_item(c, embedding)) {
        return 0;
    }
    skip_space(c);
    return take(c, ']');
}

embed_status_t embed_client_parse(const char *json, size_t length, int8_t *out, size_t dim) {
    cursor_t c;
    embedding_t embedding;
    int first = 1;
    int found = 0;
    int is_key;
    int step;
    double norm;

    if (json == NULL || out == NULL || dim == 0u) {
        return EMBED_ERR_ARGUMENT;
    }
    c.p = json;
    c.end = json + length;
    memset(&embedding, 0, sizeof(embedding));

    skip_space(&c);
    if (!take(&c, '{')) {
        return EMBED_ERR_PARSE;
    }
    while ((step = next_member(&c, &first, "data", &is_key)) == 1) {
        if (is_key && !found) {
            if (!read_data(&c, &embedding)) {
                return EMBED_ERR_PARSE;
            }
            found = 1;
        } else if (!skip_value(&c)) {
            return EMBED_ERR_PARSE;
        }
    }
    skip_space(&c);
    if (step != 0 || !found || c.p != c.end) {
        return EMBED_ERR_PARSE;
    }
    if (embedding.count != dim) {
        return EMBED_ERR_DIM;
    }

    norm = sqrt(embedding.sum_of_squares);
    if (!(norm > 0.0) || !isfinite(norm)) {
        return EMBED_ERR_PARSE;
    }
    c.p = embedding.start;
    return read_vector(&c, &embedding, out, norm) ? EMBED_OK : EMBED_ERR_PARSE;
}
