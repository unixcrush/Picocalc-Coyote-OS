#ifndef COYOTE_AI_EMBED_CLIENT_H
#define COYOTE_AI_EMBED_CLIENT_H

#include <stddef.h>
#include <stdint.h>

/*
 * The request and response halves of asking an OpenAI-compatible embedding
 * endpoint ({embed_url}/v1/embeddings) for a query vector, as
 * docs/design/on-device-search.md pins them.  No transport, no scheduling and
 * no UI: the caller sends the body this builds and hands the response body
 * back to embed_client_parse.  Nothing allocates.
 */

typedef enum {
    EMBED_OK = 0,
    EMBED_ERR_ARGUMENT,
    EMBED_ERR_DIM,
    EMBED_ERR_HTTP,
    EMBED_ERR_PARSE,
} embed_status_t;

/* Text limit; longer input is EMBED_ERR_ARGUMENT. */
#define EMBED_TEXT_MAX 512

/* The longest number embed_client_parse accepts, in characters. */
#define EMBED_NUMBER_MAX 63u

/*
 * Builds the request body {"input":["text"],"model":"model"}, NUL-terminated.
 * Returns the length written, or 0 when an argument is NULL, text or model is
 * empty, text is longer than EMBED_TEXT_MAX bytes or the body does not fit.
 */
size_t embed_client_body(char *body, size_t capacity, const char *model, const char *text);

/*
 * Parses {"data":[{"embedding":[...]}]} into dim int8 values, L2-normalized and
 * quantized exactly as the builder does: the squares summed in order, each
 * value divided by the norm, multiplied by 127, rounded half to even and
 * clamped to [-127, 127].  Returns EMBED_ERR_DIM when the vector length
 * disagrees.  Other members, in any order, are skipped.  EMBED_ERR_PARSE for
 * text that is not such a response: no "data" array, no "embedding" array,
 * more than one embedding, a value that is not a JSON number or overflows, a
 * vector that cannot be normalized, or bytes after the closing brace.
 * EMBED_ERR_ARGUMENT for a NULL pointer or a dim of 0.  out is written only on
 * EMBED_OK.
 */
embed_status_t embed_client_parse(const char *json, size_t length, int8_t *out, size_t dim);

#endif
