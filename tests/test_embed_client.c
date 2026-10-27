#include <stdint.h>
#include <string.h>

#include "ai/embed_client.h"
#include "test_util.h"

static embed_status_t parse(const char *json, int8_t *out, size_t dim) {
    return embed_client_parse(json, strlen(json), out, dim);
}

static void check_body(void) {
    char body[1200];
    char small[24];
    char text[EMBED_TEXT_MAX + 2];
    size_t length;

    length = embed_client_body(body, sizeof(body), "nomic-embed-text", "wifi password");
    CHECK_STR_EQ(body, "{\"input\":[\"wifi password\"],\"model\":\"nomic-embed-text\"}");
    CHECK(length == strlen(body));

    /* Quotes, backslashes and control characters are escaped; UTF-8 passes. */
    length = embed_client_body(body, sizeof(body), "m", "say \"hi\"\\\n\t\x01 Z\xc3\xbcrich");
    CHECK_STR_EQ(body, "{\"input\":[\"say \\\"hi\\\"\\\\\\n\\t\\u0001 Z\xc3\xbcrich\"],\"model\":\"m\"}");
    CHECK(length == strlen(body));

    /* EMBED_TEXT_MAX bytes are taken, one more is not. */
    memset(text, 'a', sizeof(text));
    text[EMBED_TEXT_MAX] = '\0';
    CHECK(embed_client_body(body, sizeof(body), "m", text) ==
          EMBED_TEXT_MAX + strlen("{\"input\":[\"\"],\"model\":\"m\"}"));
    text[EMBED_TEXT_MAX] = 'a';
    text[EMBED_TEXT_MAX + 1] = '\0';
    CHECK(embed_client_body(body, sizeof(body), "m", text) == 0u);
    CHECK_STR_EQ(body, "");

    /* A body that does not fit, with room for all but the NUL. */
    CHECK(embed_client_body(small, sizeof(small), "nomic-embed-text", "wifi") == 0u);
    CHECK_STR_EQ(small, "");
    length = embed_client_body(body, sizeof(body), "m", "abc");
    CHECK(embed_client_body(small, length, "m", "abc") == 0u);
    CHECK(embed_client_body(small, length + 1u, "m", "abc") == length);

    CHECK(embed_client_body(NULL, sizeof(body), "m", "abc") == 0u);
    CHECK(embed_client_body(body, 0u, "m", "abc") == 0u);
    CHECK(embed_client_body(body, sizeof(body), NULL, "abc") == 0u);
    CHECK(embed_client_body(body, sizeof(body), "m", NULL) == 0u);
    CHECK(embed_client_body(body, sizeof(body), "", "abc") == 0u);
    CHECK(embed_client_body(body, sizeof(body), "m", "") == 0u);
}

/*
 * Canned responses.  The expected bytes are what build_index.quantize gives
 * for the same values; tools/search_index/test_build_index.py checks the
 * Python side against the same numbers, so the two quantizers are compared.
 */
static void check_parse(void) {
    static const int8_t real_expected[8] = {2, -64, 0, -3, 47, 0, 98, -17};
    /* Norm 254: these land exactly on -0.5, 2.5, -29.5 and 123.5, which round
     * half to even, as Python's round() does. */
    static const int8_t ties_expected[4] = {0, 2, -30, 124};
    int8_t out[8];
    int8_t untouched[8];

    memset(out, 0x55, sizeof(out));
    CHECK(parse("{\"object\":\"list\",\"data\":[{\"object\":\"embedding\",\"index\":0,\"embedding\":"
                "[0.0123,-0.4567,1.5e-3,-2.25E-2,0.3333333333333333,-0.0,7.0e-1,-0.125]}],"
                "\"model\":\"nomic-embed-text\",\"usage\":{\"prompt_tokens\":2,\"total_tokens\":2}}",
                out, 8u) == EMBED_OK);
    CHECK_BYTES_EQ(out, real_expected, sizeof(real_expected));

    /* Whitespace anywhere JSON allows it, members in another order. */
    memset(out, 0x55, sizeof(out));
    CHECK(parse(" \r\n{ \"model\" : \"x\" ,\n\t\"data\" : [ {\n \"embedding\" : [ -1e0 ,0.5E1,\n-5.9e+1 , 247.0 ] ,"
                " \"index\" : 0 } ] }\n",
                out, 4u) == EMBED_OK);
    CHECK_BYTES_EQ(out, ties_expected, sizeof(ties_expected));

    /* Members skipped on the way: nested values, strings holding brackets,
     * quotes and escapes, literals. */
    CHECK(parse("{\"meta\":{\"a\":[1,{\"b\":\"]}\\\"\"}],\"c\":null},\"ok\":true,\"no\":false,"
                "\"data\":[{\"extra\":[[],{}],\"embedding\":[3,4]}]}",
                out, 2u) == EMBED_OK);
    CHECK(out[0] == 76 && out[1] == 102);

    /* The builder's stored vectors parse back to themselves. */
    CHECK(parse("{\"data\":[{\"embedding\":[110,63,0,0]}]}", out, 4u) == EMBED_OK);
    CHECK(out[0] == 110 && out[1] == 63 && out[2] == 0 && out[3] == 0);
    CHECK(parse("{\"data\":[{\"embedding\":[2,-64,0,-3,47,0,98,-17]}]}", out, 8u) == EMBED_OK);
    CHECK_BYTES_EQ(out, real_expected, sizeof(real_expected));

    /* Lengths that disagree with dim, and out left alone. */
    memset(untouched, 0x55, sizeof(untouched));
    memset(out, 0x55, sizeof(out));
    CHECK(parse("{\"data\":[{\"embedding\":[0.1,0.2,0.3]}]}", out, 4u) == EMBED_ERR_DIM);
    CHECK(parse("{\"data\":[{\"embedding\":[0.1,0.2,0.3,0.4,0.5]}]}", out, 4u) == EMBED_ERR_DIM);
    CHECK(parse("{\"data\":[{\"embedding\":[]}]}", out, 4u) == EMBED_ERR_DIM);
    CHECK_BYTES_EQ(out, untouched, sizeof(out));

    /* Not an embedding response. */
    CHECK(parse("", out, 4u) == EMBED_ERR_PARSE);
    CHECK(embed_client_parse("{}", 0u, out, 4u) == EMBED_ERR_PARSE);
    CHECK(parse("   ", out, 4u) == EMBED_ERR_PARSE);
    CHECK(parse("{}", out, 4u) == EMBED_ERR_PARSE);
    CHECK(parse("{\"error\":{\"message\":\"model not found\"}}", out, 4u) == EMBED_ERR_PARSE);
    CHECK(parse("{\"data\":[]}", out, 4u) == EMBED_ERR_PARSE);
    CHECK(parse("{\"data\":{}}", out, 4u) == EMBED_ERR_PARSE);
    CHECK(parse("{\"data\":[{\"index\":0}]}", out, 4u) == EMBED_ERR_PARSE);
    CHECK(parse("{\"data\":[{\"embedding\":\"0.1,0.2\"}]}", out, 2u) == EMBED_ERR_PARSE);
    CHECK(parse("{\"data\":[{\"embedding\":[1,2]},{\"embedding\":[1,2]}]}", out, 2u) == EMBED_ERR_PARSE);
    CHECK(parse("{\"embedding\":[1,2]}", out, 2u) == EMBED_ERR_PARSE);
    CHECK(parse("[{\"embedding\":[1,2]}]", out, 2u) == EMBED_ERR_PARSE);
    CHECK(parse("{\"data\":[{\"embedding\":[1,2]}]", out, 2u) == EMBED_ERR_PARSE);
    CHECK(parse("{\"data\":[{\"embedding\":[1,2]}]} x", out, 2u) == EMBED_ERR_PARSE);
    CHECK(parse("{\"data\":[{\"embedding\":[1,2]}],}", out, 2u) == EMBED_ERR_PARSE);
    CHECK(parse("{\"data\":[{\"embedding\":[1,2,]}]}", out, 2u) == EMBED_ERR_PARSE);
    CHECK(parse("{\"x\":[1}],\"data\":[{\"embedding\":[1,2]}]}", out, 2u) == EMBED_ERR_PARSE);
    /* Numbers JSON does not allow, or a double cannot hold. */
    CHECK(parse("{\"data\":[{\"embedding\":[1,NaN]}]}", out, 2u) == EMBED_ERR_PARSE);
    CHECK(parse("{\"data\":[{\"embedding\":[1,+2]}]}", out, 2u) == EMBED_ERR_PARSE);
    CHECK(parse("{\"data\":[{\"embedding\":[1,.5]}]}", out, 2u) == EMBED_ERR_PARSE);
    CHECK(parse("{\"data\":[{\"embedding\":[1,01]}]}", out, 2u) == EMBED_ERR_PARSE);
    CHECK(parse("{\"data\":[{\"embedding\":[1,1e]}]}", out, 2u) == EMBED_ERR_PARSE);
    CHECK(parse("{\"data\":[{\"embedding\":[1,1e999]}]}", out, 2u) == EMBED_ERR_PARSE);
    CHECK(parse("{\"data\":[{\"embedding\":[1,\"2\"]}]}", out, 2u) == EMBED_ERR_PARSE);
    /* A vector of zeros has no direction. */
    CHECK(parse("{\"data\":[{\"embedding\":[0,-0.0]}]}", out, 2u) == EMBED_ERR_PARSE);
    /* The length given is honoured: the closing braces lie past it. */
    CHECK(embed_client_parse("{\"data\":[{\"embedding\":[3,4]}]}", 27u, out, 2u) == EMBED_ERR_PARSE);
    CHECK_BYTES_EQ(out, untouched, sizeof(out));

    CHECK(embed_client_parse(NULL, 2u, out, 2u) == EMBED_ERR_ARGUMENT);
    CHECK(parse("{\"data\":[{\"embedding\":[3,4]}]}", NULL, 2u) == EMBED_ERR_ARGUMENT);
    CHECK(parse("{\"data\":[{\"embedding\":[3,4]}]}", out, 0u) == EMBED_ERR_ARGUMENT);
}

void test_embed_client(void) {
    check_body();
    check_parse();
}
