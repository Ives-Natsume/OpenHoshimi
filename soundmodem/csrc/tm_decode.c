/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * ASRTU-1 telemetry decoder: KISS de-framing, field extraction and rendering.
 *
 * The framing follows BG4QBF's Go reference: a 223-byte frame begins with a
 * four-byte header, a continuation-length byte (frame[4]) and that many bytes
 * of the packet body carried over from the previous frame, after which the
 * body is a 0xC0-delimited packet stream. Within a packet the seven-byte
 * header places the main type at offset 1 and the sub type at offset 6; the
 * body (which the format tables describe) begins at offset 7.
 */

#include "tm.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define KISS_FEND	0xC0u
#define FRAME_LEN	223u
#define PKT_MAX		512u

/* ANSI colours, emitted only when the caller asks for colour. */
#define A_RST	"\033[0m"
#define A_HDR	"\033[1;97m"	/* bold white: packet header		*/
#define A_DIM	"\033[2m"	/* dim: names, units, gaps		*/
#define A_F0	"\033[36m"	/* cyan: even-indexed field span		*/
#define A_F1	"\033[33m"	/* yellow: odd-indexed field span	*/
#define A_NUM	"\033[32m"	/* green: numeric value			*/
#define A_ENUM	"\033[1;36m"	/* bold cyan: enumerated value		*/
#define A_HEX	"\033[35m"	/* magenta: raw hex value		*/

static const char *col(int color, const char *code)
{
	return color ? code : "";
}

const struct tm_packet *tm_lookup(unsigned char main, unsigned char sub)
{
	unsigned int i;

	for (i = 0; i < tm_npackets; i++) {
		if (tm_packets[i].main == main && tm_packets[i].sub == sub)
			return &tm_packets[i];
	}
	return NULL;
}

/* Extract up to 64 bits MSB-first; bit 7 is the MSB of a byte. */
static uint64_t extract_bits(const unsigned char *body, size_t len,
			     const struct tm_field *f)
{
	size_t start = (size_t)f->byte * 8u + (7u - f->bit);
	uint64_t v = 0;
	unsigned int i;

	for (i = 0; i < f->width; i++) {
		size_t idx = start + i;
		size_t byte = idx / 8u;
		unsigned int off = 7u - (unsigned int)(idx % 8u);

		if (byte >= len)
			break;
		v = (v << 1) | ((uint64_t)(body[byte] >> off) & 1u);
	}
	return v;
}

/* Sign-extend a `width`-bit two's-complement value held in `raw`. */
static int64_t sign_extend(uint64_t raw, unsigned int width)
{
	if (width == 0 || width >= 64)
		return (int64_t)raw;
	if (raw & ((uint64_t)1 << (width - 1)))
		raw |= ~(((uint64_t)1 << width) - 1);
	return (int64_t)raw;
}

/* Reinterpret raw bits as an IEEE-754 float of `width` (32 or 64) bits. */
static double raw_to_float(uint64_t raw, unsigned int width)
{
	if (width == 64) {
		double d;

		memcpy(&d, &raw, sizeof(d));
		return d;
	} else {
		uint32_t u = (uint32_t)raw;
		float fv;

		memcpy(&fv, &u, sizeof(fv));
		return (double)fv;
	}
}

static const char *enum_desc(const struct tm_field *f, uint64_t raw)
{
	unsigned int i;

	for (i = 0; i < f->nmap; i++) {
		if (f->map[i].key == raw)
			return f->map[i].desc;
	}
	return NULL;
}

/* Render one field's value (without name/unit) into buf. */
static void format_value(const struct tm_field *f, const unsigned char *body,
			 size_t len, char *buf, size_t cap)
{
	uint64_t raw;

	/* Wide blobs (Beidou dumps, debug buffers) exceed 64 bits; render the
	 * byte-aligned span as raw hex rather than a single scalar. */
	if (f->width > 64) {
		unsigned int nb = (unsigned int)f->width / 8u;
		unsigned int j;
		size_t pos = 0;

		for (j = 0; j < nb && f->byte + j < len && pos + 3 < cap; j++) {
			int w = snprintf(buf + pos, cap - pos, "%02X ",
					 body[f->byte + j]);
			if (w > 0)
				pos += (size_t)w;
		}
		if (pos > 0)
			buf[pos - 1] = '\0';
		else if (cap > 0)
			buf[0] = '\0';
		return;
	}

	raw = extract_bits(body, len, f);

	if (f->map != NULL) {
		const char *d = enum_desc(f, raw);

		if (d != NULL)
			snprintf(buf, cap, "%s", d);
		else
			snprintf(buf, cap, "0x%" PRIX64 "(未知)", raw);
		return;
	}

	/* Switch on the raw base (not the enum type) so a defensive default is
	 * allowed without tripping -Wcovered-switch-default. */
	switch (f->base) {
	case TM_UX: {
		double v = (double)raw * f->mul / f->div + f->off;

		if (f->dec >= 0)
			snprintf(buf, cap, "%.*f", (int)f->dec, v);
		else
			snprintf(buf, cap, "%" PRIu64, raw);
		break;
	}
	case TM_SX: {
		int64_t s = sign_extend(raw, f->width);
		double v = (double)s * f->mul / f->div + f->off;

		if (f->dec >= 0)
			snprintf(buf, cap, "%.*f", (int)f->dec, v);
		else
			snprintf(buf, cap, "%" PRId64, s);
		break;
	}
	case TM_FX: {
		double v = raw_to_float(raw, f->width) * f->mul / f->div + f->off;
		int d = f->dec >= 0 ? (int)f->dec : 6;

		snprintf(buf, cap, "%.*f", d, v);
		break;
	}
	case TM_HX:
	case TM_RSVD:
	default:
		snprintf(buf, cap, "0x%0*" PRIX64, (f->width + 3) / 4, raw);
		break;
	}
}

/* Bytes a field spans in the body, accounting for its start bit. */
static unsigned int field_nbyte(const struct tm_field *f)
{
	return ((7u - f->bit) + f->width - 1u) / 8u + 1u;
}

/* Colour code for a decoded value, by field class. */
static const char *value_color(const struct tm_field *f, int color)
{
	if (!color)
		return "";
	if (f->map != NULL)
		return A_ENUM;
	if (f->base == TM_HX || f->base == TM_RSVD)
		return A_HEX;
	return A_NUM;
}

/* Append `s` at offset n within out[cap]; returns the new offset. */
#define PUT(...)							\
	do {								\
		int w_ = snprintf(out + n, cap - n, __VA_ARGS__);	\
		if (w_ > 0)						\
			n += (size_t)w_ < cap - n ?			\
				(size_t)w_ : cap - n;			\
	} while (0)

/* Field colour for a body byte, from its owning-field parity map. */
static const char *byte_color(unsigned char pm)
{
	return pm == 0 ? A_F0 : pm == 1 ? A_F1 : A_DIM;
}

/*
 * Field-coloured hex dump of the packet body, hexdump -C style: offset, two
 * eight-byte hex groups and a `|...|` ASCII sidebar, each byte and its ASCII
 * glyph tinted by the field it belongs to.
 */
static size_t render_body(const unsigned char *body, size_t len,
			  unsigned char *pmap, char *out, size_t cap,
			  size_t n, int color)
{
	size_t row;

	for (row = 0; row < len; row += 16) {
		size_t j;

		PUT("  %s%04zu%s  ", col(color, A_DIM), row, col(color, A_RST));
		for (j = 0; j < 16; j++) {
			if (row + j < len)
				PUT("%s%02X%s ",
				    col(color, byte_color(pmap[row + j])),
				    body[row + j], col(color, A_RST));
			else
				PUT("   ");
			if (j == 7)
				PUT(" ");
		}
		PUT("%s|%s", col(color, A_DIM), col(color, A_RST));
		for (j = 0; j < 16 && row + j < len; j++) {
			unsigned char b = body[row + j];
			char ch = (b >= 0x20 && b < 0x7F) ? (char)b : '.';

			PUT("%s%c%s", col(color, byte_color(pmap[row + j])), ch,
			    col(color, A_RST));
		}
		PUT("%s|%s\n", col(color, A_DIM), col(color, A_RST));
	}
	return n;
}

/* Append a rendered packet to the report; returns bytes written to out. */
static size_t render_packet(const struct tm_packet *p,
			    const unsigned char *body, size_t len,
			    char *out, size_t cap, int color, int full)
{
	unsigned char pmap[PKT_MAX];
	size_t n = 0;
	unsigned int i;

	if (!full) {
		PUT("  %s%-7s%s  main=%02X sub=%02X  %zu B\n",
		    col(color, A_HDR), p->name, col(color, A_RST),
		    p->main, p->sub, len);
		return n;
	}

	memset(pmap, 0xFF, sizeof(pmap));
	for (i = 0; i < p->nfield; i++) {
		const struct tm_field *f = &p->fields[i];
		unsigned int nb = field_nbyte(f);
		unsigned int b;

		for (b = 0; b < nb && f->byte + b < len; b++) {
			if (pmap[f->byte + b] == 0xFF)
				pmap[f->byte + b] = (unsigned char)(i & 1u);
		}
	}

	PUT("%s== %s  main=%02X sub=%02X  %zu bytes ==%s\n",
	    col(color, A_HDR), p->name, p->main, p->sub, len, col(color, A_RST));
	n = render_body(body, len, pmap, out, cap, n, color);

	for (i = 0; i < p->nfield && n < cap; i++) {
		const struct tm_field *f = &p->fields[i];
		char val[128];
		const char *vc = value_color(f, color);

		format_value(f, body, len, val, sizeof(val));
		PUT("  %s%3u%s %s = %s%s%s%s%s  %s%s%s\n",
		    col(color, A_DIM), f->byte, col(color, A_RST), f->cn,
		    vc, val, col(color, A_RST),
		    col(color, A_DIM), f->unit,
		    col(color, A_DIM), f->en, col(color, A_RST));
	}
	return n;
}

/*
 * Stateful KISS de-packetiser. The header state machine is reset per frame;
 * the packet buffer and the last-seen type carry over so a body split across
 * frames is stitched back together.
 */
struct depacket {
	unsigned char	buf[PKT_MAX];
	size_t		idx;
	unsigned char	main;
	unsigned char	sub;
};

struct tm_decoder {
	struct depacket dp;
};

/* Process one frame's KISS stream, rendering completed packets at out+n. */
static size_t process_frame(struct depacket *dp, const unsigned char *frame,
			    char *out, size_t cap, size_t n, int color,
			    int full, int *count)
{
	unsigned int contlen = frame[4];
	unsigned int bi;
	int stat = 0;
	int head = 0;

	/*
	 * A telemetry frame's continuation length cannot exceed the body space
	 * after the five-byte prefix; a larger value means this is not a KISS
	 * telemetry frame (e.g. a data-channel frame, where frame[4] is
	 * unrelated), so skip it rather than mis-parse.
	 */
	if (contlen > FRAME_LEN - 5u)
		return n;

	/* Body tail carried over from the previous frame. */
	for (bi = 5; bi < 5u + contlen && bi < FRAME_LEN; bi++) {
		if (dp->idx < PKT_MAX)
			dp->buf[dp->idx++] = frame[bi];
	}

	for (bi = 5u + contlen; bi < FRAME_LEN; bi++) {
		unsigned char b = frame[bi];

		if (b == KISS_FEND) {
			const struct tm_packet *p = tm_lookup(dp->main, dp->sub);

			if (p != NULL && dp->idx > 0) {
				n += render_packet(p, dp->buf, dp->idx,
						   out + n, cap - n, color, full);
				(*count)++;
			}
			dp->idx = 0;
			stat = 0;
			head = 0;
			continue;
		}

		switch (stat) {
		case 0:
			stat = 1;
			head = 1;
			break;
		case 1:
			if (head == 1)
				dp->main = b;
			else if (head == 6) {
				dp->sub = b;
				stat = 2;
			}
			head++;
			break;
		default:
			if (dp->idx < PKT_MAX)
				dp->buf[dp->idx++] = b;
			break;
		}
	}
	return n;
}

int tm_decode_frames(const unsigned char *frames, size_t nframes,
		     char *out, size_t cap, int color, int full)
{
	struct depacket dp;
	size_t n = 0;
	size_t fi;
	int count = 0;

	memset(&dp, 0, sizeof(dp));
	if (cap > 0)
		out[0] = '\0';
	for (fi = 0; fi < nframes; fi++)
		n = process_frame(&dp, frames + fi * FRAME_LEN, out, cap, n,
				  color, full, &count);
	return count;
}

struct tm_decoder *tm_decoder_open(void)
{
	struct tm_decoder *dec = (struct tm_decoder *)calloc(1, sizeof(*dec));

	return dec;
}

void tm_decoder_close(struct tm_decoder *dec)
{
	free(dec);
}

int tm_decoder_feed(struct tm_decoder *dec, const unsigned char *frame,
		    char *out, size_t cap, int color, int full)
{
	int count = 0;

	if (cap > 0)
		out[0] = '\0';
	process_frame(&dec->dp, frame, out, cap, 0, color, full, &count);
	return count;
}
