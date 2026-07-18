/* SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef ASRTU_TM_H
#define ASRTU_TM_H

#include <stddef.h>
#include <stdint.h>

/*
 * ASRTU-1 telemetry decoder.
 *
 * The RS/Viterbi core emits 223-byte CCSDS frames. Each frame carries a
 * KISS-framed (0xC0-delimited) stream of variable-length telemetry packets,
 * and a packet may straddle two frames. A packet is identified by a main type
 * (subsystem nID) and a sub type; tm_tables.c describes the field layout of
 * every known (main, sub) pair, transcribed from the ASRTU-1 format sheets.
 *
 * The KISS framing was reverse-engineered by BG4QBF (asrtu-sndmdm-backend, an
 * archived Go tool); it is re-implemented here in C.
 */

enum tm_base {
	TM_UX,		/* unsigned integer				*/
	TM_SX,		/* signed integer, two's complement		*/
	TM_FX,		/* IEEE-754 float (width 32 or 64)		*/
	TM_HX,		/* opaque, printed as hex			*/
	TM_RSVD,	/* reserved / unspecified			*/
};

struct tm_enum {
	unsigned long	key;
	const char	*desc;
};

/*
 * A decoded scalar is value = raw * mul / div + off, printed with `dec`
 * fractional digits (dec < 0 selects integer formatting). Enumerated fields
 * carry a `map` of `nmap` entries and ignore the scaling terms.
 */
struct tm_field {
	unsigned short		byte;	/* offset into the packet body	  */
	unsigned char		bit;	/* first bit, 7 = MSB of the byte */
	unsigned short		width;	/* field width in bits		  */
	unsigned char		base;	/* enum tm_base			  */
	signed char		dec;	/* fractional digits, -1 = integer */
	double			mul;
	double			div;
	double			off;
	const char		*unit;	/* "" when unit-less		  */
	const struct tm_enum	*map;	/* NULL unless enumerated	  */
	unsigned char		nmap;
	const char		*cn;
	const char		*en;
};

struct tm_packet {
	unsigned char		main;	/* subsystem nID		  */
	unsigned char		sub;
	const char		*name;	/* e.g. "SAT-S0"		  */
	const struct tm_field	*fields;
	unsigned short		nfield;
};

extern const struct tm_packet	tm_packets[];
extern const unsigned int	tm_npackets;

/* Look up a packet table by (main, sub); NULL when unknown. */
const struct tm_packet *tm_lookup(unsigned char main, unsigned char sub);

/*
 * Decode a stream of 223-byte frames into a human-readable report. `frames`
 * points at nframes * 223 bytes in receive order. The report is written to
 * `out` (NUL-terminated, truncated to `cap`). When `color` is non-zero the
 * report carries ANSI colour (field-coloured hex dump and value highlighting).
 * When `full` is zero only a one-line header per packet is emitted (a console
 * summary); when non-zero each packet is rendered in full (hex dump + fields).
 * Returns the number of packets that matched a known table.
 */
int tm_decode_frames(const unsigned char *frames, size_t nframes,
		     char *out, size_t cap, int color, int full);

/*
 * Stateful variant, so a caller printing frames one at a time can show the
 * telemetry a frame produced directly beneath it. The decoder carries the
 * KISS reassembly state across feeds (packets straddle frames).
 */
struct tm_decoder;

struct tm_decoder *tm_decoder_open(void);
void tm_decoder_close(struct tm_decoder *dec);

/*
 * Feed one 223-byte frame; render the packets that completed within it into
 * `out` (NUL-terminated, truncated to `cap`), honouring `color` and `full` as
 * in tm_decode_frames. Returns the number of packets rendered.
 */
int tm_decoder_feed(struct tm_decoder *dec, const unsigned char *frame,
		    char *out, size_t cap, int color, int full);

/*
 * One decoded field, as filled by tm_decoder_feed_fields. The string pointers
 * reference static table storage (valid for the program's lifetime); `value`
 * is a formatted copy in the caller's array. `kind` is 0 for a number, 1 for
 * an enumeration / textual value, 2 for a raw hex value.
 */
struct tm_field_out {
	const char	*packet;	/* packet code, e.g. "SAT-S0"	*/
	const char	*cn;		/* field Chinese name		*/
	const char	*en;		/* field English name		*/
	const char	*unit;		/* unit, "" when none		*/
	char		value[64];
	unsigned char	kind;
};

/*
 * Feed one 223-byte frame and write the fields of every packet completed in
 * it into `out` (up to `cap` entries). Returns the number of fields written.
 */
int tm_decoder_feed_fields(struct tm_decoder *dec, const unsigned char *frame,
			   struct tm_field_out *out, size_t cap);

#endif /* ASRTU_TM_H */
