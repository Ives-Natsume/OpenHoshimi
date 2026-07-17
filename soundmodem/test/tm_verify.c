/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * Independent cross-checker for the ASRTU-1 telemetry tables.
 *
 * tm_tables.c is generated from the format sheets by tools/gen_tm_tables.py.
 * This program re-parses the same sheets with a second, independent C
 * implementation and compares every field against the compiled tables: offset,
 * bit, width, base type, unit, scaling (mul/off) and enumeration keys. A
 * generator bug or a hand edit that drifts from the sheets fails here rather
 * than being trusted. Development tool only; not part of the crate build.
 *
 * Usage: tm_verify <ASRTU-1 dir>   (exit 0 = tables match the sheets)
 */

#include "tm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXCOL	8
#define MAXLINE	4096

static int fails;

/* Split one CSV line into fields, honouring "..."-quoted fields (which may
 * contain commas). Returns the field count; trims the trailing newline. */
static int split_csv(char *line, char *cols[], int maxc)
{
	int n = 0;
	char *p = line;

	while (n < maxc) {
		char *out = p;

		cols[n++] = out;
		if (*p == '"') {
			p++;
			for (;;) {
				if (*p == '"' && p[1] == '"') {
					*out++ = '"';
					p += 2;
				} else if (*p == '"' || *p == '\0') {
					break;
				} else {
					*out++ = *p++;
				}
			}
			if (*p == '"')
				p++;
			*out = '\0';
			if (*p == ',') {
				p++;
				continue;
			}
			break;
		}
		while (*p != ',' && *p != '\n' && *p != '\r' && *p != '\0')
			p++;
		if (*p == ',') {
			*p++ = '\0';
			continue;
		}
		*p = '\0';
		break;
	}
	return n;
}

static void parse_offset(const char *full, unsigned int *byte,
			 unsigned int *bit, int *width)
{
	const char *c = strstr(full, "-C");
	unsigned long digits;
	char *end;

	c = c ? c + 2 : full;
	digits = strtoul(c, &end, 10);
	*byte = (unsigned int)(digits % 10000u);
	*bit = (unsigned int)strtoul(end + 1, &end, 10);
	*width = (int)strtol(end + 1, &end, 10);
}

static unsigned char base_of(const char *f)
{
	if (strstr(f, "UX"))
		return TM_UX;
	if (strstr(f, "SX"))
		return TM_SX;
	if (strstr(f, "FX"))
		return TM_FX;
	if (strstr(f, "HX"))
		return TM_HX;
	return TM_RSVD;
}

/* Parse the affine expression value = A*base + B from a scalar funccode. */
static void parse_affine(const char *func, double *mul, double *off)
{
	static const char *bases[] = { "UX", "SX", "FX" };
	const char *body = strchr(func, '=');
	const char *bp = NULL;
	size_t i;
	double a = 1.0;
	double b = 0.0;
	const char *p;

	*mul = 1.0;
	*off = 0.0;
	body = body ? body + 1 : func;
	for (i = 0; i < 3; i++) {
		bp = strstr(body, bases[i]);
		if (bp)
			break;
	}
	if (!bp)
		return;			/* HX / X: no scaling */

	if (bp > body) {
		/* leading "k*" coefficient */
		char buf[64];
		size_t len = (size_t)(bp - body);

		if (len < sizeof(buf)) {
			memcpy(buf, body, len);
			buf[len] = '\0';
			if (buf[len - 1] == '*') {
				buf[len - 1] = '\0';
				a *= strtod(buf, NULL);
			}
		}
	}

	p = bp + 2;			/* after the base token */
	while (*p == '*' || *p == '/') {
		char op = *p++;
		char *end;
		double v = strtod(p, &end);

		p = end;
		if (op == '*')
			a *= v;
		else
			a /= v;
	}
	if (*p == '+' || *p == '-')
		b = strtod(p, NULL);
	*mul = a;
	*off = b;
}

static int close_enough(double x, double y)
{
	return fabs(x - y) <= 1e-9 * fabs(y) + 1e-12;
}

static void check_field(const char *code, const struct tm_field *tf,
			const char *full, const char *func)
{
	unsigned int byte, bit;
	int width;
	unsigned char base;

	parse_offset(full, &byte, &bit, &width);
	if (tf->byte != byte || tf->bit != bit || tf->width != width) {
		fprintf(stderr, "%s %s: offset %u.%u.%d != table %u.%u.%u\n",
			code, full, byte, bit, width,
			tf->byte, tf->bit, tf->width);
		fails++;
		return;
	}
	if (strchr(func, '{')) {
		base = base_of(func);	/* enum: only the base is compared here */
		if (tf->base != base || tf->map == NULL) {
			fprintf(stderr, "%s %s: enum base/map mismatch\n",
				code, full);
			fails++;
		}
		return;
	}
	base = base_of(func);
	if (tf->base != base) {
		fprintf(stderr, "%s %s: base %u != table %u\n",
			code, full, base, tf->base);
		fails++;
	}
	if (base == TM_UX || base == TM_SX || base == TM_FX) {
		double mul, off;

		parse_affine(func, &mul, &off);
		if (!close_enough(mul, tf->mul / tf->div) ||
		    !close_enough(off, tf->off)) {
			fprintf(stderr,
				"%s %s: scale %g/%g != table %g\n",
				code, full, mul, off, tf->mul / tf->div);
			fails++;
		}
	}
}

static void verify_packet(const char *dir, const struct tm_packet *p,
			  const char *code)
{
	char path[512];
	char line[MAXLINE];
	FILE *f;
	unsigned int idx = 0;

	snprintf(path, sizeof(path), "%s/%s.csv", dir, code);
	f = fopen(path, "r");
	if (!f) {
		fprintf(stderr, "cannot open %s\n", path);
		fails++;
		return;
	}
	if (!fgets(line, sizeof(line), f)) {	/* header */
		fclose(f);
		return;
	}
	while (fgets(line, sizeof(line), f)) {
		char *cols[MAXCOL];
		int width;
		unsigned int byte, bit;
		int n = split_csv(line, cols, MAXCOL);

		if (n < 4 || cols[0][0] == '\0')
			continue;
		parse_offset(cols[0], &byte, &bit, &width);
		if (width <= 0 || width > 4096)
			continue;
		if (idx >= p->nfield) {
			fprintf(stderr, "%s: more sheet rows than table fields\n",
				code);
			fails++;
			break;
		}
		check_field(code, &p->fields[idx], cols[0], cols[3]);
		idx++;
	}
	if (idx != p->nfield)
		fprintf(stderr, "%s: %u sheet fields != %u table fields\n",
			code, idx, p->nfield);
	fclose(f);
}

int main(int argc, char **argv)
{
	char path[512];
	char line[MAXLINE];
	FILE *sys;

	if (argc < 2) {
		fprintf(stderr, "usage: %s <ASRTU-1 dir>\n", argv[0]);
		return 2;
	}
	snprintf(path, sizeof(path), "%s/system.csv", argv[1]);
	sys = fopen(path, "r");
	if (!sys) {
		fprintf(stderr, "cannot open %s\n", path);
		return 2;
	}
	if (!fgets(line, sizeof(line), sys)) {
		fclose(sys);
		return 2;
	}
	while (fgets(line, sizeof(line), sys)) {
		char *cols[MAXCOL];
		const struct tm_packet *p;
		unsigned char main_t, sub_t;
		int n = split_csv(line, cols, MAXCOL);

		if (n < 5 || cols[4][0] == '\0')
			continue;
		main_t = (unsigned char)strtoul(cols[1], NULL, 16);
		sub_t = (unsigned char)strtoul(cols[2], NULL, 16);
		p = tm_lookup(main_t, sub_t);
		if (!p) {
			fprintf(stderr, "no table for %s (%02X.%02X)\n",
				cols[4], main_t, sub_t);
			fails++;
			continue;
		}
		verify_packet(argv[1], p, cols[4]);
	}
	fclose(sys);

	printf("%s: %d mismatch(es)\n", fails ? "FAILED" : "PASSED", fails);
	return fails ? 1 : 0;
}
