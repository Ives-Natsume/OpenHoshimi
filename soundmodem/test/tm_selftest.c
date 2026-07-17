/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * Standalone self-test for the ASRTU-1 telemetry decoder. It synthesises a
 * KISS frame carrying one SAT-S0 packet with known field values, decodes it
 * and checks a spread of value classes (enum, scaled signed/unsigned, hex,
 * two's-complement temperature). Not part of the crate build; compiled and run
 * directly under clang -Weverything during development.
 */

#include "tm.h"

#include <stdio.h>
#include <string.h>

#define FRAME_LEN 223u

static void put_be(unsigned char *p, size_t off, uint64_t v, unsigned int n)
{
	unsigned int i;

	for (i = 0; i < n; i++)
		p[off + i] = (unsigned char)(v >> ((n - 1 - i) * 8));
}

static int expect(const char *report, const char *needle)
{
	if (strstr(report, needle) != NULL) {
		printf("  ok: %s\n", needle);
		return 0;
	}
	printf("  FAIL: expected \"%s\"\n", needle);
	return 1;
}

int main(void)
{
	unsigned char frame[FRAME_LEN];
	unsigned char *body;
	char report[65536];
	int fails = 0;
	int npkt;

	memset(frame, 0, sizeof(frame));

	/* Transport header + no continuation, then a KISS delimiter. */
	frame[0] = 0x03;
	frame[1] = 0x24;
	frame[4] = 0x00;	/* continuation length */
	frame[5] = 0xC0;	/* FEND */

	/* Packet header: main type at offset 1, sub type at offset 6. */
	frame[6] = 0x08;
	frame[7] = 0xF0;	/* main = SAT */
	frame[12] = 0x00;	/* sub  = S0  */

	/* Packet body begins at frame[13]. */
	body = &frame[13];
	put_be(body, 0, 0xF0, 1);		/* main type echo	*/
	put_be(body, 1, 0x00, 1);		/* sub type echo	*/
	put_be(body, 2, 6797766u, 4);		/* OBC seconds		*/
	put_be(body, 6, 543u, 2);		/* OBC ms		*/
	put_be(body, 8, 0x12345678u, 4);	/* UTC seconds		*/
	body[12] = 0x81;			/* GNSS on, storage on	*/
	put_be(body, 18, 12000u, 2);		/* 12V bus = 12000 mV	*/
	put_be(body, 24, 0xFFCEu, 2);		/* 12V current = -50 mA	*/
	put_be(body, 56, 258u, 2);		/* rx command count	*/
	put_be(body, 60, 0xDEADBEEFu, 4);	/* last command (hex)	*/
	body[78] = 0x00;			/* not test mode	*/
	put_be(body, 83, 0x2222u, 2);		/* attitude: 三轴对日	*/
	body[97] = 0x04;			/* upload: 读取完成	*/
	put_be(body, 128, 123456u, 4);		/* euler = 12.3456	*/
	put_be(body, 164, 68000000u, 4);	/* semi-major = 6.8e6 m	*/
	put_be(body, 189, 0xAA, 1);		/* A 机当班		*/
	body[194] = 0xF0;			/* 3.3V temp = -16	*/
	body[200] = 0xCB;			/* frame tail		*/

	frame[13 + 201] = 0xC0;			/* close the packet	*/

	npkt = tm_decode_frames(frame, 1, report, sizeof(report), 0, 1);
	printf("decoded %d packet(s)\n\n%s\n", npkt, report);

	if (npkt != 1) {
		printf("FAIL: expected 1 packet, got %d\n", npkt);
		fails++;
	}
	fails += expect(report, "SAT-S0");
	fails += expect(report, "卫星平台");
	fails += expect(report, "第0类遥测");
	fails += expect(report, "= 6797766s");
	fails += expect(report, "= 12000mV");
	fails += expect(report, "= -50mA");
	fails += expect(report, "= 258");
	fails += expect(report, "0xDEADBEEF");
	fails += expect(report, "非测试模式");
	fails += expect(report, "三轴对日");
	fails += expect(report, "读取完成");
	fails += expect(report, "= 12.345600");
	fails += expect(report, "= 6800000.000000m");
	fails += expect(report, "A机当班");
	fails += expect(report, "= -16");
	fails += expect(report, "0xCB");

	printf("\n%s (%d failure(s))\n", fails ? "FAILED" : "PASSED", fails);
	return fails ? 1 : 0;
}
