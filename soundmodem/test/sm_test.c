// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * sm_test - standalone parity harness for the sm_api decoder.
 *
 * Reads a 48 kHz mono 16-bit PCM WAV (12 kHz IF), feeds it to sm_decoder
 * and prints every decoded CCSDS frame. Not part of the shipped library;
 * used to prove the C port reproduces the upstream soundmodem decode.
 */

#include "sm_api.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t rd_u32(const uint8_t *p)
{
	return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
	       ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint16_t rd_u16(const uint8_t *p)
{
	return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

int main(int argc, char **argv)
{
	FILE *fp;
	uint8_t hdr[12];
	uint8_t ch[8];
	long data_off = -1;
	uint32_t data_len = 0;
	uint16_t channels = 0, bits = 0;
	uint32_t rate = 0;
	int16_t *samples;
	size_t n_samples, i, frames = 0;
	struct sm_decoder *dec;
	uint8_t frame[SM_FRAME_LEN];
	int corrected;

	if (argc < 2) {
		fprintf(stderr, "usage: %s in48k_mono.wav\n", argv[0]);
		return 2;
	}
	fp = fopen(argv[1], "rb");
	if (!fp) {
		perror("fopen");
		return 1;
	}
	if (fread(hdr, 1, 12, fp) != 12 || memcmp(hdr, "RIFF", 4) ||
	    memcmp(hdr + 8, "WAVE", 4)) {
		fprintf(stderr, "not a RIFF/WAVE file\n");
		return 1;
	}
	while (fread(ch, 1, 8, fp) == 8) {
		uint32_t sz = rd_u32(ch + 4);

		if (!memcmp(ch, "fmt ", 4)) {
			uint8_t fmt[16];

			if (fread(fmt, 1, 16, fp) != 16)
				break;
			channels = rd_u16(fmt + 2);
			rate = rd_u32(fmt + 4);
			bits = rd_u16(fmt + 14);
			if (sz > 16)
				fseek(fp, (long)sz - 16, SEEK_CUR);
		} else if (!memcmp(ch, "data", 4)) {
			data_off = ftell(fp);
			data_len = sz;
			fseek(fp, (long)sz, SEEK_CUR);
		} else {
			fseek(fp, (long)sz, SEEK_CUR);
		}
	}
	if (data_off < 0 || channels != 1 || bits != 16) {
		fprintf(stderr, "need mono 16-bit PCM (got ch=%u bits=%u)\n",
			channels, bits);
		return 1;
	}
	fprintf(stderr, "wav: %u Hz mono 16-bit, %u bytes\n", rate, data_len);

	n_samples = data_len / 2;
	samples = malloc(n_samples * sizeof(*samples));
	if (!samples) {
		fprintf(stderr, "oom\n");
		return 1;
	}
	fseek(fp, data_off, SEEK_SET);
	if (fread(samples, sizeof(*samples), n_samples, fp) != n_samples) {
		fprintf(stderr, "short read\n");
		return 1;
	}
	fclose(fp);

	dec = sm_decoder_new();
	if (!dec) {
		fprintf(stderr, "decoder alloc failed\n");
		return 1;
	}
	for (i = 0; i < n_samples; i += 1920) {
		size_t take = n_samples - i < 1920 ? n_samples - i : 1920;

		sm_decoder_feed(dec, samples + i, take);
		while (sm_decoder_poll(dec, frame, &corrected)) {
			frames++;
			printf("#%03zu corrected=%d  %02x %02x %02x %02x "
			       "%02x %02x %02x %02x\n",
			       frames, corrected, frame[0], frame[1], frame[2],
			       frame[3], frame[4], frame[5], frame[6],
			       frame[7]);
		}
	}
	fprintf(stderr, "TOTAL FRAMES: %zu\n", frames);

	sm_decoder_free(dec);
	free(samples);
	return frames > 0 ? 0 : 3;
}
