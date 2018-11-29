#pragma once
#pragma pack (push, 1)
typedef struct {
    char id1;
    char id2;
    unsigned int file_size;
    unsigned int reserved;
    unsigned int bmp_data_offset;
    unsigned int bmp_header_size;
    unsigned int width;
    unsigned int height;
    unsigned short int planes;
    unsigned short int bits_per_pixel;
    unsigned int compression;
    unsigned int bmp_data_size;
    unsigned int h_resolution;
    unsigned int v_resolution;
    unsigned int colors;
    unsigned int important_colors;
} bmp_header;
#pragma pack(pop)
extern bool bmp_read(char *str, bmp_header *header, unsigned char **data);
extern bool bmp_write(char *str, bmp_header *header, unsigned char *data);
