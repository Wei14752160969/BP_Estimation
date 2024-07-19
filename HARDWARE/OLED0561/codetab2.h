#include "stm32f10x.h"

/***************************16*16的点阵字体取模方式：共阴——列行式——逆向输出*********/

/****************************************8*16的点阵************************************/
unsigned char zf[]=	  
{
  0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,// 0
	0x00,0x00,0x00,0xF8,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x33,0x00,0x00,0x00,0x00,//! 1
  0x00,0x10,0x0C,0x06,0x10,0x0C,0x06,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,//" 2
  0x40,0xC0,0x78,0x40,0xC0,0x78,0x40,0x00,0x04,0x3F,0x04,0x04,0x3F,0x04,0x04,0x00,//# 3
  0x00,0x70,0x88,0xFC,0x08,0x30,0x00,0x00,0x00,0x18,0x20,0xFF,0x21,0x1E,0x00,0x00,//$ 4
  0xF0,0x08,0xF0,0x00,0xE0,0x18,0x00,0x00,0x00,0x21,0x1C,0x03,0x1E,0x21,0x1E,0x00,//% 5
  0x00,0xF0,0x08,0x88,0x70,0x00,0x00,0x00,0x1E,0x21,0x23,0x24,0x19,0x27,0x21,0x10,//& 6
  0x10,0x16,0x0E,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,//' 7
  0x00,0x00,0x00,0xE0,0x18,0x04,0x02,0x00,0x00,0x00,0x00,0x07,0x18,0x20,0x40,0x00,//( 8
  0x00,0x02,0x04,0x18,0xE0,0x00,0x00,0x00,0x00,0x40,0x20,0x18,0x07,0x00,0x00,0x00,//) 9
  0x40,0x40,0x80,0xF0,0x80,0x40,0x40,0x00,0x02,0x02,0x01,0x0F,0x01,0x02,0x02,0x00,//* 10
  0x00,0x00,0x00,0xF0,0x00,0x00,0x00,0x00,0x01,0x01,0x01,0x1F,0x01,0x01,0x01,0x00,//+ 11
  0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x80,0xB0,0x70,0x00,0x00,0x00,0x00,0x00,//, 12
  0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x01,0x01,0x01,0x01,0x01,0x01,0x01,//- 13
  0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x30,0x30,0x00,0x00,0x00,0x00,0x00,//. 14
  0x00,0x00,0x00,0x00,0x80,0x60,0x18,0x04,0x00,0x60,0x18,0x06,0x01,0x00,0x00,0x00,/// 15
  0x00,0xE0,0x10,0x08,0x08,0x10,0xE0,0x00,0x00,0x0F,0x10,0x20,0x20,0x10,0x0F,0x00,//0 16
  0x00,0x10,0x10,0xF8,0x00,0x00,0x00,0x00,0x00,0x20,0x20,0x3F,0x20,0x20,0x00,0x00,//1 17
  0x00,0x70,0x08,0x08,0x08,0x88,0x70,0x00,0x00,0x30,0x28,0x24,0x22,0x21,0x30,0x00,//2 18
  0x00,0x30,0x08,0x88,0x88,0x48,0x30,0x00,0x00,0x18,0x20,0x20,0x20,0x11,0x0E,0x00,//3 19
  0x00,0x00,0xC0,0x20,0x10,0xF8,0x00,0x00,0x00,0x07,0x04,0x24,0x24,0x3F,0x24,0x00,//4 20
  0x00,0xF8,0x08,0x88,0x88,0x08,0x08,0x00,0x00,0x19,0x21,0x20,0x20,0x11,0x0E,0x00,//5 21
  0x00,0xE0,0x10,0x88,0x88,0x18,0x00,0x00,0x00,0x0F,0x11,0x20,0x20,0x11,0x0E,0x00,//6 22
  0x00,0x38,0x08,0x08,0xC8,0x38,0x08,0x00,0x00,0x00,0x00,0x3F,0x00,0x00,0x00,0x00,//7 23
  0x00,0x70,0x88,0x08,0x08,0x88,0x70,0x00,0x00,0x1C,0x22,0x21,0x21,0x22,0x1C,0x00,//8 24
  0x00,0xE0,0x10,0x08,0x08,0x10,0xE0,0x00,0x00,0x00,0x31,0x22,0x22,0x11,0x0F,0x00,//9 25
  0x00,0x00,0x00,0xC0,0xC0,0x00,0x00,0x00,0x00,0x00,0x00,0x30,0x30,0x00,0x00,0x00,//: 26
  0x00,0x00,0x00,0x80,0x00,0x00,0x00,0x00,0x00,0x00,0x80,0x60,0x00,0x00,0x00,0x00,//; 27
  0x00,0x00,0x80,0x40,0x20,0x10,0x08,0x00,0x00,0x01,0x02,0x04,0x08,0x10,0x20,0x00,//< 28
  0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x00,0x04,0x04,0x04,0x04,0x04,0x04,0x04,0x00,//= 29
  0x00,0x08,0x10,0x20,0x40,0x80,0x00,0x00,0x00,0x20,0x10,0x08,0x04,0x02,0x01,0x00,//> 30
  0x00,0x70,0x48,0x08,0x08,0x08,0xF0,0x00,0x00,0x00,0x00,0x30,0x36,0x01,0x00,0x00,//? 31
  0xC0,0x30,0xC8,0x28,0xE8,0x10,0xE0,0x00,0x07,0x18,0x27,0x24,0x23,0x14,0x0B,0x00,//@ 32
  0x00,0x00,0xC0,0x38,0xE0,0x00,0x00,0x00,0x20,0x3C,0x23,0x02,0x02,0x27,0x38,0x20,//A 33
  0x08,0xF8,0x88,0x88,0x88,0x70,0x00,0x00,0x20,0x3F,0x20,0x20,0x20,0x11,0x0E,0x00,//B 34
  0xC0,0x30,0x08,0x08,0x08,0x08,0x38,0x00,0x07,0x18,0x20,0x20,0x20,0x10,0x08,0x00,//C 35
  0x08,0xF8,0x08,0x08,0x08,0x10,0xE0,0x00,0x20,0x3F,0x20,0x20,0x20,0x10,0x0F,0x00,//D 36
  0x08,0xF8,0x88,0x88,0xE8,0x08,0x10,0x00,0x20,0x3F,0x20,0x20,0x23,0x20,0x18,0x00,//E 37
  0x08,0xF8,0x88,0x88,0xE8,0x08,0x10,0x00,0x20,0x3F,0x20,0x00,0x03,0x00,0x00,0x00,//F 38
  0xC0,0x30,0x08,0x08,0x08,0x38,0x00,0x00,0x07,0x18,0x20,0x20,0x22,0x1E,0x02,0x00,//G 39
  0x08,0xF8,0x08,0x00,0x00,0x08,0xF8,0x08,0x20,0x3F,0x21,0x01,0x01,0x21,0x3F,0x20,//H 40
  0x00,0x08,0x08,0xF8,0x08,0x08,0x00,0x00,0x00,0x20,0x20,0x3F,0x20,0x20,0x00,0x00,//I 41
  0x00,0x00,0x08,0x08,0xF8,0x08,0x08,0x00,0xC0,0x80,0x80,0x80,0x7F,0x00,0x00,0x00,//J 42
  0x08,0xF8,0x88,0xC0,0x28,0x18,0x08,0x00,0x20,0x3F,0x20,0x01,0x26,0x38,0x20,0x00,//K 43
  0x08,0xF8,0x08,0x00,0x00,0x00,0x00,0x00,0x20,0x3F,0x20,0x20,0x20,0x20,0x30,0x00,//L 44
  0x08,0xF8,0xF8,0x00,0xF8,0xF8,0x08,0x00,0x20,0x3F,0x00,0x3F,0x00,0x3F,0x20,0x00,//M 45
  0x08,0xF8,0x30,0xC0,0x00,0x08,0xF8,0x08,0x20,0x3F,0x20,0x00,0x07,0x18,0x3F,0x00,//N 46
  0xE0,0x10,0x08,0x08,0x08,0x10,0xE0,0x00,0x0F,0x10,0x20,0x20,0x20,0x10,0x0F,0x00,//O 47
  0x08,0xF8,0x08,0x08,0x08,0x08,0xF0,0x00,0x20,0x3F,0x21,0x01,0x01,0x01,0x00,0x00,//P 48
  0xE0,0x10,0x08,0x08,0x08,0x10,0xE0,0x00,0x0F,0x18,0x24,0x24,0x38,0x50,0x4F,0x00,//Q 49
  0x08,0xF8,0x88,0x88,0x88,0x88,0x70,0x00,0x20,0x3F,0x20,0x00,0x03,0x0C,0x30,0x20,//R 50
  0x00,0x70,0x88,0x08,0x08,0x08,0x38,0x00,0x00,0x38,0x20,0x21,0x21,0x22,0x1C,0x00,//S 51
  0x18,0x08,0x08,0xF8,0x08,0x08,0x18,0x00,0x00,0x00,0x20,0x3F,0x20,0x00,0x00,0x00,//T 52
  0x08,0xF8,0x08,0x00,0x00,0x08,0xF8,0x08,0x00,0x1F,0x20,0x20,0x20,0x20,0x1F,0x00,//U 53
  0x08,0x78,0x88,0x00,0x00,0xC8,0x38,0x08,0x00,0x00,0x07,0x38,0x0E,0x01,0x00,0x00,//V 54
  0xF8,0x08,0x00,0xF8,0x00,0x08,0xF8,0x00,0x03,0x3C,0x07,0x00,0x07,0x3C,0x03,0x00,//W 55
  0x08,0x18,0x68,0x80,0x80,0x68,0x18,0x08,0x20,0x30,0x2C,0x03,0x03,0x2C,0x30,0x20,//X 56
  0x08,0x38,0xC8,0x00,0xC8,0x38,0x08,0x00,0x00,0x00,0x20,0x3F,0x20,0x00,0x00,0x00,//Y 57
  0x10,0x08,0x08,0x08,0xC8,0x38,0x08,0x00,0x20,0x38,0x26,0x21,0x20,0x20,0x18,0x00,//Z 58
  0x00,0x00,0x00,0xFE,0x02,0x02,0x02,0x00,0x00,0x00,0x00,0x7F,0x40,0x40,0x40,0x00,//[ 59
  0x00,0x0C,0x30,0xC0,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x01,0x06,0x38,0xC0,0x00,//\\ 60
  0x00,0x02,0x02,0x02,0xFE,0x00,0x00,0x00,0x00,0x40,0x40,0x40,0x7F,0x00,0x00,0x00,//] 61
  0x00,0x00,0x04,0x02,0x02,0x02,0x04,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,//^ 62
  0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,//_ 63
  0x00,0x02,0x02,0x04,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,//` 64
  0x00,0x00,0x80,0x80,0x80,0x80,0x00,0x00,0x00,0x19,0x24,0x22,0x22,0x22,0x3F,0x20,//a 65
  0x08,0xF8,0x00,0x80,0x80,0x00,0x00,0x00,0x00,0x3F,0x11,0x20,0x20,0x11,0x0E,0x00,//b 66
  0x00,0x00,0x00,0x80,0x80,0x80,0x00,0x00,0x00,0x0E,0x11,0x20,0x20,0x20,0x11,0x00,//c 67
  0x00,0x00,0x00,0x80,0x80,0x88,0xF8,0x00,0x00,0x0E,0x11,0x20,0x20,0x10,0x3F,0x20,//d 68
  0x00,0x00,0x80,0x80,0x80,0x80,0x00,0x00,0x00,0x1F,0x22,0x22,0x22,0x22,0x13,0x00,//e 69
  0x00,0x80,0x80,0xF0,0x88,0x88,0x88,0x18,0x00,0x20,0x20,0x3F,0x20,0x20,0x00,0x00,//f 70
  0x00,0x00,0x80,0x80,0x80,0x80,0x80,0x00,0x00,0x6B,0x94,0x94,0x94,0x93,0x60,0x00,//g 71
  0x08,0xF8,0x00,0x80,0x80,0x80,0x00,0x00,0x20,0x3F,0x21,0x00,0x00,0x20,0x3F,0x20,//h 72
  0x00,0x80,0x98,0x98,0x00,0x00,0x00,0x00,0x00,0x20,0x20,0x3F,0x20,0x20,0x00,0x00,//i 73
  0x00,0x00,0x00,0x80,0x98,0x98,0x00,0x00,0x00,0xC0,0x80,0x80,0x80,0x7F,0x00,0x00,//j 74
  0x08,0xF8,0x00,0x00,0x80,0x80,0x80,0x00,0x20,0x3F,0x24,0x02,0x2D,0x30,0x20,0x00,//k 75
  0x00,0x08,0x08,0xF8,0x00,0x00,0x00,0x00,0x00,0x20,0x20,0x3F,0x20,0x20,0x00,0x00,//l 76
  0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x00,0x20,0x3F,0x20,0x00,0x3F,0x20,0x00,0x3F,//m 77
  0x80,0x80,0x00,0x80,0x80,0x80,0x00,0x00,0x20,0x3F,0x21,0x00,0x00,0x20,0x3F,0x20,//n 78
  0x00,0x00,0x80,0x80,0x80,0x80,0x00,0x00,0x00,0x1F,0x20,0x20,0x20,0x20,0x1F,0x00,//o 79
  0x80,0x80,0x00,0x80,0x80,0x00,0x00,0x00,0x80,0xFF,0xA1,0x20,0x20,0x11,0x0E,0x00,//p 80
  0x00,0x00,0x00,0x80,0x80,0x80,0x80,0x00,0x00,0x0E,0x11,0x20,0x20,0xA0,0xFF,0x80,//q 81
  0x80,0x80,0x80,0x00,0x80,0x80,0x80,0x00,0x20,0x20,0x3F,0x21,0x20,0x00,0x01,0x00,//r 82
  0x00,0x00,0x80,0x80,0x80,0x80,0x80,0x00,0x00,0x33,0x24,0x24,0x24,0x24,0x19,0x00,//s 83
  0x00,0x80,0x80,0xE0,0x80,0x80,0x00,0x00,0x00,0x00,0x00,0x1F,0x20,0x20,0x00,0x00,//t 84
  0x80,0x80,0x00,0x00,0x00,0x80,0x80,0x00,0x00,0x1F,0x20,0x20,0x20,0x10,0x3F,0x20,//u 85
  0x80,0x80,0x80,0x00,0x00,0x80,0x80,0x80,0x00,0x01,0x0E,0x30,0x08,0x06,0x01,0x00,//v 86
  0x80,0x80,0x00,0x80,0x00,0x80,0x80,0x80,0x0F,0x30,0x0C,0x03,0x0C,0x30,0x0F,0x00,//w 87
  0x00,0x80,0x80,0x00,0x80,0x80,0x80,0x00,0x00,0x20,0x31,0x2E,0x0E,0x31,0x20,0x00,//x 88
  0x80,0x80,0x80,0x00,0x00,0x80,0x80,0x80,0x80,0x81,0x8E,0x70,0x18,0x06,0x01,0x00,//y 89
  0x00,0x80,0x80,0x80,0x80,0x80,0x80,0x00,0x00,0x21,0x30,0x2C,0x22,0x21,0x30,0x00,//z 90
  0x00,0x00,0x00,0x00,0x80,0x7C,0x02,0x02,0x00,0x00,0x00,0x00,0x00,0x3F,0x40,0x40,//{ 91
  0x00,0x00,0x00,0x00,0xFF,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xFF,0x00,0x00,0x00,//| 92
  0x00,0x02,0x02,0x7C,0x80,0x00,0x00,0x00,0x00,0x40,0x40,0x3F,0x00,0x00,0x00,0x00,//} 93
  0x00,0x02,0x01,0x02,0x02,0x04,0x02,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,//~ 94	
};

/***************************16*16的点阵字体取模方式：共阴——列行式——逆向输出*********/
//unsigned char hz[]={
//	
//0x80,0x82,0x82,0x82,0xFE,0x82,0x82,0x82,0x82,0x82,0xFE,0x82,0x82,0x82,0x80,0x00,
//0x00,0x80,0x40,0x30,0x0F,0x00,0x00,0x00,0x00,0x00,0xFF,0x00,0x00,0x00,0x00,0x00,//开6,

//0x00,0x00,0x10,0x11,0x16,0x10,0x10,0xF0,0x10,0x10,0x14,0x13,0x10,0x00,0x00,0x00,
//0x81,0x81,0x41,0x41,0x21,0x11,0x0D,0x03,0x0D,0x11,0x21,0x41,0x41,0x81,0x81,0x00,//关7,

//0x00,0x00,0x00,0xFC,0x44,0x44,0x44,0x45,0x46,0x44,0x44,0x44,0x44,0x7C,0x00,0x00,
//0x40,0x20,0x18,0x07,0x00,0xFC,0x44,0x44,0x44,0x44,0x44,0x44,0x44,0xFC,0x00,0x00,//启8,

//0x40,0x44,0xC4,0x44,0x44,0x44,0x40,0x10,0x10,0xFF,0x10,0x10,0x10,0xF0,0x00,0x00,
//0x10,0x3C,0x13,0x10,0x14,0xB8,0x40,0x30,0x0E,0x01,0x40,0x80,0x40,0x3F,0x00,0x00,//动9,

//0x82,0x9A,0x96,0x93,0xFA,0x52,0x52,0x80,0x7E,0x12,0x12,0x12,0xF1,0x11,0x10,0x00,
//0x00,0x01,0x00,0xFE,0x93,0x92,0x93,0x92,0x92,0x92,0x92,0xFE,0x03,0x00,0x00,0x00,//暂10,

//0x80,0x60,0xF8,0x07,0x00,0x04,0x74,0x54,0x55,0x56,0x54,0x54,0x74,0x04,0x00,0x00,
//0x00,0x00,0xFF,0x00,0x03,0x01,0x05,0x45,0x85,0x7D,0x05,0x05,0x05,0x01,0x03,0x00,//停11,

//0x00,0x00,0xF8,0x88,0x88,0x88,0x88,0xFF,0x88,0x88,0x88,0x88,0xF8,0x00,0x00,0x00,
//0x00,0x00,0x1F,0x08,0x08,0x08,0x08,0x7F,0x88,0x88,0x88,0x88,0x9F,0x80,0xF0,0x00,//电12,

//0x10,0x60,0x02,0x8C,0x00,0xFE,0x02,0xF2,0x52,0x5A,0x56,0x52,0x52,0xF2,0x02,0x00,
//0x04,0x04,0x7E,0x41,0x30,0x0F,0x20,0x13,0x49,0x81,0x7F,0x01,0x09,0x13,0x20,0x00,//源13,

//0x10,0x94,0x53,0x32,0x1E,0x32,0x52,0x10,0x00,0x7E,0x42,0x42,0x42,0x7E,0x00,0x00,
//0x00,0x00,0x00,0xFF,0x49,0x49,0x49,0x49,0x49,0x49,0x49,0xFF,0x00,0x00,0x00,0x00,//智14,

//0x08,0xCC,0x4A,0x49,0x48,0x4A,0xCC,0x18,0x00,0x7F,0x88,0x88,0x84,0x82,0xE0,0x00,
//0x00,0xFF,0x12,0x12,0x52,0x92,0x7F,0x00,0x00,0x7E,0x88,0x88,0x84,0x82,0xE0,0x00,//能15,

//0x10,0x10,0x10,0xFF,0x90,0x20,0x98,0x48,0x28,0x09,0x0E,0x28,0x48,0xA8,0x18,0x00,
//0x02,0x42,0x81,0x7F,0x00,0x40,0x40,0x42,0x42,0x42,0x7E,0x42,0x42,0x42,0x40,0x00,//控16,

//0x40,0x50,0x4E,0x48,0x48,0xFF,0x48,0x48,0x48,0x40,0xF8,0x00,0x00,0xFF,0x00,0x00,
//0x00,0x00,0x3E,0x02,0x02,0xFF,0x12,0x22,0x1E,0x00,0x0F,0x40,0x80,0x7F,0x00,0x00,//制17,

//0x80,0x80,0x9E,0x92,0x92,0x92,0x9E,0xE0,0x80,0x9E,0xB2,0xD2,0x92,0x9E,0x80,0x00,
//0x08,0x08,0xF4,0x94,0x92,0x92,0xF1,0x00,0x01,0xF2,0x92,0x94,0x94,0xF8,0x08,0x00,//器18,

//0x40,0x7C,0x40,0x7F,0x48,0x48,0x40,0xF2,0x12,0x1A,0xD6,0x12,0x12,0xF2,0x02,0x00,
//0x90,0x8E,0x40,0x4F,0x20,0x1E,0x80,0x4F,0x20,0x18,0x07,0x10,0x20,0x4F,0x80,0x00,//频19,

//0x00,0x14,0xA4,0x44,0x24,0x34,0xAD,0x66,0x24,0x94,0x04,0x44,0xA4,0x14,0x00,0x00,
//0x08,0x09,0x08,0x08,0x09,0x09,0x09,0xFD,0x09,0x09,0x0B,0x08,0x08,0x09,0x08,0x00,//率20,

//0x00,0xF8,0x08,0xFF,0x08,0xF8,0x00,0x02,0x7A,0x4A,0x4A,0x4A,0x7A,0x02,0x02,0x00,
//0x00,0x0F,0x00,0xFF,0x08,0x0F,0x00,0xFF,0x49,0x49,0x7F,0x49,0x49,0xFF,0x00,0x00,//幅21,

//0x00,0x80,0x60,0xF8,0x07,0x04,0xE4,0xA4,0xA4,0xBF,0xA4,0xA4,0xE4,0x04,0x00,0x00,
//0x01,0x00,0x00,0xFF,0x40,0x40,0x7F,0x4A,0x4A,0x4A,0x4A,0x4A,0x7F,0x40,0x40,0x00,//值22,

//0x00,0x00,0xF8,0x88,0x88,0x88,0x88,0xFF,0x88,0x88,0x88,0x88,0xF8,0x00,0x00,0x00,
//0x00,0x00,0x1F,0x08,0x08,0x08,0x08,0x7F,0x88,0x88,0x88,0x88,0x9F,0x80,0xF0,0x00,//电23,

//0x00,0x00,0xFE,0x02,0x82,0x82,0x82,0x82,0xFA,0x82,0x82,0x82,0x82,0x82,0x02,0x00,
//0x80,0x60,0x1F,0x40,0x40,0x40,0x40,0x40,0x7F,0x40,0x40,0x44,0x58,0x40,0x40,0x00,//压24,

//0x10,0x60,0x02,0x8C,0x00,0x44,0x64,0x54,0x4D,0x46,0x44,0x54,0x64,0xC4,0x04,0x00,
//0x04,0x04,0x7E,0x01,0x80,0x40,0x3E,0x00,0x00,0xFE,0x00,0x00,0x7E,0x80,0xE0,0x00,//流25,

//0x00,0x00,0xFE,0x02,0x82,0x82,0x82,0x82,0xFA,0x82,0x82,0x82,0x82,0x82,0x02,0x00,
//0x80,0x60,0x1F,0x40,0x40,0x40,0x40,0x40,0x7F,0x40,0x40,0x44,0x58,0x40,0x40,0x00,//压0,

//0x00,0x10,0x10,0x10,0x10,0x10,0xFF,0x10,0x10,0x10,0x10,0x10,0xF0,0x00,0x00,0x00,
//0x00,0x80,0x40,0x20,0x18,0x06,0x01,0x00,0x20,0x40,0x80,0x40,0x3F,0x00,0x00,0x00,//力1,

//0x00,0x00,0x00,0xF8,0x88,0x8C,0x8A,0x89,0x88,0x88,0x88,0xF8,0x00,0x00,0x00,0x00,
//0x00,0x00,0x00,0xFF,0x44,0x44,0x44,0x44,0x44,0x44,0x44,0xFF,0x00,0x00,0x00,0x00,//自2,

//0x40,0x44,0xC4,0x44,0x44,0x44,0x40,0x10,0x10,0xFF,0x10,0x10,0x10,0xF0,0x00,0x00,
//0x10,0x3C,0x13,0x10,0x14,0xB8,0x40,0x30,0x0E,0x01,0x40,0x80,0x40,0x3F,0x00,0x00,//动3,

//0x00,0x00,0x24,0x24,0x24,0x24,0x24,0xFC,0x22,0x22,0x22,0x23,0x22,0x00,0x00,0x00,
//0x02,0x02,0x02,0x02,0x02,0x42,0x82,0x7F,0x02,0x02,0x02,0x02,0x02,0x02,0x02,0x00,//手4,

//0x10,0x60,0x02,0x0C,0xC0,0x02,0x1E,0xE2,0x02,0x02,0x02,0xE2,0x1E,0x00,0x00,0x00,
//0x04,0x04,0x7C,0x03,0x80,0x80,0x40,0x20,0x13,0x0C,0x13,0x20,0x40,0x80,0x80,0x00,//汉0,

//0x10,0x0C,0x04,0x24,0x24,0x24,0x25,0x26,0xA4,0x64,0x24,0x04,0x04,0x14,0x0C,0x00,
//0x02,0x02,0x02,0x02,0x02,0x42,0x82,0x7F,0x02,0x02,0x02,0x02,0x02,0x02,0x02,0x00,//字1,

//0x80,0x82,0x82,0x82,0x82,0x82,0x82,0xE2,0xA2,0x92,0x8A,0x86,0x82,0x80,0x80,0x00,
//0x00,0x00,0x00,0x00,0x00,0x40,0x80,0x7F,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,//子0,

//0x00,0x04,0x04,0x04,0x04,0x04,0x04,0xFC,0x04,0x04,0x04,0x04,0x04,0x04,0x00,0x00,
//0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x3F,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x00,//工1,

//0x00,0x80,0x60,0xF8,0x07,0x40,0x30,0x0F,0xF8,0x88,0x88,0x88,0x88,0x08,0x08,0x00,
//0x01,0x00,0x00,0xFF,0x00,0x00,0x00,0x00,0xFF,0x08,0x08,0x08,0x08,0x08,0x00,0x00,//作2,

//0x10,0x0C,0x24,0x24,0xA4,0x64,0x25,0x26,0x24,0x24,0xA4,0x24,0x24,0x14,0x0C,0x00,
//0x40,0x40,0x48,0x49,0x49,0x49,0x49,0x7F,0x49,0x49,0x49,0x4B,0x48,0x40,0x40,0x00,//室3,

//0x10,0x60,0x02,0x8C,0x00,0x44,0x54,0x54,0x54,0x7F,0x54,0x54,0x54,0x44,0x40,0x00,
//0x04,0x04,0x7E,0x01,0x00,0x00,0xFF,0x15,0x15,0x15,0x55,0x95,0x7F,0x00,0x00,0x00,//清0,

//0x00,0x00,0xFE,0x02,0x12,0x22,0xC2,0x02,0xC2,0x32,0x02,0xFE,0x00,0x00,0x00,0x00,
//0x80,0x60,0x1F,0x00,0x20,0x10,0x0C,0x03,0x0C,0x30,0x00,0x0F,0x30,0x40,0xF8,0x00,//风1,

//0x40,0x40,0x42,0xCC,0x00,0x40,0xA0,0x9E,0x82,0x82,0x82,0x9E,0xA0,0x20,0x20,0x00,
//0x00,0x00,0x00,0x3F,0x90,0x88,0x40,0x43,0x2C,0x10,0x28,0x46,0x41,0x80,0x80,0x00,//设4,

//0x40,0x40,0x42,0xCC,0x00,0x40,0x40,0x40,0x40,0xFF,0x40,0x40,0x40,0x40,0x40,0x00,
//0x00,0x00,0x00,0x7F,0x20,0x10,0x00,0x00,0x00,0xFF,0x00,0x00,0x00,0x00,0x00,0x00,//计5,
//0x20,0x30,0xAC,0x63,0x20,0x18,0x80,0x82,0x42,0x22,0x12,0x1A,0x26,0x42,
//0x80,0x00,0x22,0x67,0x22,0x12,0x12,0x12,0x40,0x42,0x42,0x42,0x7E,0x42,
//0x42,0x42,0x40,0x00,//经0,

//0x20,0x30,0xAC,0x63,0x20,0x18,0x00,0x08,0x48,0x48,0xFF,0x48,0x48,0x48,
//0x08,0x00,0x22,0x67,0x22,0x12,0x12,0x12,0x00,0x02,0x02,0x02,0xFF,0x02,
//0x12,0x22,0x1E,0x00,//纬1,

//0x00,0x00,0xFC,0x24,0x24,0x24,0xFC,0x25,0x26,0x24,0xFC,0x24,0x24,0x24,
//0x04,0x00,0x40,0x30,0x8F,0x80,0x84,0x4C,0x55,0x25,0x25,0x25,0x55,0x4C,
//0x80,0x80,0x80,0x00,//度2,

//0x00,0x08,0x30,0x00,0xFF,0x20,0x20,0x20,0x20,0xFF,0x20,0x20,0x22,0x2C,
//0x20,0x00,0x04,0x04,0x02,0x01,0xFF,0x80,0x40,0x30,0x0E,0x01,0x06,0x18,
//0x20,0x40,0x80,0x00,//状0,

//0x00,0x04,0x84,0x84,0x44,0x24,0x54,0x8F,0x14,0x24,0x44,0x84,0x84,0x04,
//0x00,0x00,0x41,0x39,0x00,0x00,0x3C,0x40,0x40,0x42,0x4C,0x40,0x40,0x70,
//0x04,0x09,0x31,0x00,//态1,

//0x80,0x80,0x88,0x88,0x88,0x88,0x88,0xFF,0x88,0x88,0x88,0x88,0x88,0x80,
//0x80,0x00,0x20,0x20,0x10,0x08,0x04,0x02,0x01,0xFF,0x01,0x02,0x04,0x08,
//0x10,0x20,0x20,0x00,//未2,

//0x10,0x10,0x10,0xFF,0x10,0x90,0xA4,0x44,0x24,0xB5,0x6E,0xA4,0x14,0x44,
//0xA4,0x00,0x04,0x44,0x82,0x7F,0x01,0x00,0x08,0x08,0x09,0x09,0xFD,0x09,
//0x0B,0x08,0x08,0x00,//摔3,

//0x80,0x60,0xF8,0x07,0x04,0x64,0x5C,0xC4,0x64,0x44,0x00,0xF8,0x00,0xFF,
//0x00,0x00,0x00,0x00,0xFF,0x00,0x20,0x62,0x22,0x1F,0x12,0x12,0x00,0x4F,
//0x80,0x7F,0x00,0x00,//倒4,

//0x40,0x40,0x42,0xCC,0x00,0x04,0xF4,0x94,0x94,0xFF,0x94,0x94,0xF4,0x04,
//0x00,0x00,0x00,0x40,0x20,0x1F,0x20,0x48,0x44,0x42,0x41,0x5F,0x41,0x42,
//0x44,0x48,0x40,0x00,//速0,

//0x04,0x04,0x04,0x04,0xF4,0x94,0x95,0x96,0x94,0x94,0xF4,0x04,0x04,0x04,
//0x04,0x00,0x00,0xFE,0x02,0x02,0x7A,0x4A,0x4A,0x4A,0x4A,0x4A,0x7A,0x02,
//0x82,0xFE,0x00,0x00,//高1,

//      0x10,0x22,0x64,0x0C,0x90,0x08,0xF7,0x14,
//      0x34,0x54,0x14,0x14,0xF6,0x04,0x00,0x00,
//      0x04,0x04,0xFE,0x01,0x01,0x1F,0x11,0x11,
//      0x13,0x15,0x51,0x91,0x7F,0x11,0x01,0x00,//"海", 

//      0x08,0x08,0x08,0xFF,0x88,0x48,0x10,0x90,
//      0xFF,0x90,0x91,0x96,0x90,0x18,0x10,0x00,
//      0x02,0x42,0x81,0x7F,0x40,0x30,0x8C,0x83,
//      0x44,0x28,0x18,0x24,0x43,0xC0,0x40,0x00,//"拔",


//};

//unsigned char hz_index[]={"开关启动暂停电源智能控制器频率幅值电压流压力自动手汉字子工作室清风设计经纬度状态未摔倒速高海拔"};


unsigned char hz[]=
	{
		
//"智", 
		0x20,0x28,0x27,0xE4,0x3C,0xA4,0x26,0x24,
      0x20,0xFC,0x84,0x84,0x84,0xFE,0x04,0x00,
      0x04,0x02,0x01,0xFC,0x54,0x54,0x57,0x54,
      0x54,0x55,0x54,0xFE,0x04,0x01,0x00,0x00,

//"能", 
		  0x10,0xD8,0x54,0x53,0x50,0xDC,0x30,0x00,
      0x7F,0x90,0x88,0x84,0x86,0xE0,0x00,0x00,
      0x00,0xFF,0x09,0x49,0x89,0x7F,0x00,0x00,
      0x7E,0x90,0x88,0x84,0x86,0x80,0xE0,0x00,

//"婴", 
		  0x00,0x80,0xBE,0x42,0x3A,0x42,0xBE,0x00,
      0xBE,0x42,0x3A,0x42,0xBF,0x02,0x00,0x00,
      0x02,0x02,0x82,0x82,0x8A,0x56,0x23,0x22,
      0x22,0x52,0x4E,0x82,0x02,0x03,0x02,0x00,

//"儿", 
		  0x00,0x00,0x00,0x00,0x00,0xFF,0x00,0x00,
      0x00,0xFF,0x00,0x00,0x00,0x00,0x00,0x00,
      0x80,0x40,0x20,0x10,0x0C,0x03,0x00,0x00,
      0x00,0x3F,0x40,0x40,0x40,0x40,0x78,0x00,

//"床", 
      0x00,0x00,0xFC,0x44,0x44,0x44,0x44,0x45,
      0xFE,0x44,0x44,0x44,0x64,0x46,0x04,0x00,
      0x40,0x30,0x0F,0x20,0x10,0x08,0x04,0x03,
      0xFF,0x01,0x02,0x04,0x18,0x30,0x10,0x00,

//"环", 
      0x42,0x42,0xFE,0x43,0x42,0x04,0x04,0x04,
      0x84,0xE4,0x1C,0x84,0x04,0x06,0x04,0x00,
      0x20,0x60,0x3F,0x10,0x10,0x04,0x02,0x01,
      0x00,0xFF,0x00,0x00,0x01,0x03,0x06,0x00,

//"境", 
      0x20,0x20,0xFF,0x20,0x20,0x24,0xA4,0xAC,
      0xB5,0xA6,0xB4,0xAC,0xE6,0xB4,0x20,0x00,
      0x10,0x30,0x1F,0x08,0x88,0x80,0x4F,0x3A,
      0x0A,0x0A,0x7A,0x8A,0x8F,0x80,0xE0,0x00,

//"温", 
      0x10,0x22,0x64,0x0C,0x80,0x00,0xFE,0x92,
      0x92,0x92,0x92,0x92,0xFF,0x02,0x00,0x00,
      0x04,0x04,0xFE,0x01,0x40,0x7E,0x42,0x42,
      0x7E,0x42,0x7E,0x42,0x42,0x7E,0x40,0x00,

//"度", 
      0x00,0x00,0xFC,0x24,0x24,0x24,0xFC,0xA5,
      0xA6,0xA4,0xFC,0x24,0x34,0x26,0x04,0x00,
      0x40,0x20,0x9F,0x80,0x42,0x42,0x26,0x2A,
      0x12,0x2A,0x26,0x42,0x40,0xC0,0x40,0x00,

//"湿", 
      0x10,0x22,0x64,0x0C,0x80,0xFE,0x92,0x92,
      0x92,0x92,0x92,0x92,0xFF,0x02,0x00,0x00,
      0x04,0x04,0xFE,0x41,0x44,0x48,0x50,0x7F,
      0x40,0x40,0x7F,0x50,0x48,0x64,0x40,0x00,
			
//"分", 
      0x00,0x80,0x40,0x20,0x98,0x86,0x80,0x80,
      0x83,0x8C,0x90,0x20,0xC0,0x80,0x80,0x00,
      0x01,0x00,0x80,0x40,0x20,0x1F,0x00,0x40,
      0x80,0x40,0x3F,0x00,0x00,0x01,0x00,0x00,

//"贝", 
      0x00,0x00,0x00,0xFE,0x02,0x02,0x02,0xFA,
      0x02,0x02,0x02,0xFF,0x02,0x00,0x00,0x00,
      0x00,0x00,0x80,0x8F,0x40,0x20,0x18,0x07,
      0x00,0x10,0x20,0x4F,0xC0,0x00,0x00,0x00,

//"值", 
      0x80,0x40,0x20,0xF8,0x07,0x04,0xE4,0xA4,
      0xA4,0xBF,0xA4,0xA4,0xF6,0x24,0x00,0x00,
      0x00,0x00,0x00,0xFF,0x40,0x40,0x7F,0x4A,
      0x4A,0x4A,0x4A,0x4A,0x7F,0x40,0x40,0x00,

//"高", 
      0x04,0x04,0x04,0xF4,0x94,0x94,0x95,0x96,
      0x94,0x94,0x94,0xF4,0x04,0x06,0x04,0x00,
      0x00,0xFE,0x02,0x02,0x7A,0x4A,0x4A,0x4A,
      0x4A,0x4A,0x7A,0x02,0x82,0xFF,0x02,0x00,

//"底", 
      0x00,0x00,0xFC,0x04,0xE4,0x24,0x25,0x26,
      0xE4,0x14,0x14,0x14,0x84,0x06,0x04,0x00,
      0x40,0x30,0x0F,0x00,0x7F,0x21,0x11,0x21,
      0x67,0x19,0x21,0x41,0x81,0xE1,0x00,0x00,

//"检", 
      0x10,0x10,0xD0,0xFF,0x10,0x50,0x20,0x50,
      0x4C,0x43,0x4C,0x50,0x20,0x60,0x20,0x00,
      0x04,0x03,0x00,0xFF,0x41,0x42,0x42,0x5C,
      0x40,0x5F,0x40,0x50,0x4E,0x64,0x40,0x00,

//"测", 
      0x10,0x22,0x6C,0x00,0x80,0xFC,0x04,0xF4,
      0x04,0xFE,0x04,0xF8,0x00,0xFE,0x00,0x00,
      0x04,0x04,0xFE,0x01,0x40,0x27,0x10,0x0F,
      0x10,0x67,0x00,0x47,0x80,0x7F,0x00,0x00,

//"哭", 
      0x00,0x00,0x7E,0x22,0x22,0x22,0x7F,0x82,
      0x7E,0xA2,0x22,0x22,0x7F,0x02,0x00,0x00,
      0x82,0x82,0x42,0x42,0x22,0x12,0x0A,0x07,
      0x0A,0x12,0x23,0x22,0x42,0xC3,0x42,0x00,

//"泣", 
      0x20,0xC2,0x04,0x0C,0x80,0x08,0xE8,0x08,
      0x09,0x0E,0x08,0x08,0xEC,0x08,0x00,0x00,
      0x04,0x04,0xFE,0x01,0x40,0x40,0x41,0x5E,
      0x40,0x60,0x58,0x47,0x40,0x60,0x40,0x00,

//"尿", 
      0x00,0x00,0xFE,0x12,0x92,0x92,0x92,0x12,
      0xF2,0x12,0x92,0x12,0x12,0xBF,0x02,0x00,
      0x40,0x30,0x0F,0x20,0x10,0x0C,0x43,0x80,
      0x7F,0x00,0x07,0x0A,0x11,0x30,0x10,0x00,

  };
unsigned char hz_index[]={"智能婴儿床环境温度湿分贝值高低检测哭泣尿"};






//注意：下面字库声明的汉字顺序必须和取字模的顺序相同，且不能有相同的字

char zf_index[]={" !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"};
