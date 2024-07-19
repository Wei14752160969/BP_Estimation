#ifndef __VAFA_H
#define __VAFA_H
#include "sys.h"
#include "math.h"
#include "stm32f10x.h"


typedef union   
{
	float fdata;
	unsigned long ldata;
}FloatLongType;




void JustFloat_Send(float * fdata,uint16_t fdata_num,USART_TypeDef *Usart_choose);


#endif
