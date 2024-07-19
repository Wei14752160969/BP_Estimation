#ifndef _BSP_LED_H_
#define _BSP_LED_H_
#include "stm32f10x.h"

#define LED_PORT_CLK  		RCC_APB2Periph_GPIOB
#define LED_PIN  					GPIO_Pin_12
#define LED_Output_Mode 	GPIO_Mode_Out_PP
#define LED_PORT					GPIOB

#define LED_Turn 			GPIOB->ODR^=LED_PIN 

void LED_Init(void);
void KEY_Init(void);

extern u8 INT_MARK;//中断标志位
void INT_INIT (void);

#endif //_LED_H_
