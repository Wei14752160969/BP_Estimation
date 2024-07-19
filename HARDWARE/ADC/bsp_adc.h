#ifndef _BSP_ADC_H_
#define _BSP_ADC_H_

#include "stm32f10x.h"
/*GPIO*/
#define ADCx_PORT_CLK  		RCC_APB2Periph_GPIOA
#define ADCx_PIN  				GPIO_Pin_6
#define ADCx_PORT					GPIOA

/*ADC*/
#define ADC_x							ADC1
#define ADCx_CLK  		    RCC_APB2Periph_ADC1
#define ADCx_CHx 		    	ADC_Channel_6

/*DMA*/
#define ADCx_DMA_CLK  		RCC_AHBPeriph_DMA1
#define ADCx_DMA_CHx 		  DMA1_Channel1


void ADCx_Init(void);

#endif //_BSP_ADC_H_
