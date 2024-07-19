#ifndef _BSP_ADVANCED_TIMER_H_
#define _BSP_ADVANCED_TIMER_H_

#include "stm32f10x.h"
/*GPIO*/
#define TIMx_CH1_GPIO_PORT 				GPIOA
#define TIMx_CH1_GPIO_PIN 				GPIO_Pin_8
#define TIMx_CH1_GPIO_CLK     		RCC_APB2Periph_GPIOA
#define TIMx_CH1_GPIO_CLK_CMD 		RCC_APB2PeriphClockCmd

#define TIMx_CH1N_GPIO_PORT 			GPIOB
#define TIMx_CH1N_GPIO_PIN 				GPIO_Pin_14
#define TIMx_CH1N_GPIO_CLK     		RCC_APB2Periph_GPIOB
#define TIMx_CH1N_GPIO_CLK_CMD 		RCC_APB2PeriphClockCmd

#define TIMx_BKIN_GPIO_PORT 			GPIOB
#define TIMx_BKIN_GPIO_PIN 				GPIO_Pin_12
#define TIMx_BKIN_GPIO_CLK     		RCC_APB2Periph_GPIOB
#define TIMx_BKIN_GPIO_CLK_CMD 		RCC_APB2PeriphClockCmd
/*Advanced Timer*/

#define ADVANCED_TIM 							TIM1
#define ADVANCED_TIM_CLK					RCC_APB2Periph_TIM1

#define ADVANCED_TIM_PSC_SET 			36
#define ADVANCED_TIM_PSC 					(ADVANCED_TIM_PSC_SET-1)

#define ADVANCED_TIM_ARR_SET 			65536
#define ADVANCED_TIM_ARR 					(ADVANCED_TIM_ARR_SET-1)

#define ADVANCED_TIM_CCR_SET 			12000
#define ADVANCED_TIM_CCR					(ADVANCED_TIM_CCR_SET)


void AdvancedTim_Init(void);

#endif//_BSP_ADVANCED_TIMER_H_
