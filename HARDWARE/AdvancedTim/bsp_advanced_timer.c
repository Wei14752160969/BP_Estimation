#include "bsp_advanced_timer.h"

static void AdvancedTim_GPIO_Config(void)
{
	GPIO_InitTypeDef GPIO_InitStruct;
	/*开时钟*/
	TIMx_CH1_GPIO_CLK_CMD(TIMx_CH1_GPIO_CLK, ENABLE);
	TIMx_CH1N_GPIO_CLK_CMD(TIMx_CH1N_GPIO_CLK, ENABLE);
	TIMx_BKIN_GPIO_CLK_CMD(TIMx_BKIN_GPIO_CLK ,ENABLE);
	/*设置参数1*/
	GPIO_InitStruct.GPIO_Mode=GPIO_Mode_AF_PP;
	GPIO_InitStruct.GPIO_Pin=TIMx_CH1_GPIO_PIN;
	GPIO_InitStruct.GPIO_Speed=GPIO_Speed_50MHz;
	/*写参数*/
	GPIO_Init(TIMx_CH1_GPIO_PORT, &GPIO_InitStruct);
	
	
	/*设置参数2*/
	GPIO_InitStruct.GPIO_Pin=TIMx_CH1N_GPIO_PIN;
	/*写参数*/
	GPIO_Init(TIMx_CH1N_GPIO_PORT, &GPIO_InitStruct);
	
	
	/*设置参数3*/
	GPIO_InitStruct.GPIO_Mode=GPIO_Mode_AF_PP;
	GPIO_InitStruct.GPIO_Pin=TIMx_BKIN_GPIO_PIN;
	/*写参数*/
	GPIO_Init(TIMx_BKIN_GPIO_PORT, &GPIO_InitStruct);
	
}

static void AdvancedTim_Config(void)
{
	TIM_TimeBaseInitTypeDef TIM_TimeBaseInitStruct;
	TIM_OCInitTypeDef TIM_OCInitStruct;
	TIM_BDTRInitTypeDef TIM_BDTRInitStruct;
	/*开时钟*/
	RCC_APB2PeriphClockCmd(ADVANCED_TIM_CLK, ENABLE);
	/*配置时基参数*/
	TIM_TimeBaseInitStruct.TIM_Prescaler=ADVANCED_TIM_PSC;/*预分频因子*/
	TIM_TimeBaseInitStruct.TIM_CounterMode=TIM_CounterMode_Up;/*向上计数*/
	TIM_TimeBaseInitStruct.TIM_Period=ADVANCED_TIM_ARR;/*周期*/
	TIM_TimeBaseInitStruct.TIM_ClockDivision=TIM_CKD_DIV1;/*Tdts:这里会与死区时间有关*/
	TIM_TimeBaseInitStruct.TIM_RepetitionCounter=0;/*不使用重复计数器*/
	/*写参数*/
	TIM_TimeBaseInit(ADVANCED_TIM, &TIM_TimeBaseInitStruct);
	
	/*配置输出比较的参数*/
	TIM_OCInitStruct.TIM_OCMode=TIM_OCMode_PWM1;/*PWM模式一*/
	TIM_OCInitStruct.TIM_OutputState=TIM_OutputState_Enable;/*主通道使能*/
	TIM_OCInitStruct.TIM_OutputNState=TIM_OutputNState_Enable;/*互补通道使能*/
	TIM_OCInitStruct.TIM_Pulse=ADVANCED_TIM_CCR;/*占空比*/
	TIM_OCInitStruct.TIM_OCPolarity=TIM_OCPolarity_High;/*主通道高电平为有效*/
	TIM_OCInitStruct.TIM_OCNPolarity=TIM_OCNPolarity_High;/*互补通道高电平为有效*/
	TIM_OCInitStruct.TIM_OCIdleState=TIM_OCIdleState_Reset;/*刹车后的输出状态*/
	TIM_OCInitStruct.TIM_OCNIdleState=TIM_OCNIdleState_Reset;
	/*写参数*/
	TIM_OC1Init(ADVANCED_TIM, &TIM_OCInitStruct);
	TIM_OC1PreloadConfig(ADVANCED_TIM, TIM_OCPreload_Enable);
	
	/*配置死区刹车寄存器*/
	TIM_BDTRInitStruct.TIM_OSSRState=TIM_OSSRState_Enable;
	TIM_BDTRInitStruct.TIM_OSSIState=TIM_OSSIState_Enable;
	TIM_BDTRInitStruct.TIM_LOCKLevel=TIM_LOCKLevel_OFF;
	TIM_BDTRInitStruct.TIM_DeadTime=7;	/*死区时间97ns*/
	TIM_BDTRInitStruct.TIM_Break=TIM_Break_Enable;/*使能刹车功能*/
	TIM_BDTRInitStruct.TIM_BreakPolarity=TIM_BreakPolarity_High;/*当刹车通道为高电平时停止输出*/
	TIM_BDTRInitStruct.TIM_AutomaticOutput=TIM_AutomaticOutput_Enable;
	TIM_BDTRConfig(ADVANCED_TIM, &TIM_BDTRInitStruct);
	
	TIM_Cmd(ADVANCED_TIM, ENABLE);
	TIM_CtrlPWMOutputs(ADVANCED_TIM, ENABLE);
}

void AdvancedTim_Init(void)
{
		AdvancedTim_GPIO_Config();
		AdvancedTim_Config();
}
