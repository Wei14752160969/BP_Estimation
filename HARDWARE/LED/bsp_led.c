#include "bsp_led.h"

uint8_t INT_MARK;//中断标志位
void LED_Init(void)
{
	GPIO_InitTypeDef GPIO_InitStruct;
	
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOB,ENABLE);
	
	GPIO_InitStruct.GPIO_Pin=GPIO_Pin_12;
	GPIO_InitStruct.GPIO_Mode=LED_Output_Mode;
	GPIO_InitStruct.GPIO_Speed=GPIO_Speed_50MHz;
	
	GPIO_Init(LED_PORT, &GPIO_InitStruct);
}	


void KEY_Init(void)
{
	GPIO_InitTypeDef GPIO_InitStruct;
	
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOA|RCC_APB2Periph_GPIOB,ENABLE);
	
	GPIO_InitStruct.GPIO_Pin=GPIO_Pin_8;
	GPIO_InitStruct.GPIO_Mode=GPIO_Mode_IPU;
	GPIO_InitStruct.GPIO_Speed=GPIO_Speed_50MHz;
	
	GPIO_Init(GPIOA, &GPIO_InitStruct);
	

	GPIO_InitStruct.GPIO_Pin=GPIO_Pin_9;
	GPIO_InitStruct.GPIO_Mode=GPIO_Mode_IPU;
	GPIO_InitStruct.GPIO_Speed=GPIO_Speed_50MHz;
	
	GPIO_Init(GPIOB, &GPIO_InitStruct);
}	



void INT_INIT (void){	 //按键中断初始化
	NVIC_InitTypeDef  NVIC_InitStruct;	//定义结构体变量
	EXTI_InitTypeDef  EXTI_InitStruct;
  GPIO_InitTypeDef  GPIO_InitStructure; 	
	
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_AFIO,ENABLE);        
  GPIO_PinRemapConfig(GPIO_Remap_SWJ_JTAGDisable, ENABLE);
	
  RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOB,ENABLE); 
   
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_15;                        
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IPU; //选择IO接口工作方式       
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz; //设置IO接口速度（2/10/50MHz） 
	GPIO_Init(GPIOB, &GPIO_InitStructure);		
	
	  
//第1个中断	
	GPIO_EXTILineConfig(GPIO_PortSourceGPIOB, GPIO_PinSource15);  //定义  GPIO 中断
	
	EXTI_InitStruct.EXTI_Line=EXTI_Line15;  //定义中断线
	EXTI_InitStruct.EXTI_LineCmd=ENABLE;              //中断使能
	EXTI_InitStruct.EXTI_Mode=EXTI_Mode_Interrupt;     //中断模式为中断
	EXTI_InitStruct.EXTI_Trigger=EXTI_Trigger_Falling;   //下降沿触发
	
	EXTI_Init(& EXTI_InitStruct);
	
	NVIC_InitStruct.NVIC_IRQChannel=EXTI15_10_IRQn;   //中断线
	NVIC_InitStruct.NVIC_IRQChannelCmd=ENABLE;  //使能中断
	NVIC_InitStruct.NVIC_IRQChannelPreemptionPriority=2;  //抢占优先级 2
	NVIC_InitStruct.NVIC_IRQChannelSubPriority=2;     //子优先级  2
	NVIC_Init(& NVIC_InitStruct);


}

void  EXTI15_10_IRQHandler(void)
{
	if(EXTI_GetITStatus(EXTI_Line15)!=RESET)//开入信号输入，开启定时器
		{
		  INT_MARK++;
			if(INT_MARK==3)INT_MARK=0;
			EXTI_ClearITPendingBit(EXTI_Line15); 
	  }     
	   
	
}



