#include "bsp_led.h"

uint8_t INT_MARK;//�жϱ�־λ
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



void INT_INIT (void){	 //�����жϳ�ʼ��
	NVIC_InitTypeDef  NVIC_InitStruct;	//����ṹ�����
	EXTI_InitTypeDef  EXTI_InitStruct;
  GPIO_InitTypeDef  GPIO_InitStructure; 	
	
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_AFIO,ENABLE);        
  GPIO_PinRemapConfig(GPIO_Remap_SWJ_JTAGDisable, ENABLE);
	
  RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOB,ENABLE); 
   
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_15;                        
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IPU; //ѡ��IO�ӿڹ�����ʽ       
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz; //����IO�ӿ��ٶȣ�2/10/50MHz�� 
	GPIO_Init(GPIOB, &GPIO_InitStructure);		
	
	  
//��1���ж�	
	GPIO_EXTILineConfig(GPIO_PortSourceGPIOB, GPIO_PinSource15);  //����  GPIO �ж�
	
	EXTI_InitStruct.EXTI_Line=EXTI_Line15;  //�����ж���
	EXTI_InitStruct.EXTI_LineCmd=ENABLE;              //�ж�ʹ��
	EXTI_InitStruct.EXTI_Mode=EXTI_Mode_Interrupt;     //�ж�ģʽΪ�ж�
	EXTI_InitStruct.EXTI_Trigger=EXTI_Trigger_Falling;   //�½��ش���
	
	EXTI_Init(& EXTI_InitStruct);
	
	NVIC_InitStruct.NVIC_IRQChannel=EXTI15_10_IRQn;   //�ж���
	NVIC_InitStruct.NVIC_IRQChannelCmd=ENABLE;  //ʹ���ж�
	NVIC_InitStruct.NVIC_IRQChannelPreemptionPriority=2;  //��ռ���ȼ� 2
	NVIC_InitStruct.NVIC_IRQChannelSubPriority=2;     //�����ȼ�  2
	NVIC_Init(& NVIC_InitStruct);


}

void  EXTI15_10_IRQHandler(void)
{
	if(EXTI_GetITStatus(EXTI_Line15)!=RESET)//�����ź����룬������ʱ��
		{
		  INT_MARK++;
			if(INT_MARK==3)INT_MARK=0;
			EXTI_ClearITPendingBit(EXTI_Line15); 
	  }     
	   
	
}



