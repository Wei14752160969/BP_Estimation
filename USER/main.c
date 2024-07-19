#include "sys.h"
#include "delay.h"
#include "bsp_adc.h"
#include "bsp_i2c_gpio.h"
#include "OLED_I2C.h"
#include "bsp_systick.h"
#include "bsp_led.h"
#include <math.h>
#include "OLED_I2C2.h"
#include "usart.h"
#include "iwdg.h"
#include "vafa.h"
#include "queue.h"
uint8_t buf[20];
uint8_t m=0,h;
float f;
float TEMP[1];
#define pi 3.1415926535
#define accur 0.017295//accur=18*3.3/4096��3.3/4096����ADC�������ȣ�18��Ϊ���ò���ת��һ���ܹ���ʾ���ʵ�λ�ӣ�
extern uint16_t ConvData;//ADC��������
extern unsigned char BMP1[];
int main()
{
	uint8_t x,f;
	//uint8_t a,b=0;
	LED_Init();
	KEY_Init();
	INT_INIT ();
  delay_init();
	delay_ms(500);
	OLED_Init();					 /* OLED��ʼ�� */
	ADCx_Init();
 // USART2_Init(115200);
//	AdvancedTim_Init();
	Before_State_Update(accur*ConvData);
	OLED_CLS();
	OLED_Init2();//OLED��ʼ��
	//OLED_Clear();//����     
  INT_MARK=0;	
	//NVIC_PriorityGroupConfig(NVIC_PriorityGroup_2);
	IWDG_Init(); //��ʼ���������������Ź�
	Queue* q	= CreateQueue();//��������
	while(1)
	{
		
		while(INT_MARK==2)
	{
		
//		if(b==0)
//		{
//				for(a=0;a<128;a++)
//			{
//				AddQ(q, (accur*ConvData));
//				//PrintQueue(q);
//				
//			}
//			b=1;
//	  } 	
		  AddQ(q, (accur*(ConvData-700)));
   		DeleteQ(q);
		  delay_ms(30);
		  f++;
		if(f==50)
		{
   	  PrintQueue(q);
			f=0;
		}
		
//				if(!INT_MARK)
//					{ 
//					  OLED_CLS();
//						OLED_ShowCH(20,4,"����������");
//						break;
//				  }
				IWDG_Feed(); //ι��
	}
		
		
		
		
		
		while(INT_MARK==1)
	{
			for(x=0;x<128;x=(x+1)%128)
			{
				if(accur*(ConvData-700)>63)h=63;
				if(accur*(ConvData-700)<0)h=0;
				else h= accur*(ConvData-700);
				OLED_DrawWave(x,h);//���Ǹ������εĺ���
				delay_ms(25);
				//֮ǰд�˸����㺯������ʾ�Ĳ��β�������Ȼ���Ҿ͸���һ�»��㺯������������Ļ�Ͼ�������
				if(!INT_MARK)
					{ 
					  OLED_CLS();
						OLED_ShowCH(20,4,"����������");
						break;
				  }
				IWDG_Feed(); //ι��
			}	 
		}	
	
	
	while(!INT_MARK)
	{
		//USART2_printf(" %1.3f \n", (float)(ConvData*3.3/4095));
	 if(GPIO_ReadInputDataBit(GPIOB,GPIO_Pin_9)&&m==0) 
		{ 
			  USART2_Init(115200);
		    m=1;
			 OLED_ShowCH(20,4,"����������");
		}
		
//			f=(ConvData+99)*3.3/4095;
//			if(f<0)f=0;
//			sprintf(buf, "voltage:%1.3fV",f); //������ʾ����
//			OLED_ShowCH(0,1,buf);
		//	delay_ms(100);	 
		
  
		if(GPIO_ReadInputDataBit(GPIOB,GPIO_Pin_9)&&m==1) 
		{

      USART2_printf(" %d\n", ConvData);//firewater�������ͣ�����ֱ�Ӹ�������
//			  TEMP[0]=ConvData;
//			  JustFloat_Send(TEMP,1,USART2);//justfloat��������,�����ٶȸ���
			  IWDG_Feed(); //ι��
		}
		if(m==0)
		   OLED_ShowCH(20,4,"����������");
//		else
//	     OLED_ShowCH(20,4,"����������");
	  IWDG_Feed(); //ι��
	}
	}
}
