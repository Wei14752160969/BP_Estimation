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
#define accur 0.017295//accur=18*3.3/4096（3.3/4096就是ADC采样精度，18是为了让波形转化一下能够显示在适当位子）
extern uint16_t ConvData;//ADC采样数据
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
	OLED_Init();					 /* OLED初始化 */
	ADCx_Init();
 // USART2_Init(115200);
//	AdvancedTim_Init();
	Before_State_Update(accur*ConvData);
	OLED_CLS();
	OLED_Init2();//OLED初始化
	//OLED_Clear();//清屏     
  INT_MARK=0;	
	//NVIC_PriorityGroupConfig(NVIC_PriorityGroup_2);
	IWDG_Init(); //初始化并启动独立看门狗
	Queue* q	= CreateQueue();//建立队列
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
//						OLED_ShowCH(20,4,"蓝牙已连接");
//						break;
//				  }
				IWDG_Feed(); //喂狗
	}
		
		
		
		
		
		while(INT_MARK==1)
	{
			for(x=0;x<128;x=(x+1)%128)
			{
				if(accur*(ConvData-700)>63)h=63;
				if(accur*(ConvData-700)<0)h=0;
				else h= accur*(ConvData-700);
				OLED_DrawWave(x,h);//这是个画波形的函数
				delay_ms(25);
				//之前写了个画点函数，显示的波形不连续，然后我就改了一下画点函数，波形在屏幕上就连续了
				if(!INT_MARK)
					{ 
					  OLED_CLS();
						OLED_ShowCH(20,4,"蓝牙已连接");
						break;
				  }
				IWDG_Feed(); //喂狗
			}	 
		}	
	
	
	while(!INT_MARK)
	{
		//USART2_printf(" %1.3f \n", (float)(ConvData*3.3/4095));
	 if(GPIO_ReadInputDataBit(GPIOB,GPIO_Pin_9)&&m==0) 
		{ 
			  USART2_Init(115200);
		    m=1;
			 OLED_ShowCH(20,4,"蓝牙已连接");
		}
		
//			f=(ConvData+99)*3.3/4095;
//			if(f<0)f=0;
//			sprintf(buf, "voltage:%1.3fV",f); //构建显示数组
//			OLED_ShowCH(0,1,buf);
		//	delay_ms(100);	 
		
  
		if(GPIO_ReadInputDataBit(GPIOB,GPIO_Pin_9)&&m==1) 
		{

      USART2_printf(" %d\n", ConvData);//firewater数据类型，便于直接复制数据
//			  TEMP[0]=ConvData;
//			  JustFloat_Send(TEMP,1,USART2);//justfloat数据类型,传输速度更快
			  IWDG_Feed(); //喂狗
		}
		if(m==0)
		   OLED_ShowCH(20,4,"蓝牙搜索中");
//		else
//	     OLED_ShowCH(20,4,"蓝牙已连接");
	  IWDG_Feed(); //喂狗
	}
	}
}
