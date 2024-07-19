#include "vafa.h"
#include "stm32f10x.h"
const uint8_t tail [4] = {0x00, 0x00, 0x80, 0x7f};
uint8_t c_data[4];
void float_turn_u8(float f,uint8_t * c){
	uint8_t x;
	FloatLongType data;
	data.fdata=f;
	
	for(x=0;x<4;x++){
		c[x]=(uint8_t)(data.ldata>>(x*8));
	}
}
void JustFloat_Send(float * fdata,uint16_t fdata_num,USART_TypeDef *Usart_choose){
	uint16_t x;
	uint8_t y;
		for(x=0;x<fdata_num;x++){
			float_turn_u8(fdata[x],c_data);
			for(y=0;y<4;y++){
				Usart_choose->DR=c_data[y];
				while((Usart_choose->SR&0X40)==0);
			}
		}
		for(y=0;y<4;y++){
				Usart_choose->DR=tail[y];
			while((Usart_choose->SR&0X40)==0);
		}
	
}
