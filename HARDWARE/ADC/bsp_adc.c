#include "bsp_adc.h"

uint16_t ConvData;

static void ADCx_GPIO_Config(void)
{
	GPIO_InitTypeDef GPIO_InitStruct;
	/*开时钟*/
	RCC_APB2PeriphClockCmd(ADCx_PORT_CLK, ENABLE);
	/*配置参数*/
	GPIO_InitStruct.GPIO_Pin=ADCx_PIN;
	GPIO_InitStruct.GPIO_Mode=GPIO_Mode_AIN;/*模拟输入模式*/
	/*写入寄存器*/
	GPIO_Init(ADCx_PORT, &GPIO_InitStruct);

}
static void ADCx_Config(void)
{
	ADC_InitTypeDef ADC_InitStruct;
	/*打开ADC时钟*/
	RCC_APB2PeriphClockCmd(ADCx_CLK, ENABLE);
	/*配置参数*/
	ADC_InitStruct.ADC_Mode=ADC_Mode_Independent;/*独立模式*/
	ADC_InitStruct.ADC_ContinuousConvMode=ENABLE;/*连续转换*/
	ADC_InitStruct.ADC_DataAlign=ADC_DataAlign_Right;/*数据右对齐*/
	ADC_InitStruct.ADC_ExternalTrigConv=ADC_ExternalTrigConv_None;/*不使用外部硬件触发*/
	ADC_InitStruct.ADC_NbrOfChannel=ADCx_CHx;/*ADC是哪个通道*/
	ADC_InitStruct.ADC_ScanConvMode=DISABLE;/*不使用连续扫描*/
	/*将参数写入寄存器*/
	ADC_Init(ADC_x, &ADC_InitStruct);
	/*设置ADC_CLK*/
	RCC_ADCCLKConfig(RCC_PCLK2_Div6);
	/*规则通道设置，通道，采样顺序，采样时间*/
	ADC_RegularChannelConfig(ADC_x, ADCx_CHx, 1, ADC_SampleTime_239Cycles5);
	/*ADC使能*/
	ADC_Cmd(ADC_x, ENABLE);
	/*校验ADC*/
	ADC_StartCalibration(ADC_x);
	while(ADC_GetCalibrationStatus(ADC_x)==RESET);
	/*DMA设置*/
	ADC_DMACmd(ADC_x, ENABLE);
	/*ADC软件触发，开始转换*/
	ADC_SoftwareStartConvCmd(ADC_x, ENABLE);
}
static void ADCx_DMA_Config(void)
{
	DMA_InitTypeDef DMA_InitStruct;
	/*打开DMA时钟*/
  RCC_AHBPeriphClockCmd(RCC_AHBPeriph_DMA1, ENABLE);
  /*配置DMA参数*/  
  DMA_InitStruct.DMA_PeripheralBaseAddr=(uint32_t)(&(ADC_x->DR));/*ADC的数据寄存器*/
  DMA_InitStruct.DMA_MemoryBaseAddr=(uint32_t)(&ConvData);/*存储器地址*/
  DMA_InitStruct.DMA_DIR=DMA_DIR_PeripheralSRC;/*外设为源*/
  DMA_InitStruct.DMA_BufferSize=1;/*数据的个数为1*/
  DMA_InitStruct.DMA_PeripheralInc=DMA_PeripheralInc_Disable;/*外设地址不增加*/
  DMA_InitStruct.DMA_MemoryInc=DMA_MemoryInc_Disable;/*存储器地址不增加*/
  DMA_InitStruct.DMA_PeripheralDataSize=DMA_PeripheralDataSize_HalfWord; /*16位数据，所以是半字*/
  DMA_InitStruct.DMA_MemoryDataSize=DMA_MemoryDataSize_HalfWord;
  DMA_InitStruct.DMA_Mode=DMA_Mode_Circular;/*这里是指数据循环不停地发送*/
  DMA_InitStruct.DMA_Priority=DMA_Priority_High;/*DMA优先级设为高*/
  DMA_InitStruct.DMA_M2M=DMA_M2M_Disable;/*不使能存储器到存储器*/
	/*将参数写入寄存器*/
  DMA_Init(ADCx_DMA_CHx, &DMA_InitStruct);
	/*使能DMA*/
  DMA_Cmd(ADCx_DMA_CHx, ENABLE);
}
void ADCx_Init(void)
{
	
	ADCx_GPIO_Config();
	ADCx_DMA_Config();
	ADCx_Config();
	
}


