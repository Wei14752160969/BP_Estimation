#include "bsp_systick.h"

#if 0
static __INLINE uint32_t SysTick_Config(uint32_t ticks)
{ 
  if (ticks > SysTick_LOAD_RELOAD_Msk)  return (1);            /* Reload value impossible */
                                                               
  SysTick->LOAD  = (ticks & SysTick_LOAD_RELOAD_Msk) - 1;      /* set reload register */
  NVIC_SetPriority (SysTick_IRQn, (1<<__NVIC_PRIO_BITS) - 1);  /* set Priority for Cortex-M0 System Interrupts */
  SysTick->VAL   = 0;                                          /* Load the SysTick Counter Value */
  SysTick->CTRL  = SysTick_CTRL_CLKSOURCE_Msk | 
                   SysTick_CTRL_TICKINT_Msk   | 
                   SysTick_CTRL_ENABLE_Msk;                    /* Enable SysTick IRQ and SysTick Timer */
  return (0);                                                  /* Function successful */
}
#endif

void Delay_ms(uint32_t ms)
{
	uint32_t i;
	SysTick_Config(72000);									/*配置为1ms中断一次*/
	for(i=0;i<ms;i++)
	{
		while(!(SysTick->CTRL & (1<<16)));		/*检测标志位（最高位）*/
	}
	SysTick->CTRL&=~SysTick_CTRL_ENABLE_Msk;/*关闭SysTick*/
}
void Delay_us(uint32_t us)
{
	uint32_t i;
	SysTick_Config(72);
	for(i=0;i<us;i++)
	{
		while(!(SysTick->CTRL & (1<<16)));
	}
	SysTick->CTRL&=~SysTick_CTRL_ENABLE_Msk;
}

