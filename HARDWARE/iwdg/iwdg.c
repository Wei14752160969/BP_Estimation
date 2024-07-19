#include "iwdg.h"

void IWDG_Init(void){ //初始化独立看门狗
    IWDG_WriteAccessCmd(IWDG_WriteAccess_Enable); //使能对寄存器IWDG_PR和IWDG_RLR的写操作
    IWDG_SetPrescaler(pre); //设置IWDG预分频值
    IWDG_SetReload(rlr); //设置IWDG重装载值
    IWDG_ReloadCounter(); //按照IWDG重装载寄存器的值重装载IWDG计数器
    IWDG_Enable(); //使能IWDG
}

void IWDG_Feed(void){ //喂狗程序
    IWDG_ReloadCounter();//固件库的喂狗函数
}

