#ifndef __QUEUE_H
#define __QUEUE_H
#include "sys.h"
#include "math.h"
#include "stm32f10x.h"
#define ElementType uint16_t //存储数据元素的类型
#define MAXSIZE 128 //存储数据元素的最大个数
#define ERROR -2 //ElementType的特殊值，标志错误
 
typedef struct {
    ElementType data[MAXSIZE];
    uint16_t front; //记录队列头元素位置
    uint16_t rear; //记录队列尾元素位置
    uint16_t size; //存储数据元素的个数
}Queue;

Queue* CreateQueue(void);
uint16_t IsFullQ(Queue* q);
void AddQ(Queue* q, ElementType item);//数据添加
uint16_t IsEmptyQ(Queue* q);
uint16_t DeleteQ(Queue* q);//数据移除
void PrintQueue(Queue* q); //打印数组
	


#endif
