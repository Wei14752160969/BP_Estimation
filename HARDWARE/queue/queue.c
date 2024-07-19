#include "queue.h"
#include "stm32f10x.h"
#include "OLED_I2C.h"
#include <stdlib.h>


Queue* CreateQueue() {
    Queue* q = (Queue*)malloc(sizeof(Queue));
    q->front = 0;
		q->rear = 127;
    q->size = 128;
    return q;
}
 
uint16_t IsFullQ(Queue* q) {
    return (q->size == MAXSIZE);//设置队列大小
}
 
void AddQ(Queue* q, ElementType item) //数据添加
	{
//    if (IsFullQ(q))return;
    q->rear++;
    q->rear %= MAXSIZE;//数据达到最大则回到第一个
    //q->size++;
    q->data[q->rear] = item;
 }
 
uint16_t IsEmptyQ(Queue* q) {
    return (q->size == 0);//清空数组
}
 
uint16_t DeleteQ(Queue* q)//数据移除
{
//    if (IsEmptyQ(q)) return 0;
    q->front++;
    q->front %= MAXSIZE; //0 1 2 3 4 5
    //q->size--;
    return q->data[q->front];
}
 
void PrintQueue(Queue* q) //打印数组
	{
//    if (IsEmptyQ(q)) return;
    uint16_t index = q->front;
    uint16_t i;
    for (i = 0; i < q->size; i++) 
		 {
        index++;
        index %= MAXSIZE;
			  OLED_DrawWave(i,q->data[index]);
     }
    
}
 
