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
    return (q->size == MAXSIZE);//���ö��д�С
}
 
void AddQ(Queue* q, ElementType item) //�������
	{
//    if (IsFullQ(q))return;
    q->rear++;
    q->rear %= MAXSIZE;//���ݴﵽ�����ص���һ��
    //q->size++;
    q->data[q->rear] = item;
 }
 
uint16_t IsEmptyQ(Queue* q) {
    return (q->size == 0);//�������
}
 
uint16_t DeleteQ(Queue* q)//�����Ƴ�
{
//    if (IsEmptyQ(q)) return 0;
    q->front++;
    q->front %= MAXSIZE; //0 1 2 3 4 5
    //q->size--;
    return q->data[q->front];
}
 
void PrintQueue(Queue* q) //��ӡ����
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
 
