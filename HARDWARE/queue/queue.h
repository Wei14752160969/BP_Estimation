#ifndef __QUEUE_H
#define __QUEUE_H
#include "sys.h"
#include "math.h"
#include "stm32f10x.h"
#define ElementType uint16_t //�洢����Ԫ�ص�����
#define MAXSIZE 128 //�洢����Ԫ�ص�������
#define ERROR -2 //ElementType������ֵ����־����
 
typedef struct {
    ElementType data[MAXSIZE];
    uint16_t front; //��¼����ͷԪ��λ��
    uint16_t rear; //��¼����βԪ��λ��
    uint16_t size; //�洢����Ԫ�صĸ���
}Queue;

Queue* CreateQueue(void);
uint16_t IsFullQ(Queue* q);
void AddQ(Queue* q, ElementType item);//�������
uint16_t IsEmptyQ(Queue* q);
uint16_t DeleteQ(Queue* q);//�����Ƴ�
void PrintQueue(Queue* q); //��ӡ����
	


#endif
