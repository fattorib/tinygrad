typedef float float8 __attribute__((aligned(8),ext_vector_type(8)));
void E_8388608_8(unsigned char* restrict args_24) {
  unsigned long val0 = (*(((unsigned long*)(args_24))+0));
  unsigned long val1 = (*(((unsigned long*)(args_24))+1));
  unsigned long val2 = (*(((unsigned long*)(args_24))+2));
  for (int Lidx0 = 0; Lidx0 < 8388608; Lidx0++) {
    int alu0 = (Lidx0<<3);
    float8 val3 = (*((float8*)((((float*)(val1))+alu0))));
    float8 val4 = (*((float8*)((((float*)(val2))+alu0))));
    *((float8*)((((float*)(val0))+alu0))) = (float8){((2.0f*val3[0])+val4[0]),((2.0f*val3[1])+val4[1]),((2.0f*val3[2])+val4[2]),((2.0f*val3[3])+val4[3]), ((2.0f*val3[4])+val4[4]), ((2.0f*val3[5])+val4[5]), ((2.0f*val3[6])+val4[6]), ((2.0f*val3[7])+val4[7])};
  }
}