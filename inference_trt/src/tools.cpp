#include "tools.h"
void np_to_vec(const string& filename,const int N,float *data){
    ifstream in(filename,ios::in|ios::binary);
    in.read(reinterpret_cast<char*>(data),sizeof(float)*N);
}
