
#ifndef CPU_UTILS_H
#define CPU_UTILS_H

#ifdef _OPENMP
#include <omp.h>
#endif

template <class T>
void copy3D_HtoH(T* src_data, T* dst_data,
		 int src_x0, int src_y0, int src_z0,
		 size_t src_xsize, size_t src_ysize,
		 int dst_x0, int dst_y0, int dst_z0,
		 size_t width, size_t height, size_t depth,
		 size_t dst_xsize, size_t dst_ysize) {

  int src_pos, dst_pos;
  int x, y, z;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) private(x, y, z, src_pos, dst_pos)
#endif
  for (z=0;z < depth;z++) {
    for (y=0;y < height;y++) {
      src_pos = (src_x0) + ((src_y0+y) + (src_z0+z)*src_ysize)*src_xsize;
      dst_pos = (dst_x0) + ((dst_y0+y) + (dst_z0+z)*dst_ysize)*dst_xsize;
      for (x=0;x < width;x++) {
	dst_data[dst_pos + x] = src_data[src_pos + x];
      }
    }
  }
  
}

//----------------------------------------------------------------------------------------

#endif // CPU_UTILS_H
