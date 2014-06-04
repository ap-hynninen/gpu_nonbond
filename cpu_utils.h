
#ifndef CPU_UTILS_H
#define CPU_UTILS_H

template <class T>
void copy3D_HtoH(T* src_data, T* dst_data,
		 int src_x0, int src_y0, int src_z0,
		 size_t src_xsize, size_t src_ysize,
		 int dst_x0, int dst_y0, int dst_z0,
		 size_t width, size_t height, size_t depth,
		 size_t dst_xsize, size_t dst_ysize) {

  for (int z=0;z < depth;z++)
    for (int y=0;y < height;y++)
      for (int x=0;x < width;x++) {
	int src_pos = (src_x0+x) + ((src_y0+y) + (src_z0+z)*src_ysize)*src_xsize;
	int dst_pos = (dst_x0+x) + ((dst_y0+y) + (dst_z0+z)*dst_ysize)*dst_xsize;
	dst_data[dst_pos] = src_data[src_pos];
      }
  
}

//----------------------------------------------------------------------------------------

#endif // CPU_UTILS_H
