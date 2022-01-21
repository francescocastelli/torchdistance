#include "editdistance.h"

TORCH_LIBRARY(editdistance, m) {
  m.def("editdistance(Tensor self, Tensor other) -> Tensor");
}
