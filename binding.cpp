#include "editdistance.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("editdistance", &editdistance, "editdistance forward");
}
