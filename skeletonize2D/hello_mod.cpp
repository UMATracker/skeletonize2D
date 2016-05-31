#include "eigen_numpy.h"

#include "hello.h"

namespace py = boost::python;
namespace np = boost::numpy;

BOOST_PYTHON_MODULE(__MACRO_PROJECT_NAME) {
  np::initialize();
  SetupEigenConverters();

  py::def("get_skeleton_from_polygon", get_skeleton_from_polygon);
  py::def("get_concave_hull", get_concave_hull);
  py::def("find_optimal_path", find_optimal_path);
}
