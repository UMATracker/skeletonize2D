#pragma once

#include <Eigen/Dense>
#include <utility>

#include "eigen_numpy.h"


boost::python::tuple get_skeleton_from_polygon(Eigen::MatrixD2Double const& polygon);
boost::python::list get_concave_hull(Eigen::MatrixD2Int const& datasets);
boost::python::list find_optimal_path(boost::python::list const& path_list, Eigen::MatrixD2Double const& vtx_list);