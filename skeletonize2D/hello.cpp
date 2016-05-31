#include "hello.h"

#include <iostream>
#include <map>
#include <vector>
#include <cmath>

#include <boost/shared_ptr.hpp>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/create_straight_skeleton_2.h>

#include <CGAL/Polygon_2.h>
#include <CGAL/Polyline_simplification_2/simplify.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/algorithm.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Alpha_shape_2.h>

using Eigen::Matrix3f;
using Eigen::Matrix3d;

using namespace std;
namespace py = boost::python;
namespace np = boost::numpy;
namespace PS = CGAL::Polyline_simplification_2;

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point = K::Point_2;
using Polygon_2 = CGAL::Polygon_2<K>;
using Ss = CGAL::Straight_skeleton_2<K>;
using SsPtr = boost::shared_ptr<Ss>;

using Polygon_2 = CGAL::Polygon_2<K>;
using Stop = PS::Stop_below_count_ratio_threshold;
using Cost = PS::Squared_distance_cost;

using FT = K::FT;
using Segment = K::Segment_2;
using Vb = CGAL::Alpha_shape_vertex_base_2<K>;
using Fb = CGAL::Alpha_shape_face_base_2<K>;
using Tds = CGAL::Triangulation_data_structure_2<Vb, Fb>;
using Triangulation_2 = CGAL::Delaunay_triangulation_2<K, Tds>;
using Alpha_shape_2 = CGAL::Alpha_shape_2<Triangulation_2>;
using Alpha_shape_edges_iterator = Alpha_shape_2::Alpha_shape_edges_iterator;
using Alpha_shape_vertices_iterator = Alpha_shape_2::Alpha_shape_vertices_iterator;

template<class K> py::tuple get_skeleton(CGAL::Straight_skeleton_2<K> const& ss);

py::list get_concave_hull(Eigen::MatrixD2Int const& datasets)
{
	std::vector<Point> points;
	points.reserve(datasets.rows());

	for (size_t i = 0; i < datasets.rows(); i++)
	{
		points.push_back(Point(datasets(i, 0), datasets(i, 1)));
	}
	Alpha_shape_2 A(points.begin(), points.end(), FT(10000), Alpha_shape_2::REGULARIZED);
	//auto alpha = *A.find_optimal_alpha(10);
	A.set_alpha(500.0);
	//LOG(ERROR) << alpha;

	py::list out;

	for (auto itr = A.alpha_shape_edges_begin(); itr != A.alpha_shape_edges_end(); ++itr) {

		auto type = A.classify(*itr);
		auto segment = A.segment(*itr);

		Point p1 = segment.vertex(0);
		Point p2 = segment.vertex(1);
		out.append(py::make_tuple(py::make_tuple(p1.x(), p1.y()), py::make_tuple(p2.x(), p2.y())));
	}

	return out;
}

py::tuple get_skeleton_from_polygon(Eigen::MatrixD2Double const& polygon)
{
	Polygon_2 poly;

	for (int i = polygon.rows() - 1; i >= 0; i--) {
		poly.push_back(Point(polygon(i, 0), polygon(i, 1)));
	}
	Cost cost;
	poly = PS::simplify(poly, cost, Stop(0.5));

	SsPtr iss = CGAL::create_interior_straight_skeleton_2(poly);
	return get_skeleton(*iss);
}


template<class K> py::tuple get_skeleton(CGAL::Straight_skeleton_2<K> const& ss)
{
	typedef typename Ss::Vertex_const_handle     Vertex_const_handle;
	typedef typename Ss::Halfedge_const_handle   Halfedge_const_handle;
	typedef typename Ss::Halfedge_const_iterator Halfedge_const_iterator;

	Halfedge_const_handle null_halfedge;
	Vertex_const_handle   null_vertex;

	auto num_edges = ss.size_of_halfedges();
	auto num_vertices = ss.size_of_vertices();

	//Eigen::MatrixD2Int edges(num_edges, 3);
	py::list edges;
	Eigen::MatrixD2Double vertices(num_vertices, 2);
	std::map<int, int> mapping;

	int count = 0;
	for (auto i = ss.vertices_begin(); i != ss.vertices_end(); ++i)
	{
		auto id = i->id();
		auto pt = i->point();

		vertices(count, 0) = pt.x();
		vertices(count, 1) = pt.y();

		mapping[id] = count;

		count++;
	}

	for (auto i = ss.halfedges_begin(); i != ss.halfedges_end(); ++i)
	{
		auto opposite_vtx = i->opposite()->vertex();
		auto current_vtx = i->vertex();

		auto opposite_pos = mapping[opposite_vtx->id()];
		auto current_pos = mapping[current_vtx->id()];

		// && opposite_vtx->is_skeleton() && current_vtx->is_skeleton()
		if (i->is_inner_bisector() && opposite_vtx->is_skeleton() && current_vtx->is_skeleton())
		{
			edges.append(
				py::make_tuple(
					opposite_pos,
					current_pos,
					(vertices.row(current_pos) - vertices.row(opposite_pos)).squaredNorm()
					)
				);
		}
	}

	if (py::len(edges) == 0)
	{
		for (auto i = ss.halfedges_begin(); i != ss.halfedges_end(); ++i)
		{
			auto opposite_vtx = i->opposite()->vertex();
			auto current_vtx = i->vertex();

			auto opposite_pos = mapping[opposite_vtx->id()];
			auto current_pos = mapping[current_vtx->id()];

			edges.append(
				py::make_tuple(
					opposite_pos,
					current_pos,
					(vertices.row(current_pos) - vertices.row(opposite_pos)).squaredNorm()
					)
				);
		}
	}

	return py::make_tuple(edges, vertices);
}

py::list find_optimal_path(py::list const& path_list, Eigen::MatrixD2Double const& vtx_list)
{
	py::list* optimal_path = nullptr;
	double max_angle_sum = 0.0;
	for (int i = 0; i < py::len(path_list); i++)
	{
		auto& path = path_list[i];
		auto path_len = static_cast<int>(py::len(path));

		double angle_sum = 0.0;

		for (int j = 1; j < path_len-1; j++)
		{
			py::extract<int> current_pos(path[j]);
			py::extract<int> prev_pos(path[j-1]);
			py::extract<int> next_pos(path[j+1]);

			auto current_vtx = vtx_list.row(current_pos());
			auto prev_vtx = vtx_list.row(prev_pos());
			auto next_vtx = vtx_list.row(next_pos());

			auto v1 = prev_vtx - current_vtx;
			auto v2 = next_vtx - current_vtx;

			auto cross = fabs(v1.x()*v2.y() - v2.x()*v1.y());
			auto dot = v1.dot(v2);

			auto angle = atan2(cross, dot);
			angle_sum += fabs(angle);
		}
		angle_sum /= max(path_len - 2, 1);

		if (optimal_path == nullptr || max_angle_sum < angle_sum)
		{
			optimal_path = &static_cast<py::list&>(py::object(path));
			max_angle_sum = angle_sum;
		}
	}
	return *optimal_path;
}