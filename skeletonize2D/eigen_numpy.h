#pragma once

#include <cstdint>
#include <Eigen/Eigen>

#define BOOST_PYTHON_STATIC_LIB
#include <Python.h>
#include <boost/python.hpp>
#include <boost/numpy.hpp>

#define GOOGLE_GLOG_DLL_DECL
#include <glog/logging.h>

#include <numpy/arrayobject.h>

namespace Eigen {
	// For int
	using MatrixLongLong = Matrix<long long, Dynamic, Dynamic>;
	using MatrixLong = Matrix<long, Dynamic, Dynamic>;
	using MatrixInt = Matrix<int, Dynamic, Dynamic>;
	using MatrixShort = Matrix<short, Dynamic, Dynamic>;
	using MatrixByte = Matrix<int8_t, Dynamic, Dynamic>;

	using MatrixULongLong = Matrix<unsigned long long, Dynamic, Dynamic>;
	using MatrixULong = Matrix<unsigned long, Dynamic, Dynamic>;
	using MatrixUInt = Matrix<unsigned int, Dynamic, Dynamic>;
	using MatrixUShort = Matrix<unsigned short, Dynamic, Dynamic>;
	using MatrixUByte = Matrix<uint8_t, Dynamic, Dynamic>;

	using MatrixD2LongLong = Matrix<long long, Dynamic, 2>;
	using MatrixD2Long = Matrix<long, Dynamic, 2>;
	using MatrixD2Int = Matrix<int, Dynamic, 2>;
	using MatrixD2Short = Matrix<short, Dynamic, 2>;
	using MatrixD2Byte = Matrix<int8_t, Dynamic, 2>;

	using MatrixD2ULongLong = Matrix<unsigned long long, Dynamic, 2>;
	using MatrixD2ULong = Matrix<unsigned long, Dynamic, 2>;
	using MatrixD2UInt = Matrix<unsigned int, Dynamic, 2>;
	using MatrixD2UShort = Matrix<unsigned short, Dynamic, 2>;
	using MatrixD2UByte = Matrix<uint8_t, Dynamic, 2>;

	using MatrixD3LongLong = Matrix<long long, Dynamic, 3>;
	using MatrixD3Long = Matrix<long, Dynamic, 3>;
	using MatrixD3Int = Matrix<int, Dynamic, 3>;
	using MatrixD3Short = Matrix<short, Dynamic, 3>;
	using MatrixD3Byte = Matrix<int8_t, Dynamic, 3>;

	using MatrixD3ULongLong = Matrix<unsigned long long, Dynamic, 3>;
	using MatrixD3ULong = Matrix<unsigned long, Dynamic, 3>;
	using MatrixD3UInt = Matrix<unsigned int, Dynamic, 3>;
	using MatrixD3UShort = Matrix<unsigned short, Dynamic, 3>;
	using MatrixD3UByte = Matrix<uint8_t, Dynamic, 3>;

	using VectorLongLong = Matrix<long long, 1, Dynamic>;
	using VectorLong = Matrix<long, 1, Dynamic>;
	using VectorInt = Matrix<int, 1, Dynamic>;
	using VectorShort = Matrix<short, 1, Dynamic>;
	using VectorByte = Matrix<int8_t, 1, Dynamic>;

	using VectorULongLong = Matrix<unsigned long long, 1, Dynamic>;
	using VectorULong = Matrix<unsigned long, 1, Dynamic>;
	using VectorUInt = Matrix<unsigned int, 1, Dynamic>;
	using VectorUShort = Matrix<unsigned short, 1, Dynamic>;
	using VectorUByte = Matrix<uint8_t, 1, Dynamic>;

	using Vector2LongLong = Matrix<long long, 1, 2>;
	using Vector2Long = Matrix<long, 1, 2>;
	using Vector2Int = Matrix<int, 1, 2>;
	using Vector2Short = Matrix<short, 1, 2>;
	using Vector2Byte = Matrix<int8_t, 1, 2>;

	using Vector2ULongLong = Matrix<unsigned long long, 1, 2>;
	using Vector2ULong = Matrix<unsigned long, 1, 2>;
	using Vector2UInt = Matrix<unsigned int, 1, 2>;
	using Vector2UShort = Matrix<unsigned short, 1, 2>;
	using Vector2UByte = Matrix<uint8_t, 1, 2>;

	using Vector3LongLong = Matrix<long long, 1, 3>;
	using Vector3Long = Matrix<long, 1, 3>;
	using Vector3Int = Matrix<int, 1, 3>;
	using Vector3Short = Matrix<short, 1, 3>;
	using Vector3Byte = Matrix<int8_t, 1, 3>;

	using Vector3ULongLong = Matrix<unsigned long long, 1, 3>;
	using Vector3ULong = Matrix<unsigned long, 1, 3>;
	using Vector3UInt = Matrix<unsigned int, 1, 3>;
	using Vector3UShort = Matrix<unsigned short, 1, 3>;
	using Vector3UByte = Matrix<uint8_t, 1, 3>;

	// for FP
	using MatrixFloat = Matrix<float, Dynamic, Dynamic>;
	using MatrixDouble = Matrix<double, Dynamic, Dynamic>;

	using MatrixD2Float = Matrix<float, Dynamic, 2>;
	using MatrixD2Double = Matrix<double, Dynamic, 2>;

	using MatrixD3Float = Matrix<float, Dynamic, 3>;
	using MatrixD3Double = Matrix<double, Dynamic, 3>;

	// for Bool
	using MatrixBool = Matrix<bool, Dynamic, Dynamic>;
	using MatrixD2Bool = Matrix<bool, Dynamic, 2>;
	using MatrixD3Bool = Matrix<bool, Dynamic, 3>;

	using VectorBool = Matrix<bool, 1, Dynamic>;
	using Vector2Bool = Matrix<bool, 1, 2>;
	using Vector3Bool = Matrix<bool, 1, 3>;
}

int SetupEigenConverters();
