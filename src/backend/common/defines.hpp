/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <boost/functional/hash.hpp>
#include <common/util.hpp>

#include <unordered_map>
#include <utility>

namespace forge {
namespace common {

using CellIndex     = std::tuple<int, int, int>;
using MatrixHashMap = std::unordered_map<CellIndex, glm::mat4>;

constexpr int ARCBALL_CIRCLE_POINTS = 100;
constexpr float MOVE_SPEED          = 0.005f;
constexpr float ZOOM_SPEED          = 0.0075f;
constexpr float EPSILON             = 1.0e-6f;
constexpr float ARC_BALL_RADIUS     = 0.75f;
constexpr double PI                 = 3.14159265358979323846;
constexpr float BLACK[]             = {0.0f, 0.0f, 0.0f, 1.0f};
constexpr float GRAY[]              = {0.75f, 0.75f, 0.75f, 1.0f};
constexpr float WHITE[]             = {1.0f, 1.0f, 1.0f, 1.0f};
constexpr float AF_BLUE[]           = {0.0588f, 0.1137f, 0.2745f, 1.0f};
static const glm::mat4 IDENTITY(1.0f);

#if defined(OS_WIN)
#define __PRETTY_FUNCTION__ __FUNCSIG__
#define __FG_FILENAME__ (forge::common::clipPath(__FILE__, "src\\").c_str())
#else
#define __FG_FILENAME__ (forge::common::clipPath(__FILE__, "src/").c_str())
#endif

}  // namespace common
}  // namespace forge

namespace std {

template<>
struct hash<forge::common::CellIndex> {
    std::size_t operator()(const forge::common::CellIndex& key) const {
        size_t seed = 0;
        boost::hash_combine(seed, std::get<0>(key));
        boost::hash_combine(seed, std::get<1>(key));
        boost::hash_combine(seed, std::get<2>(key));
        return seed;
    }
};

}  // namespace std
