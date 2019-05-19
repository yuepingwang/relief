//
// Created by Yueping Wang on 5/11/19.
//

#ifndef SRC_MODIFIED_RAY_MESH_INTERSECT_H
#define SRC_MODIFIED_RAY_MESH_INTERSECT_H

#include <igl/Hit.h>
#include <igl/raytri.c>

bool modified_ray_mesh_intersect(
        const Eigen::Vector3d & s,
        const Eigen::Vector3d & dir,
        const Eigen::MatrixXd & V,
        const Eigen::MatrixXi & F,
        std::vector<igl::Hit> & hits,
        const int vid)
{
    using namespace Eigen;
    using namespace std;
    // Should be but can't be const
    Vector3d s_d = s;
    Vector3d dir_d = dir;
    hits.clear();
    // loop over all triangles
    for(int f = 0;f<F.rows();f++)
    {
        if(F(f,0)==vid || F(f,1)==vid || F(f,2)==vid)
            continue;
        // Should be but can't be ΩΩconst
        RowVector3d v0 = V.row(F(f,0)).template cast<double>();
        RowVector3d v1 = V.row(F(f,1)).template cast<double>();
        RowVector3d v2 = V.row(F(f,2)).template cast<double>();
        // shoot ray, record hit
        double t,u,v;
        if(intersect_triangle1(
                s_d.data(), dir_d.data(), v0.data(), v1.data(), v2.data(), &t, &u, &v) &&
           t>0)
        {
            hits.push_back({(int)f,(int)-1,(float)u,(float)v,(float)t});
        }
    }
    // Sort hits based on distance
    std::sort(
            hits.begin(),
            hits.end(),
            [](const igl::Hit & a, const igl::Hit & b)->bool{ return a.t < b.t;});
    return hits.size() > 0;
}
#endif //SRC_MODIFIED_RAY_MESH_INTERSECT_H
