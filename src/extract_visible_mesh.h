//
// Created by Yueping Wang on 5/9/19.
//

#ifndef SRC_EXTRACT_VISIBLE_MESH_H
#define SRC_EXTRACT_VISIBLE_MESH_H

// to create a list of weights that weight more for visible vertices during optimization
void get_weighting_list(
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& F,
        Eigen::MatrixXd& N,
        Eigen::VectorXd& V_w,
        const double view_dist
){
    using namespace std;
    using namespace Eigen;

    // Eliminate faces and vertices: pick vertices visible from viewpoint, conditioning on dot product (or just z>0 for now)
    // first get the normals and barycenters of each face
    igl::per_face_normals(V,F,N);
    MatrixXd Bc;
    igl::barycenter(V,F,Bc);
    // create a vector to record which vertices are used in the visible faces
    V_w.resize(V.rows());
    V_w.setZero();
    for (int i = 0; i<V.rows();i++){
        V_w(i)=0.1;
    }
    for (int i = 0; i<F.rows();i++){
        if(N(i,2)>0 || Bc(i,2)> -1*view_dist){
            Vector3i ftemp = F.row(i);
            V_w(ftemp(0))=1;
            V_w(ftemp(1))=1;
            V_w(ftemp(2))=1;
        }
    }

}

// this function is not used in the final code, because there are fragmented triangles and hard to compute the laplacian matrix
void extract_visible_mesh(
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& F,
        Eigen::MatrixXd& N,
        Eigen::MatrixXd& Vnew,
        Eigen::MatrixXi& Fnew,
        std::vector<int>& Fv,
        const double view_dist
){
    using namespace std;
    using namespace Eigen;

    // Eliminate faces and vertices: pick vertices visible from viewpoint, conditioning on dot product (or just z>0 for now)
    // first get the normals and barycenters of each face
    igl::per_face_normals(V,F,N);
    MatrixXd Bc;
    igl::barycenter(V,F,Bc);
    // create a vector to record which vertices are used in the visible faces
    VectorXi Vflags(V.rows());
    Vflags.setZero();
    for (int i = 0; i<F.rows();i++){
        if(N(i,2)>0 || Bc(i,2)> -1*view_dist){
            Fv.push_back(i);
            Vector3i ftemp = F.row(i);
            Vflags(ftemp(0))=1;
            Vflags(ftemp(1))=1;
            Vflags(ftemp(2))=1;
        }
    }
    // populate Vnew with only the visible vertices
    int Vsize = Vflags.sum();
    Vnew.conservativeResize(Vsize,3);
    int walk = 0;
    for (int i = 0; i<V.rows();i++){
        if(Vflags(i)>0){
            Vnew.row(walk)<< V(i,0), V(i,1), V(i,2);
            Vflags(i)=walk; // replace the "1" flag with the vertex index as seen in Vnew
            walk++;
        }
    }
    // populate Fnew with references to visible vertices in Vnew
    Fnew.conservativeResize(Fv.size(),3);
    Fnew.setZero();
    for (int i=0; i<Fv.size(); i++){
        int fid = Fv[i];
        Fnew.row(i)<< Vflags(F(fid,0)), Vflags(F(fid,1)), Vflags(F(fid,2));
    }
}

#endif //SRC_EXTRACT_VISIBLE_MESH_H
