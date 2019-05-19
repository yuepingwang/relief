#include <math.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Sparse>
//#include <Eigen/KroneckerProduct>

#include <igl/readOFF.h>
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/adjacency_list.h>
#include <igl/per_face_normals.h>
#include <igl/barycenter.h>
//#include <igl/doublearea.h>
//#include <igl/vertex_triangle_adjacency.h>
#include <igl/adjacency_list.h>
#include <igl/massmatrix.h>
#include <igl/cotmatrix.h>
#include <igl/diag.h>
#include <igl/ray_mesh_intersect.h>
#include <igl/Hit.h>
#include <igl/raytri.c>
#include <igl/copyleft/quadprog.h>

#include "extract_visible_mesh.h"
#include "modified_ray_mesh_intersect.h"

// Mesh (Original Model)
Eigen::MatrixXd V;// all vertices
Eigen::MatrixXi F;// all faces
Eigen::MatrixXd N;// all per-vertex normals
//Eigen::MatrixXd Bc;// all face's barycenters
Eigen::MatrixXd Vnew;// all vertices in optimization
Eigen::MatrixXi Fnew;// all faces in optimization
//std::vector<int> Fv;// indices of visible faces in (old) F

Eigen::MatrixXd Vunit;// unit vectors pointing from origin to Vnew
Eigen::VectorXd Lambda0;// lambda(scaler) terms for unit vectors of original mesh
Eigen::VectorXd max_s,min_s;// inequality constraints if using non-flat surface

// View point
Eigen::Vector3d O;// view point(0,0,0)
double view_dist;// distance from view point to center of the (original) mesh
double maxl, minl, max_thickness;//max and minimun distance from the view point(0,0,0)
double radius;// for implicit surface to project on
Eigen::Vector3d Os;// origin for defining implicit surface
int base_type;// 0 for flat surface, 1 for implicit sphere

int main(int argc, char *argv[])
{
    using namespace std;
    using namespace Eigen;

    //First import OBJ
    igl::readOBJ("../data/bunny.obj",V,F);

    base_type = 1;// 0 for projection on plane; 1 for projecting on implicit sphere
    radius = 0.36;

    // set view point at (0,0,0)
    O << 0, 0, 0;
    max_thickness = 0.032;
    maxl = 1;
    minl = maxl - max_thickness;

    // for scaling large scale meshes. This should be calculated more flexibly...
//    for(int i=0; i<V.rows(); i++) {
//        V.row(i) << V(i, 0) / 2, V(i, 1) / 2, V(i, 2) / 2;
//    }

    // Center the vertices around (0,0, view_dist)
    view_dist = 1;
    Vector3d center;
    center.setZero();
    for(int i=0; i<V.rows(); i++){
        center += V.row(i);
    }
    center /=V.rows();
    center(2) += view_dist;
    V.rowwise() -= center.transpose();

    Os << 0, 0.18, -center(2)-radius;// center for the implicit surface

    // Take all vertices and faces into account, but weight them differently during optimization
    Vnew = V;
    Fnew=F;

    // Create the weight matrix by giving more weight to vertices visible from view point:
    VectorXd Vw;
    get_weighting_list(V,F,N,Vw,view_dist);
    SparseMatrix<double> Dw;
    Dw.resize(Vnew.rows()*3,Vnew.rows()*3);

    vector< Triplet<double> > dw_rows;// rows for large matric of Da
    for(int i =0; i<Vnew.rows(); i++){
        dw_rows.push_back(Triplet<double>(i*3, i*3, Vw(i)));
        dw_rows.push_back(Triplet<double>(i*3+1, i*3+1, Vw(i)));
        dw_rows.push_back(Triplet<double>(i*3+2, i*3+2, Vw(i)));
    }
    Dw.setFromTriplets(dw_rows.begin(),dw_rows.end());

    // Calculate the unit vectors from (0,0,0) to visible vertices, and the lambda(scaler) terms for original mesh
    Vunit.conservativeResize(Vnew.rows(),3);
    Lambda0.conservativeResize(Vnew.rows());
    for (int i = 0; i<Vnew.rows(); i++){
        Lambda0(i) = Vnew.row(i).norm();
        Vunit.row(i) = Vnew.row(i) / Lambda0(i);
    }

    // Calculate the voronoi areas A
    vector<vector<int>> A_list;
    igl::adjacency_list(Fnew, A_list,true); //get ordered adjacency list

    SparseMatrix<double> Da;
    igl::massmatrix(Vnew,Fnew,igl::MASSMATRIX_TYPE_VORONOI,Da);
    SparseMatrix<double> Da_l(Vnew.rows()*3,Vnew.rows()*3);
    vector< Triplet<double> > dal_rows;// rows for large matric of Da
    for(int i =0; i<Vnew.rows(); i++){
        dal_rows.push_back(Triplet<double>(i*3, i*3, Da.coeff(i, i)));
        dal_rows.push_back(Triplet<double>(i*3+1, i*3+1, Da.coeff(i, i)));
        dal_rows.push_back(Triplet<double>(i*3+2, i*3+2, Da.coeff(i, i)));
    }
    Da_l.setFromTriplets(dal_rows.begin(),dal_rows.end());
    // calculate the laplacian terms (as diagonal sparse matrix L_0*D_V )
    SparseMatrix<double> L, LV;//the laplacian operation on Vnew( Vnew is the visible part of the original mesh)
    igl::cotmatrix(Vnew, Fnew, L);
    LV.resize(Vnew.rows()*3, Vnew.rows()*3);

    // get pronecker product
    vector< Triplet<double> > lv_rows;// rows for Aff
    for (int i = 0; i<Vnew.rows(); i++) {
        for (int j=0; j<Vnew.rows(); j++){
            double l = L.coeff(i,j);
            lv_rows.push_back(Triplet<double>(i*3, j*3, l));
            lv_rows.push_back(Triplet<double>(i*3+1, j*3+1, l));
            lv_rows.push_back(Triplet<double>(i*3+2, j*3+2, l));
        }
    }
    LV.setFromTriplets(lv_rows.begin(),lv_rows.end());

    // calculate the terms in L_theta:
    // S is a selector matrix used for both Lambda0 and the objective Lambda(for the relief)
    SparseMatrix<double> S(Vnew.rows()*3,Vnew.rows());
    vector< Triplet<double> > s_rows;
    // S*Lambda0 is basically given by populating Lambda0 values to a #V*3 by 1 vector
    VectorXd SLambda0(Vnew.rows()*3);
    // D_SLambda0 is the diagonal matrix of SLambda0, used for getting the inverse
    SparseMatrix<double> Dsl(Vnew.rows()*3, Vnew.rows()*3);
    vector< Triplet<double> > d_rows;
    for (int i = 0; i<Vnew.rows(); i++){
        // for S
        s_rows.push_back(Triplet<double>(i*3,i, 1));
        s_rows.push_back(Triplet<double>(i*3+1,i,1));
        s_rows.push_back(Triplet<double>(i*3+2,i,1));
        // for D_SLambda0
        d_rows.push_back(Triplet<double>(i*3,i*3,Lambda0(i)));
        d_rows.push_back(Triplet<double>(i*3+1,i*3+1,Lambda0(i)));
        d_rows.push_back(Triplet<double>(i*3+2,i*3+2,Lambda0(i)));
        // for SLambda0
        SLambda0(i*3) = Lambda0(i);
        SLambda0(i*3+1) = Lambda0(i);
        SLambda0(i*3+2) = Lambda0(i);
    }
    S.setFromTriplets(s_rows.begin(),s_rows.end());
    Dsl.setFromTriplets(d_rows.begin(),d_rows.end());

    // calculate the inverse of the diagonal matrix of SLambda0
    SparseLU< SparseMatrix<double> > solver;
    solver.compute(Dsl);
    SparseMatrix<double> I(Vnew.rows()*3, Vnew.rows()*3);
    I.setIdentity();
    auto Dsl_inv = solver.solve(I);

    // calculate the value of L_theta by multiplying the above terms
    SparseMatrix<double> L_theta_sparse(Vnew.rows()*3,1);
    L_theta_sparse = Dsl_inv * (LV * SLambda0);
    VectorXd L_theta(Vnew.rows()*3,1);
    L_theta = VectorXd(L_theta_sparse);

    // calculate the difference between LV and digonalized L_theta
    SparseMatrix<double> Dlt(Vnew.rows()*3, Vnew.rows()*3);
    igl::diag(L_theta, Dlt);

    // Diagonalized per-vertex areas "Da" was already calculated, and we ignore the Dw term in original paper, since we've already eliminated invisible vertices)
    // objective: ||Da * (LV-Dlt) * S * Lambda||^2, solve for Lambda
    SparseMatrix<double> D_diff(Vnew.rows()*3,Vnew.rows()*3);
    D_diff = LV-Dlt;
    SparseMatrix<double> D_temp_sparse(Vnew.rows()*3, Vnew.rows());
    SparseMatrix<double> D_temp_sparse1(Vnew.rows()*3, Vnew.rows());
    D_temp_sparse = D_diff * S;
    D_temp_sparse1 = Da_l*Dw*D_temp_sparse;

    MatrixXd D_temp(Vnew.rows()*3, Vnew.rows());//temporary results of the sqaure matrices
    D_temp = MatrixXd(D_temp_sparse1);
    MatrixXd G(Vnew.rows(), Vnew.rows());
    G = D_temp.transpose()*D_temp;
    VectorXd g0(Vnew.rows());
    g0.setZero();

    // Check for potential overlaps in the visible mesh, for each vertex-face overlap, create an inequality constraint for depth order
    // First, create a spatial data structure to store the faces
    // and sort faces by x and y positions of their barycenters
    MatrixXd Bc;
    igl::barycenter(Vnew,Fnew,Bc);
    int res = 10; //split the faces to 10 x 10 grids
    vector<Eigen::MatrixXi * > gridF(res*res);// a vector of matrices
    for(int i =0; i<res*res; i++){
        MatrixXi t;
        gridF[i]= &t;
    }
    VectorXd minCoord, maxCoord, span;
    minCoord = Bc.colwise().minCoeff();
    maxCoord = Bc.colwise().maxCoeff();
    span = maxCoord - minCoord;
    for(int i=0;i<Bc.rows();i++){
        auto diff = maxCoord - Bc.row(i).transpose();
        double res_less = res - 0.1;// to avoid geting the same min/max position and getting a index=10
        int xid = (int)(res_less*diff(0)/span(0));
        int yid = (int)(res_less*diff(1)/span(1));
        int id = xid*res+yid;
        gridF[id]->conservativeResize((gridF[id]->rows())+1,3);
        gridF[id]->row((gridF[id]->rows())-1)=Fnew.row(i);
    }
    // Then, create a spacial data structure for all visible vertices
    vector<int> gridV[res*res];// an array of vectors
    minCoord = Vnew.colwise().minCoeff();
    maxCoord = Vnew.colwise().maxCoeff();
    span = maxCoord - minCoord;
    for(int i=0;i<Vnew.rows();i++){
        auto diff = maxCoord - Vnew.row(i).transpose();
        double res_less = res - 0.1;// to avoid getting the same min/max position and resulting in an index=10
        int xid = (int)(res_less*diff(0)/span(0));
        int yid = (int)(res_less*diff(1)/span(1));
        int id = xid*res+yid;
        gridV[id].push_back(i);
    }
    // initialize the matrix for inequality constraints for Lambda
    SparseMatrix<double> CI_sparse;
    vector< Triplet<double> > ci_rows;


    for(int i=0;i<Vnew.rows();i++){
        // for first #V rows are diagonal matrix of -1, so that -1*lambda > -max
        ci_rows.push_back(Triplet<double>(i,i,-1));
        // for next #V rows constitute the identity, so that 1*lambda > min
        ci_rows.push_back(Triplet<double>(i+Vnew.rows(),i,1));
    }
    // Projection on flat surface:
    // simply set uniform min and max values for lambdas (code for ci0)

    // Projection on implicit sphere:
    if(base_type==1){
        max_s.resize(Vnew.rows());
        min_s.resize(Vnew.rows());
        Vector3d Os_displacement = O-Os;
        maxl += radius;
        minl += radius;
        double radius1 = radius + max_thickness;
        for(int i=0;i<Vunit.rows();i++){
            // first check ray intersection with inner sphere
            double a = Vunit.row(i).dot(Vunit.row(i));
            double b = 2* (Vunit.row(i).dot(Os_displacement));
            double c = Os_displacement.dot(Os_displacement) - radius * radius;
            double d = b * b - 4 * a * c;
            if(d<0){
                min_s(i)=minl;
                max_s(i)=maxl;
                continue;
            }
            // if has intersection, then get the scale value of this unit vector, which will be lambda(i)'s max value
            double root = sqrt(d);
            double ray0 = (-b+root)/(2*a);
            double ray1 = (-b-root)/(2*a);
            double ray = min(ray0, ray1);// by definition ray must be greater than 0
            max_s(i)=abs(ray);
            cout<<i<<" max ray: "<<ray<<endl;
            // also check ray intersection with outer sphere
            double c1 = Os_displacement.dot(Os_displacement) - radius1 * radius1;
            double d1 = b * b - 4 * a * c1;
            root = sqrt(d1);
            ray0 = (-b+root)/(2*a);
            ray1 = (-b-root)/(2*a);
            ray = min(ray0, ray1);// by definition ray must be greater than 0
            min_s(i)=abs(ray);
            cout<<i<<" min ray: "<<ray<<endl;
        }
    }

    // the last #(?) rows specifies the depth ordering for those ray directions that have intersections
    // check ray-mesh intersection for each grid and its neighbors
    int d_count =2*Vnew.rows();// starting row index for next depth order constraints
    for(int i=0;i<res;i++){
        for(int j=0; j<res; j++){
            int id = i*res+j;
            MatrixXi f = *gridF[id];
            // for each vertex inthis grid, cast a ray to see if it intersect with another face
            for(int k =0;k<gridV[id].size();k++){
                vector<igl::Hit> hits;
                int vid = gridV[id][k];// id of the vertex, through which we're currently doing ray tracing
                Vector3d v = Vunit.row(vid);
                modified_ray_mesh_intersect(O, v, Vnew, f, hits, vid);
                if(hits.size()>0){
                    // for each intersection, get the u, v, t, f_id of the face,
                    for(int m=0; m<hits.size();m++){
                        // get the ids of the vertices a,b,c
                        Vector3i ids= f.row(hits[m].id);
                        double aCoeff, bCoeff, cCoeff;
                        aCoeff = -1.*(hits[m].u + hits[m].v) * Vunit(ids(0),2);
                        bCoeff = hits[m].u * Vunit(ids(1),2);
                        cCoeff = hits[m].v * Vunit(ids(2),2);
                        //if t > Lambda0(vid), {the above value} - Lambda(vid) * V(vid) > 0
                        if(hits[m].t<Lambda0(vid)){
                            ci_rows.push_back(Triplet<double>(d_count,ids(0),aCoeff));
                            ci_rows.push_back(Triplet<double>(d_count,ids(1),bCoeff));
                            ci_rows.push_back(Triplet<double>(d_count,ids(2),cCoeff));
                            ci_rows.push_back(Triplet<double>(d_count,vid, -1.*Vunit(vid,2)));
                            d_count++;
                        }
                    }
                }
            }
        }
    }
    cout<<"hit number: "<<d_count-2*Vnew.rows()<<" total v: "<<Vnew.rows()<<endl;
    CI_sparse.resize(d_count,Vnew.rows());
    CI_sparse.setFromTriplets(ci_rows.begin(),ci_rows.end());
    MatrixXd CI(Vnew.rows(),d_count);
    CI = MatrixXd(CI_sparse.transpose());
    VectorXd ci0(d_count);
    ci0.setZero();
    if(base_type==0){
        for(int i =0; i<Vnew.rows();i++){
            ci0(i)= maxl;
            ci0(i+Vnew.rows())= -minl;
        }
    }
    else if(base_type==1){
        for(int i =0; i<Vnew.rows();i++){
            ci0(i)= max_s(i);
            ci0(i+Vnew.rows())= -min_s(i);
        }
    }

    // for the equality constraint, find the vertex of the mesh that's closest to (0,0,view_dist)
    double min_dist = 2;
    double max_z = -view_dist;
    int fix_vid;
    Vector3d V_center = center;
    V_center(2) -= 2*view_dist;
    for(int i=0; i<Vnew.rows();i++){
        auto dist = (Vector3d(Vnew.row(i))-V_center).norm();
        if (dist<min_dist){
            min_dist=dist;
            fix_vid = i;
        }
    }
    MatrixXd CE(Vnew.rows(),1);
    CE.setZero();
    CE(fix_vid,0)=1;

    //set Lambda(fix_vid) to (min+max)j/2
    double fix_lambda;
    if(base_type==0)
        fix_lambda=(maxl+minl)/2;
    else if (base_type==1)
        fix_lambda=(max_s(fix_vid)+min_s(fix_vid))/2;
    VectorXd ce0(1);
    ce0(0)= -fix_lambda;

    VectorXd x;
    bool result=false;
    result=igl::copyleft::quadprog(G, g0, CE, ce0, CI, ci0, x);
    if(result){
        cout<<"Solved."<<endl;
    }
    // construct new model using the lamda values
    MatrixXd V_relief(Vnew.rows(),3);
    for (int i = 0; i<Vnew.rows(); i++){
        V_relief.row(i)=Vunit.row(i) * x(i);
        V_relief(i,2) += view_dist;
    }
    igl::writeOBJ("../output_obj/bunny4.obj",V_relief,Fnew);
    // Plot the mesh
    igl::opengl::glfw::Viewer viewer;
    viewer.data().set_mesh(V_relief, Fnew);
    viewer.data().set_face_based(true);
    viewer.launch();
}
