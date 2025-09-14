#include <cuda_runtime.h>
#include "curand_kernel.h"
#include "selfPropelledVicsekAligningParticleDynamics.cuh"

/** \file selfPropelledVicsekAligningParticleDynamics.cu
    * Defines kernel callers and kernels for GPU calculations of simple active 2D cell models
*/

/*!
    \addtogroup simpleEquationOfMotionKernels
    @{
*/

/*!
Each thread calculates the velocity and displacement of an individual cell
*/
__global__ void spp_vicsek_aligning_compute_vel_disp_kernel(
                                           double2 *forces,
                                           double2 *velocities,
                                           double2 *displacements,
                                           double2 *motility,
                                           double *cellDirectors,
                                           int N,
                                           double deltaT,
                                           double mu)
    {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >=N)
        return;

    double currentTheta = cellDirectors[idx];
    double v0 = motility[idx].x;
    velocities[idx].x = v0*cos(currentTheta) + mu*forces[idx].x;
    velocities[idx].y = v0*sin(currentTheta) + mu*forces[idx].y;
    displacements[idx].x = deltaT * velocities[idx].x;
    displacements[idx].y = deltaT * velocities[idx].y;
    velocities[idx] = displacements[idx];
    };

/*!
Each thread updates the director of an individual cell
*/
__global__ void spp_vicsek_aligning_update_directors_kernel(
                                           double2 *velocities,
                                           double *cellDirectors,
                                           int *nNeighbors,
                                           int *neighbors,
                                           Index2D  n_idx,
                                           curandState *RNGs,
                                           int N,
                                           double Eta,
                                           double tau,
                                           double deltaT)
    {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >=N)
        return;

    //get an appropriate random angle displacement
    curandState_t randState;
    randState=RNGs[idx];
    double randomAngle = 2.0*PI*curand_uniform(&randState) - PI;
    RNGs[idx] = randState;

    double theta = atan2(velocities[idx].y, velocities[idx].x);
    double2 direction; direction.x = 0.0; direction.y=0.0;
    int neigh = nNeighbors[idx];
    for (int nn =0; nn < neigh; ++nn)
        {
        int neighbor = neighbors[n_idx(nn,idx)];
        double curTheta = atan2(velocities[neighbor].y, velocities[neighbor].x);
        direction.x += cos(curTheta);
        direction.y += sin(curTheta);
        }
    direction.x += neigh*Eta*cos(randomAngle);
    direction.y += neigh*Eta*sin(randomAngle);
    double phi = atan2(direction.y,direction.x);
    
    //update director
    cellDirectors[idx] = theta - (deltaT / tau) * sin(theta - phi);

    return;
    };

//!get the current timesteps vector of displacements into the displacement vector
bool gpu_spp_vicsek_aligning_eom_integration(
                    double2 *forces,
                    double2 *velocities,
                    double2 *displacements,
                    double2 *motility,
                    double *cellDirectors,
                    int *nNeighbors,
                    int *neighbors,
                    Index2D  &n_idx,
                    curandState *RNGs,
                    int N,
                    double deltaT,
                    int Timestep,
                    double mu,
                    double Eta,
                    double tau)
    {
    unsigned int block_size = 128;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;

    spp_vicsek_aligning_compute_vel_disp_kernel<<<nblocks,block_size>>>(
                                forces,velocities,displacements,motility,cellDirectors,
                                N,deltaT,mu);
    HANDLE_ERROR(cudaGetLastError());

    spp_vicsek_aligning_update_directors_kernel<<<nblocks,block_size>>>(
                                velocities,cellDirectors,
                                nNeighbors,neighbors,n_idx,
                                RNGs,
                                N,Eta,tau,deltaT);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

/** @} */ //end of group declaration
