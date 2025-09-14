#include <cuda_runtime.h>
#include "curand_kernel.h"
#include "vertexQuadraticEnergyWithNematic.cuh"

/*! \file vertexQuadraticEnergyWithNematic.cu
    Defines kernel callers and kernels for GPU calculations of vertex model with nematic forces
*/

/*!
    \addtogroup vmKernels
    @{
*/

__global__ void vm_compute_q_tensors_kernel(
    double2 *voroCur,
    int *cellVertexNum,
    double2 *areaPeri,
    double2 *cellQ,
    Index2D n_idx,
    int Ncells
    )
{
    unsigned int ii = blockDim.x * blockIdx.x + threadIdx.x;
    if (ii >= Ncells)
        return;
    // int idx = tagToIdx[ii];
    // this is fucked don't use this
    int idx = ii;
    int nVerts = cellVertexNum[idx];
    double perimeter = areaPeri[idx].y;
    double Qxx = 0.0, Qxy = 0.0;
    const double EPS = 1e-10;

    for (int nn = 0; nn < nVerts; ++nn)
    {
        double2 vCur = voroCur[n_idx(nn, idx)];
        double2 vNext = voroCur[n_idx((nn + 1) % nVerts, idx)];
        double2 edge = make_double2(vNext.x - vCur.x, vNext.y - vCur.y);
        double edgeLen = sqrt(edge.x * edge.x + edge.y * edge.y);
        if (edgeLen > EPS)
        {
            double2 tHat = make_double2(edge.x / edgeLen, edge.y / edgeLen);
            Qxx += edgeLen * (tHat.x * tHat.x);
            Qxy += edgeLen * (tHat.x * tHat.y);
        }
    }
    if (perimeter > EPS)
    {
        Qxx /= perimeter; Qxy /= perimeter;
        Qxx -= 0.5; // Qyy = -Qxx for traceless Q
    }
    else
    {
        Qxx = 0.0;
        Qxy = 0.0;
    }
    cellQ[idx] = make_double2(Qxx, Qxy);
};

__global__ void vm_nematicForceSets_kernel(
    int *vertexCellNeighbors,
    double2 *voroCur,
    double4 *voroLastNext,
    double2 *areaPeri,
    double2 *APPref,
    int *cellType,
    int *cellVertices,
    int *cellVertexNum,
    double *typeActivities,
    double2 *forceSets,
    double2 *cellQ,
    Index2D n_idx,
    bool simpleNematic,
    double zeta,
    int nForceSets,
    double KA, double KP)
{
    unsigned int fsidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (fsidx >= nForceSets)
        return;

    // Compute quadratic energy forces
    int cellIdx1 = vertexCellNeighbors[fsidx];
    double Adiff = KA * (areaPeri[cellIdx1].x - APPref[cellIdx1].x);
    double Pdiff = KP * (areaPeri[cellIdx1].y - APPref[cellIdx1].y);
    double2 vcur = voroCur[fsidx];
    double2 vlast = make_double2(voroLastNext[fsidx].x, voroLastNext[fsidx].y);
    double2 vnext = make_double2(voroLastNext[fsidx].z, voroLastNext[fsidx].w);

    double2 dEdv;
    computeForceSetVertexModel(vcur, vlast, vnext, Adiff, Pdiff, dEdv);
    forceSets[fsidx] = dEdv;

    // Nematic forces
    double zetaEdge = simpleNematic ? zeta : typeActivities[cellType[cellIdx1]];
    double2 Q = cellQ[cellIdx1];
    double sigmaXX = -zetaEdge * Q.x;
    double sigmaXY = -zetaEdge * Q.y;
    double sigmaYY = -zetaEdge * (-Q.x);

    // Find indices for vcur and vnext
    // int vCurIdx = fsidx / 3;
    // int cellNeighs = cellVertexNum[cellIdx1];
    // int vNextInt = 0;
    // if (cellVertices[n_idx(cellNeighs - 1, cellIdx1)] != vCurIdx)
    // {
    //     for (int nn = 0; nn < cellNeighs - 1; ++nn)
    //     {
    //         if (cellVertices[n_idx(nn, cellIdx1)] == vCurIdx)
    //             vNextInt = nn + 1;
    //     }
    // }
    // int vPrevInt = (vNextInt == 0) ? cellNeighs - 1 : vNextInt - 1;
    // int vNextIdx = cellVertices[n_idx(vNextInt, cellIdx1)];
    // int vPrevIdx = cellVertices[n_idx(vPrevInt, cellIdx1)];

    const double EPS = 1e-10;

    // Edge j,j+1 (vcur to vnext)
    double2 edgeNext = make_double2(vnext.x - vcur.x, vnext.y - vcur.y);
    double edgeLenNext = sqrt(edgeNext.x * edgeNext.x + edgeNext.y * edgeNext.y);
    double2 fNext = make_double2(0.0, 0.0);
    if (edgeLenNext > EPS)
    {
        double2 nNext = make_double2(-edgeNext.y / edgeLenNext, edgeNext.x / edgeLenNext);
        fNext = make_double2(
            -edgeLenNext * (sigmaXX * nNext.x + sigmaXY * nNext.y),
            -edgeLenNext * (sigmaXY * nNext.x + sigmaYY * nNext.y)
        );
    }

    // Edge j-1,j (vprev to vcur)
    // double2 vprev = voroCur[n_idx(vPrevInt, cellIdx1)];
    double2 vprev = vlast;
    double2 edgePrev = make_double2(vcur.x - vprev.x, vcur.y - vprev.y);
    double edgeLenPrev = sqrt(edgePrev.x * edgePrev.x + edgePrev.y * edgePrev.y);
    double2 fPrev = make_double2(0.0, 0.0);
    if (edgeLenPrev > EPS)
    {
        double2 nPrev = make_double2(-edgePrev.y / edgeLenPrev, edgePrev.x / edgeLenPrev);
        fPrev = make_double2(
            -edgeLenPrev * (sigmaXX * nPrev.x + sigmaXY * nPrev.y),
            -edgeLenPrev * (sigmaXY * nPrev.x + sigmaYY * nPrev.y)
        );
    }

    // Vertex force
    forceSets[fsidx].x += 0.5 * (fNext.x + fPrev.x);
    forceSets[fsidx].y += 0.5 * (fNext.y + fPrev.y);
};

bool gpu_compute_q_tensors(
    double2 *voroCur,
    int *cellVertexNum,
    double2 *areaPeri,
    double2 *cellQ,
    Index2D &n_idx,
    int Ncells)
{
    unsigned int block_size = 128;
    if (Ncells < 128) block_size = 32;
    unsigned int nblocks = Ncells / block_size + 1;

    vm_compute_q_tensors_kernel<<<nblocks, block_size>>>(
        voroCur, cellVertexNum, areaPeri, cellQ, n_idx, Ncells
    );
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
};

bool gpu_vertexModel_nematic_force_sets(
    int *vertexCellNeighbors,
    double2 *voroCur,
    double4 *voroLastNext,
    double2 *areaPeri,
    double2 *APPref,
    int *cellType,
    int *cellVertices,
    int *cellVertexNum,
    double *typeActivities,
    double2 *forceSets,
    double2 *cellQ,
    Index2D &n_idx,
    bool simpleNematic,
    double zeta,
    int nForceSets,
    double KA, double KP)
{
    unsigned int block_size = 128;
    if (nForceSets < 128) block_size = 32;
    unsigned int nblocks = nForceSets / block_size + 1;

    vm_nematicForceSets_kernel<<<nblocks, block_size>>>(
        vertexCellNeighbors, voroCur, voroLastNext, areaPeri, APPref,
        cellType, cellVertices, cellVertexNum, typeActivities, forceSets,
        cellQ, n_idx, simpleNematic, zeta, nForceSets, KA, KP
    );
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
};

/** @} */ // end of group declaration
