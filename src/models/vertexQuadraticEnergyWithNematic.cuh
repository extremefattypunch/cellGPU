#ifndef __vertexQuadraticEnergyWithNematic_CUH__
#define __vertexQuadraticEnergyWithNematic_CUH__

#include "functions.h"
#include "indexer.h"

/*!
 \file vertexQuadraticEnergyWithNematic.cuh
A file providing an interface to the relevant cuda calls for the VertexQuadraticEnergyWithNematic class
*/

/** @defgroup vmKernels vertex model Kernels
 * @{
 * \brief CUDA kernels and callers for 2D vertex models
 */

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
    double KA, double KP);

bool gpu_compute_q_tensors(
    double2 *voroCur,
    int *cellVertexNum,
    double2 *areaPeri,
    double2 *cellQ,
    Index2D &n_idx,
    int Ncells);

#endif
