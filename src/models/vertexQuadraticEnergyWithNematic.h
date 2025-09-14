#ifndef VertexQuadraticEnergyWithNematic_H
#define VertexQuadraticEnergyWithNematic_H

#include "vertexQuadraticEnergy.h"

/*! \file vertexQuadraticEnergyWithNematic.h */
//! Add nematic active force terms in the 2D Vertex model
/*!
This child class of VertexQuadraticEnergy implements active nematic forces based on cell shape anisotropy.
It replaces interfacial tension with bulk nematic stresses, computing forces from the cell's stress tensor.
*/
class VertexQuadraticEnergyWithNematic : public VertexQuadraticEnergy
{
public:
    //! initialize with random positions and set all cells to have uniform target A_0 and P_0 parameters
    VertexQuadraticEnergyWithNematic(int n, double A0, double P0, bool reprod = false, bool runSPVToInitialize = false, bool usegpu = true) : VertexQuadraticEnergy(n, A0, P0, reprod, runSPVToInitialize, usegpu)
    {
        zeta = 0; Nematic = false; simpleNematic = true; GPUcompute = usegpu;
        if (!GPUcompute)
            typeActivities.neverGPU = true;
        cellQ.resize(Ncells);
    };

    //! compute the geometry and get the forces
    virtual void computeForces();
    //! compute the quadratic energy functional
    virtual double computeEnergy();

    //! Compute the net force on vertices on the CPU with nematic forces
    virtual void computeVertexNematicForcesCPU();
    //! call gpu_force_sets kernel caller
    virtual void computeVertexNematicForceGPU();

    //! Use nematic activity
    void setUseNematic(bool use_nematic) { Nematic = use_nematic; };
    //! Set activity, with only a single value of activity
    void setActivity(double z) { zeta = z; simpleNematic = true; };
    //! Set a general vector describing activities for many cell types
    void setActivity(vector<double> zetas);
    //! Get activity
    double getActivity() { return zeta; };

protected:
    //! The value of activity for cells
    double zeta;
    //! A flag specifying whether the force calculation contains any nematic activities to compute
    bool Nematic;
    //! A flag switching between "simple" activities (single value of zeta for every cell) or not
    bool simpleNematic;
    //! A vector describing the activity, \zeta_{i} for type i
    GPUArray<double> typeActivities;
    //! The shape anisotropy tensor Q for each cell, stored as (Qxx, Qxy)
    GPUArray<double2> cellQ;

    //! Compute the Q tensors for all cells on the CPU
    void computeCellQTensorsCPU();
    //! Compute the Q tensors for all cells on the GPU
    void computeCellQTensorsGPU();
};

#endif
