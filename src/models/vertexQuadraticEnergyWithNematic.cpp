#include "vertexQuadraticEnergyWithNematic.h"
#include "vertexQuadraticEnergyWithNematic.cuh"
#include "vertexQuadraticEnergy.cuh"
/*! \file vertexQuadraticEnergyWithNematic.cpp */

/*!
This function defines a vector, \zeta_{i}, describing the activity for cell type i.
It sets the activity vector and updates the flag for computeForces to use nematic force computations.
\pre The vector has n elements, where n is the number of cell types (0, 1, ..., n-1).
*/
void VertexQuadraticEnergyWithNematic::setActivity(vector<double> zetas)
{
    simpleNematic = false;
    typeActivities.resize(zetas.size());
    ArrayHandle<double> activities(typeActivities, access_location::host, access_mode::overwrite);
    for (int i = 0; i < zetas.size(); ++i)
    {
        activities.data[i] = zetas[i];
    }
};

/*!
Compute Q tensors for all cells on the CPU
*/
void VertexQuadraticEnergyWithNematic::computeCellQTensorsCPU()
{
    ArrayHandle<double2> h_vc(voroCur, access_location::host, access_mode::read);
    ArrayHandle<int> h_cvn(cellVertexNum, access_location::host, access_mode::read);
    ArrayHandle<int> h_cv(cellVertices, access_location::host, access_mode::read);
    ArrayHandle<double2> h_AP(AreaPeri, access_location::host, access_mode::read);
    ArrayHandle<double2> h_Q(cellQ, access_location::host, access_mode::overwrite);

    const double EPS = 1e-10;

    for (int ii = 0; ii < Ncells; ++ii)
    {
        int cellIdx = tagToIdx[ii];
        int nVerts = h_cvn.data[cellIdx];
        double perimeter = h_AP.data[cellIdx].y;
        double Qxx = 0.0, Qxy = 0.0;

        for (int nn = 0; nn < nVerts; ++nn)
        {
            int vIdx = h_cv.data[n_idx(nn, cellIdx)];
            int vNextIdx = h_cv.data[n_idx((nn + 1) % nVerts, cellIdx)];
            double2 vCur = h_vc.data[n_idx(nn, cellIdx)];
            double2 vNext = h_vc.data[n_idx((nn + 1) % nVerts, cellIdx)];
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
            Qxx -= 0.5; // subtract (1/2)I for traceless Q
        }
        else
        {
            Qxx = 0.0;
            Qxy = 0.0;
        }
        h_Q.data[cellIdx] = make_double2(Qxx, Qxy);
    }
};

/*!
Compute Q tensors for all cells on the GPU
*/
void VertexQuadraticEnergyWithNematic::computeCellQTensorsGPU()
{
    ArrayHandle<double2> d_vc(voroCur, access_location::device, access_mode::read);
    ArrayHandle<int> d_cvn(cellVertexNum, access_location::device, access_mode::read);
    ArrayHandle<double2> d_AP(AreaPeri, access_location::device, access_mode::read);
    ArrayHandle<double2> d_Q(cellQ, access_location::device, access_mode::overwrite);

    gpu_compute_q_tensors(
        d_vc.data,
        d_cvn.data,
        d_AP.data,
        d_Q.data,
        n_idx,
        Ncells
    );
};

/*!
Compute forces on vertices, including nematic contributions
*/
void VertexQuadraticEnergyWithNematic::computeForces()
{
    if (forcesUpToDate)
        return;
    forcesUpToDate = true;
    computeGeometry();
    if (Nematic)
    {
        if (GPUcompute)
        {
            // computeCellQTensorsGPU();
            // computeVertexNematicForceGPU();
            computeCellQTensorsCPU();
            computeVertexNematicForceGPU();
        }
        else
        {
            computeCellQTensorsCPU();
            computeVertexNematicForcesCPU();
        }
    }
    else
    {
        if (GPUcompute)
            computeForcesGPU();
        else
            computeForcesCPU();
    }
};

/*!
Compute the net force on each vertex on the CPU with nematic forces
*/
void VertexQuadraticEnergyWithNematic::computeVertexNematicForcesCPU()
{
    ArrayHandle<int> h_vcn(vertexCellNeighbors, access_location::host, access_mode::read);
    ArrayHandle<double2> h_vc(voroCur, access_location::host, access_mode::read);
    ArrayHandle<double4> h_vln(voroLastNext, access_location::host, access_mode::read);
    ArrayHandle<double2> h_AP(AreaPeri, access_location::host, access_mode::read);
    ArrayHandle<double2> h_APpref(AreaPeriPreferences, access_location::host, access_mode::read);
    ArrayHandle<int> h_ct(cellType, access_location::host, access_mode::read);
    ArrayHandle<int> h_cv(cellVertices, access_location::host, access_mode::read);
    ArrayHandle<int> h_cvn(cellVertexNum, access_location::host, access_mode::read);
    ArrayHandle<double> h_ta(typeActivities, access_location::host, access_mode::read);
    ArrayHandle<double2> h_Q(cellQ, access_location::host, access_mode::read);
    ArrayHandle<double2> h_fs(vertexForceSets, access_location::host, access_mode::overwrite);
    ArrayHandle<double2> h_f(vertexForces, access_location::host, access_mode::overwrite);

    const double EPS = 1e-10;

    // Compute vertex force contributions from each cell
    for (int fsidx = 0; fsidx < Nvertices * 3; ++fsidx)
    {
        int cellIdx1 = h_vcn.data[fsidx]; //vertexCellNeighbors[3*i+k],vertex i, neighbor cell k(3 each vertex)
        double Adiff = KA * (h_AP.data[cellIdx1].x - h_APpref.data[cellIdx1].x);
        double Pdiff = KP * (h_AP.data[cellIdx1].y - h_APpref.data[cellIdx1].y);
        double2 vcur = h_vc.data[fsidx]; //voroCur.data[n_idx(nn,i)],vertex nn(x,y coor) of cell i(CCW order)
        double2 vlast = make_double2(h_vln.data[fsidx].x, h_vln.data[fsidx].y);
        double2 vnext = make_double2(h_vln.data[fsidx].z, h_vln.data[fsidx].w);

        // Quadratic energy forces
        double2 dEdv;
        computeForceSetVertexModel(vcur, vlast, vnext, Adiff, Pdiff, dEdv);
        h_fs.data[fsidx] = dEdv;

        // Nematic forces
        double zetaEdge = simpleNematic ? zeta : h_ta.data[h_ct.data[cellIdx1]];
        double2 Q = h_Q.data[cellIdx1];
        // Construct stress tensor: sigma = -zeta * Q
        double sigmaXX = -zetaEdge * Q.x;
        double sigmaXY = -zetaEdge * Q.y; //sigmaXY=sigmaYX
        double sigmaYY = -zetaEdge * (-Q.x); // Qyy = -Qxx for traceless Q

        // // Find indices for vcur and vnext
        // int vCurIdx = fsidx / 3;//nearest multiple of 3
        // int cellNeighs = h_cvn.data[cellIdx1];
        // int vNextInt = 0;
        // if (h_cv.data[n_idx(cellNeighs - 1, cellIdx1)] != vCurIdx)
        // {
        //     for (int nn = 0; nn < cellNeighs - 1; ++nn)
        //     {
        //         if (h_cv.data[n_idx(nn, cellIdx1)] == vCurIdx)
        //             vNextInt = nn + 1;
        //     }
        // }
        // int vPrevInt = (vNextInt == 0) ? cellNeighs - 1 : vNextInt - 1;
        // int vNextIdx = h_cv.data[n_idx(vNextInt, cellIdx1)];
        // int vPrevIdx = h_cv.data[n_idx(vPrevInt, cellIdx1)];

        // Edge j,j+1 (vcur to vnext)
        double2 edgeNext = make_double2(vnext.x - vcur.x, vnext.y - vcur.y);
        double edgeLenNext = sqrt(edgeNext.x * edgeNext.x + edgeNext.y * edgeNext.y);
        double2 fNext = make_double2(0.0, 0.0);
        if (edgeLenNext > EPS)
        {
            double2 nNext = make_double2(-edgeNext.y / edgeLenNext, edgeNext.x / edgeLenNext); // Outward normal
            fNext = make_double2(
                -edgeLenNext * (sigmaXX * nNext.x + sigmaXY * nNext.y),
                -edgeLenNext * (sigmaXY * nNext.x + sigmaYY * nNext.y)
            );
        }

        // Edge j-1,j (vprev to vcur)
        // double2 vprev = h_vc.data[n_idx(vPrevInt, cellIdx1)];
        double2 vprev=vlast;
        double2 edgePrev = make_double2(vcur.x - vprev.x, vcur.y - vprev.y);
        double edgeLenPrev = sqrt(edgePrev.x * edgePrev.x + edgePrev.y * edgePrev.y);
        double2 fPrev = make_double2(0.0, 0.0);
        if (edgeLenPrev > EPS)
        {
            double2 nPrev = make_double2(-edgePrev.y / edgeLenPrev, edgePrev.x / edgeLenPrev); // Outward normal
            fPrev = make_double2(
                -edgeLenPrev * (sigmaXX * nPrev.x + sigmaXY * nPrev.y),
                -edgeLenPrev * (sigmaXY * nPrev.x + sigmaYY * nPrev.y)
            );
        }

        // Vertex force: (f_{j,j+1} + f_{j-1,j}) / 2
        h_fs.data[fsidx].x += 0.5 * (fNext.x + fPrev.x);
        h_fs.data[fsidx].y += 0.5 * (fNext.y + fPrev.y);
    }

    // Sum force sets to get net force on each vertex
    for (int v = 0; v < Nvertices; ++v)
    {
        double2 ftemp = make_double2(0.0, 0.0);
        for (int ff = 0; ff < 3; ++ff)
        {
            ftemp.x += h_fs.data[3 * v + ff].x;
            ftemp.y += h_fs.data[3 * v + ff].y;
        }
        h_f.data[v] = ftemp;
    }
};

/*!
Compute the quadratic energy functional (currently unimplemented)
*/
double VertexQuadraticEnergyWithNematic::computeEnergy()
{
    if (!forcesUpToDate)
        computeForces();
    printf("computeEnergy function for VertexQuadraticEnergyWithNematic not written. Very sorry\n");
    throw std::exception();
    return 0;
};

/*!
Compute nematic forces on the GPU
*/
void VertexQuadraticEnergyWithNematic::computeVertexNematicForceGPU()
{
    ArrayHandle<int> d_vcn(vertexCellNeighbors, access_location::device, access_mode::read);
    ArrayHandle<double2> d_vc(voroCur, access_location::device, access_mode::read);
    ArrayHandle<double4> d_vln(voroLastNext, access_location::device, access_mode::read);
    ArrayHandle<double2> d_AP(AreaPeri, access_location::device, access_mode::read);
    ArrayHandle<double2> d_APpref(AreaPeriPreferences, access_location::device, access_mode::read);
    ArrayHandle<int> d_ct(cellType, access_location::device, access_mode::read);
    ArrayHandle<int> d_cv(cellVertices, access_location::device, access_mode::read);
    ArrayHandle<int> d_cvn(cellVertexNum, access_location::device, access_mode::read);
    ArrayHandle<double> d_ta(typeActivities, access_location::device, access_mode::read);
    ArrayHandle<double2> d_Q(cellQ, access_location::device, access_mode::read);
    ArrayHandle<double2> d_fs(vertexForceSets, access_location::device, access_mode::overwrite);
    ArrayHandle<double2> d_f(vertexForces, access_location::device, access_mode::overwrite);

    int nForceSets = Nvertices * 3;
    gpu_vertexModel_nematic_force_sets(
        d_vcn.data,
        d_vc.data,
        d_vln.data,
        d_AP.data,
        d_APpref.data,
        d_ct.data,
        d_cv.data,
        d_cvn.data,
        d_ta.data,
        d_fs.data,
        d_Q.data,
        n_idx,
        simpleNematic,
        zeta,
        nForceSets,
        KA, KP
    );

    gpu_avm_sum_force_sets(d_fs.data, d_f.data, Nvertices);
};
