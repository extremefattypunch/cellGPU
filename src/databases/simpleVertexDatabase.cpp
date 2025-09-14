#include "simpleVertexDatabase.h"
#include "baseHDF5Database.h"
#include "debuggingHelp.h"
#include "vertexModelBase.h"

simpleVertexDatabase::simpleVertexDatabase(int np, string fn,fileMode::Enum _mode) 
    : baseHDF5Database(fn,_mode)
    {
    objectName = "simpleVertexDatabase";
    N = np;
    Nc = np/2;
    timeVector.resize(1);
    boxVector.resize(4);
    intVector.resize(Nc);
    doubleVector.resize(N);
    coordinateVector.resize(2*N);
    cellCoordinateVector.resize(2*Nc);
    vertexNeighborVector.resize(3*N);
    vertexCellNeighborVector.resize(3*N);

    if(mode == fileMode::replace)
        {
        registerDatasets();
        };
    if(mode == fileMode::readwrite)
        {
        if (currentNumberOfRecords() ==0)
            registerDatasets();
        };
    logMessage(logger::verbose, "simpleVertexDatabase initialized");
    }

void simpleVertexDatabase::registerDatasets()
    {
    registerExtendableDataset<double>("time",1);
    registerExtendableDataset<double>("boxMatrix",4);

    registerExtendableDataset<int>("cellType",Nc);

    registerExtendableDataset<double>("vertexPosition", 2*N);

    registerExtendableDataset<double>("cellPosition", 2*Nc);
    registerExtendableDataset<int>("vertexVertexNeighbors", 3*N);
    registerExtendableDataset<int>("vertexCellNeighbors", 3*N);
    // registerExtendableDataset<double>("additionalData", 2*N);
    }

void simpleVertexDatabase::writeState(STATE c, double time, int rec)
    {
    if(rec >= 0)
        ERRORERROR("overwriting specific records not implemented at the moment");
    shared_ptr<vertexModelBase> s = dynamic_pointer_cast<vertexModelBase>(c);
    if (time < 0) time = s->currentTime;
    //time
    timeVector[0] = time;
    extendDataset("time", timeVector);

    //boxMatrix
    double x11,x12,x21,x22;
    s->Box->getBoxDims(x11,x12,x21,x22);
    boxVector[0]=x11;
    boxVector[1]=x12;
    boxVector[2]=x21;
    boxVector[3]=x22;
    extendDataset("boxMatrix",boxVector);

    //type
    ArrayHandle<int> h_ct(s->cellType,access_location::host,access_mode::read);
    for (int ii = 0; ii < Nc; ++ii)
        {
        int pidx = s->tagToIdx[ii];
        intVector[ii] = h_ct.data[pidx];
        }
    extendDataset("cellType",intVector);

    //position
    ArrayHandle<double2> h_p(s->returnPositions(),access_location::host,access_mode::read);
    for (int ii = 0; ii < N; ++ii)
        {
        int pidx = s->tagToIdxVertex[ii];
        coordinateVector[2*ii] = h_p.data[pidx].x;
        coordinateVector[2*ii+1] = h_p.data[pidx].y;
        }
    extendDataset("vertexPosition",coordinateVector); 

    //cellPosition
    s->getCellPositionsCPU();
    ArrayHandle<double2> h_cpos(s->cellPositions);
    for (int ii = 0; ii < Nc; ++ii)
        {
        int pidx = s->tagToIdx[ii];
        cellCoordinateVector[2*ii] = h_cpos.data[pidx].x;
        cellCoordinateVector[2*ii+1] = h_cpos.data[pidx].y;
        }
    extendDataset("cellPosition",cellCoordinateVector);
    
    //vertexVertexNeighbors
    ArrayHandle<int> h_vn(s->vertexNeighbors,access_location::host,access_mode::read);
   
    //vertexCellNeighbors
    ArrayHandle<int> h_vcn(s->vertexCellNeighbors,access_location::host,access_mode::read);
    for (int vv = 0; vv < N; ++vv)
        {
        int vertexIndex = s->tagToIdxVertex[vv];
        for (int ii = 0 ;ii < 3; ++ii)
            {
            vertexNeighborVector[3*vv+ii] = s->idxToTagVertex[h_vn.data[3*vertexIndex+ii]];
            vertexCellNeighborVector[3*vv+ii] = s->idxToTag[h_vcn.data[3*vertexIndex+ii]];
            };
        };
    extendDataset("vertexVertexNeighbors",vertexNeighborVector);
    extendDataset("vertexCellNeighbors",vertexCellNeighborVector);
  
    }

void simpleVertexDatabase::readState(STATE c, int rec, bool geometry)
    {
    shared_ptr<vertexModelBase> t = dynamic_pointer_cast<vertexModelBase>(c);

    readDataset("time",timeVector,rec);
    t->currentTime = timeVector[0];

    readDataset("boxMatrix",boxVector,rec);
    t->Box->setGeneral(boxVector[0],boxVector[1],boxVector[2],boxVector[3]);

    readDataset("cellType",intVector,rec);
    ArrayHandle<int> h_ct(t->cellType,access_location::host,access_mode::overwrite);
    for (int idx = 0; idx < Nc; ++idx)
        {
        h_ct.data[idx]=intVector[idx];
        };

    readDataset("vertexPosition",coordinateVector,rec);
    ArrayHandle<double2> h_p(t->returnPositions(),access_location::host,access_mode::overwrite);
    for (int idx = 0; idx < N; ++idx)
        {
        h_p.data[idx].x = coordinateVector[2*idx];
        h_p.data[idx].y = coordinateVector[2*idx+1];
        };

    readDataset("cellPosition",cellCoordinateVector,rec);
    ArrayHandle<double2> h_cp(t->cellPositions,access_location::host,access_mode::overwrite);
    for (int idx = 0; idx < Nc; ++idx)
        {
        h_cp.data[idx].x = cellCoordinateVector[2*idx];
        h_cp.data[idx].y = cellCoordinateVector[2*idx+1];
        };

    // Set velocities to zero since not stored
    ArrayHandle<double2> h_v(t->returnVelocities(),access_location::host,access_mode::overwrite);
    for (int idx = 0; idx < N; ++idx)
        {
        h_v.data[idx] = make_double2(0.0, 0.0);
        };

    readDataset("vertexVertexNeighbors",vertexNeighborVector,rec);
    ArrayHandle<int> h_vn(t->vertexNeighbors,access_location::host,access_mode::overwrite);
    for (int vv = 0; vv < N; ++vv)
        {
        for (int ii = 0; ii < 3; ++ii)
            {
            h_vn.data[3*vv + ii] = vertexNeighborVector[3*vv + ii];
            };
        };

    readDataset("vertexCellNeighbors",vertexCellNeighborVector,rec);
    ArrayHandle<int> h_vcn(t->vertexCellNeighbors,access_location::host,access_mode::overwrite);
    for (int vv = 0; vv < N; ++vv)
        {
        for (int ii = 0; ii < 3; ++ii)
            {
            h_vcn.data[3*vv + ii] = vertexCellNeighborVector[3*vv + ii];
            };
        };

    // Assume tags are ordered 0 to Nc-1 for cells and 0 to N-1 for vertices
    t->idxToTag.resize(Nc);
    t->tagToIdx.resize(Nc);
    for (int i = 0; i < Nc; ++i)
        {
        t->idxToTag[i] = i;
        t->tagToIdx[i] = i;
        };
    t->idxToTagVertex.resize(N);
    t->tagToIdxVertex.resize(N);
    for (int i = 0; i < N; ++i)
        {
        t->idxToTagVertex[i] = i;
        t->tagToIdxVertex[i] = i;
        };

    //by default, compute the triangulation and geometrical information
    if(geometry)
        {
        t->computeGeometry();
        };
    }

unsigned long simpleVertexDatabase::currentNumberOfRecords()
    {
    return getDatasetDimensions("time");
    }
