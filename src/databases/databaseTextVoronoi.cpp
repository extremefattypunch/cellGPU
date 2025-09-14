#include "databaseTextVoronoi.h"
/*! \file databaseTextVoronoi.cpp */

DatabaseTextVoronoi::DatabaseTextVoronoi(string fn, fileMode::Enum _mode)
    {
    filename = fn;
    mode = _mode;
    switch(mode)
        {
            case fileMode::readonly:
            inputFile.open(filename);
            break;
            case fileMode::replace:
            outputFile.open(filename);
            break;
            case fileMode::readwrite:
            outputFile.open(filename, ios::app);
            break;
        default:
            ;
        };
    };

void DatabaseTextVoronoi::writeState(STATE s, double time, int rec)
    {
    if (rec != -1)
        {
        printf("writing to the middle of text files not supported\n");
        throw std::exception();
        };
        
    int N = s->getNumberOfDegreesOfFreedom();
    if (time < 0) time = s->currentTime;

    // Ensure geometry is computed to populate voroCur
    s->computeGeometry();

    outputFile << "t=" << time << "\n";

    ArrayHandle<double2> h_pos(s->cellPositions,access_location::host,access_mode::read);
    ArrayHandle<int> h_nn(s->neighborNum,access_location::host,access_mode::read);
    ArrayHandle<double2> h_voro(s->getVoroCur(),access_location::host,access_mode::read);

    for (int ii = 0; ii < N; ++ii)
        {
        int pidx = s->tagToIdx[ii];
        double2 cellpos = h_pos.data[pidx];
        outputFile << "cell=" << cellpos.x << "," << cellpos.y << ",vertices=[";
        int neighs = h_nn.data[pidx];
        for (int nn = 0; nn < neighs; ++nn)
            {
            double2 vrel = h_voro.data[s->getNIdx()(nn, pidx)];
            double2 vabs;
            vabs.x = vrel.x + cellpos.x;
            vabs.y = vrel.y + cellpos.y;
            if (nn > 0) outputFile << ",";
            outputFile << vabs.x << "," << vabs.y;
            }
        outputFile << "]\n";
        };
    };

void DatabaseTextVoronoi::readState(STATE s, int rec, bool geometry)
    {
    printf("Reading from a text database currently not supported\n");
    throw std::exception();
    };
