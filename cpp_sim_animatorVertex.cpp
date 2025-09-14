#include "std_include.h"

#include "cuda_runtime.h"
#include "cuda_profiler_api.h"

#include "Simulation.h"
#include "simpleVertexDatabase.h"
#include "periodicBoundaries.h"
#include "vertexQuadraticEnergy.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <cmath>

int main(int argc, char*argv[])
{
    // Default parameters aligned with paper's supplementary Fig. 5
    int numpts = 100; // number of cells
    int USE_GPU = 0; // 0 or greater uses a gpu, any negative number runs on the cpu
    int c;
    int tSteps = 600000; // number of time steps to run after initialization
    int initSteps = 0; // number of initialization steps 
    double dt = 0.01; // the time step size
    double p0 = 3.95;  // the preferred perimeter (fluid phase)
    double a0 = 1.0;  // the preferred area
    double kp = 0.02;  // perimeter elasticity modulus
    double zeta_c = -0.4;  // contractile nematic activity
    double zeta_e = 0.1;  // extensile nematic activity
    double alpha_c = 0.00;  // polar motility for contractile cells
    double alpha_e = 0.00;  // polar motility for contractile cells
    double Dr = 0.0;  // rotational diffusion (deterministic)
    int cluster_size = numpts/2; // central cluster size
    int S = 3;   // 3 for analysis (HDF5 output only)
    int fIdx = 0;
    int fps = 30;

    // The defaults can be overridden from the command line
    while ((c = getopt(argc, argv, "n:g:t:i:p:a:k:c:l:d:s:f:e:")) != -1)
        switch (c)
        {
            case 'n': numpts = atoi(optarg); break;
            case 't': tSteps = atof(optarg); break;
            case 'g': USE_GPU = atoi(optarg); break;
            case 'i': initSteps = atoi(optarg); break;
            case 'p': p0 = atof(optarg); break;
            case 'a': a0 = atof(optarg); break;
            case 'k': kp = atof(optarg); break;
            case 'z': zeta_c = atof(optarg); break;
            case 'Z': zeta_e = atof(optarg); break;
            case 'd': Dr = atof(optarg); break;
            case 's': S = atoi(optarg); break;
            case 'f': fps = atoi(optarg); break;
            case 'e': dt = atof(optarg); break;
            case '?':
                if (optopt == 'c')
                    std::cerr << "Option -" << optopt << " requires an argument.\n";
                return 1;
            default:
                abort();
        };

    bool reproducible = false; // ignored, but set to false

    // construct database filename
    char dbname[256];
    sprintf(dbname, "ActivePolarNematicVertex_c%.3f_e%.3f.nc", zeta_c, zeta_e);

    // output video name
    char viname[256];
    sprintf(viname, "../visualizationTools/ActivePolarNematicVertex_c%.3f_e%.3f.avi", zeta_c, zeta_e);

    // open the database
    int Nv = numpts * 2; // approximate number of vertices
    simpleVertexDatabase db(Nv, dbname, fileMode::readonly);

    unsigned long numFrames = tSteps;

    // assume square box with side length sqrt(N), since total area ~ N * a0 = N
    double L = sqrt(static_cast<double>(numpts));
    shared_ptr<periodicBoundaries> Box = make_shared<periodicBoundaries>(L, 0.0, 0.0, L);

    shared_ptr<VertexQuadraticEnergy> model = make_shared<VertexQuadraticEnergy>(numpts, a0, p0, reproducible, false);

    // setup rendering
    const int ws = 800;
    double scale = ws / L;

    // setup video writer
    cv::VideoWriter writer(viname, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(ws, ws));

    cv::Mat img;

    for (unsigned long frame = 0; frame < numFrames; ++frame)
    {
        db.readState(model, frame, false);



        img = cv::Mat(ws, ws, CV_8UC3, cv::Scalar(255, 255, 255));

        // Draw lines between vertices
        ArrayHandle<double2> h_pos(model->returnPositions(), access_location::host, access_mode::read);
        ArrayHandle<int> h_vn(model->vertexNeighbors, access_location::host, access_mode::read);

        std::vector<cv::Point> vertexPixels(Nv);
        for (int i = 0; i < Nv; ++i)
        {
            double2 p = h_pos.data[i];
            Box->putInBoxReal(p);
            double px = p.x * scale;
            double py = p.y * scale; // Note: y is not flipped; adjust if needed
            vertexPixels[i] = cv::Point(static_cast<int>(std::round(px)), static_cast<int>(std::round(py)));
        }

        for (int v = 0; v < Nv; ++v)
        {
            cv::Point p1 = vertexPixels[v];
            double2 pd1 = make_double2(p1.x / scale, p1.y / scale);

            for (int nn = 0; nn < 3; ++nn)
            {
                int n = h_vn.data[3 * v + nn];
                cv::Point p2_base = vertexPixels[n];
                double2 pd2 = make_double2(p2_base.x / scale, p2_base.y / scale);

                double dx = pd2.x - pd1.x;
                double dy = pd2.y - pd1.y;
                dx -= L * std::round(dx / L);
                dy -= L * std::round(dy / L);

                double2 pd2_min = pd1 + make_double2(dx, dy);
                cv::Point p2_min(static_cast<int>(std::round(pd2_min.x * scale)), static_cast<int>(std::round(pd2_min.y * scale)));

                cv::line(img, p1, p2_min, cv::Scalar(0, 0, 0), 1);
            }
        }

        // Mark cell positions with colored dots
        ArrayHandle<double2> h_cp(model->cellPositions, access_location::host, access_mode::read);
        ArrayHandle<int> h_ct(model->cellType, access_location::host, access_mode::read);

        for (int cc = 0; cc < numpts; ++cc)
        {
            double2 p = h_cp.data[cc];
            Box->putInBoxReal(p);
            int px = static_cast<int>(std::round(p.x * scale));
            int py = static_cast<int>(std::round(p.y * scale));

            cv::Scalar color = (h_ct.data[cc] == 0) ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0); // red for type 0, blue for others

            cv::circle(img, cv::Point(px, py), 3, color, -1);
        }

        writer.write(img);
    }

    writer.release();

    return 0;
};
